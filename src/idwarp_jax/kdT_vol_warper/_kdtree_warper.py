"""Differentiable, batched JAX implementation of IDWarp's fast kd-tree warp."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from idwarp_jax import normal, rotation
from idwarp_jax.warp import compute_idwarp_reference_length

from ._route_cache import route_fingerprint, load_routes, save_routes

from ._kdtree_preprocessing import (
    KdTreeRoutes,
    build_idwarp_tree,
    expand_route_rows,
    next_power_of_two,
    precompute_routes,
)


def _prepare_aggregation_topology(tree, surface_area, area_weighted):
    scalar_dtype = np.asarray(surface_area).dtype
    leaf_nodes = np.flatnonzero(tree.left < 0).astype(np.int32)
    leaf_counts = (tree.end[leaf_nodes] - tree.start[leaf_nodes]).astype(np.int32)
    leaf_width = int(np.max(leaf_counts))
    leaf_surface_ids = np.zeros((len(leaf_nodes), leaf_width), dtype=np.int32)
    leaf_weights = np.zeros((len(leaf_nodes), leaf_width), dtype=scalar_dtype)
    for row, node_id in enumerate(leaf_nodes):
        positions = np.arange(tree.start[node_id], tree.end[node_id])
        count = len(positions)
        leaf_surface_ids[row, :count] = tree.permutation[positions]
        if area_weighted:
            leaf_weights[row, :count] = surface_area[leaf_surface_ids[row, :count]]
        else:
            leaf_weights[row, :count] = 1.0
    leaf_denominators = (
        tree.area[leaf_nodes].astype(scalar_dtype)
        if area_weighted
        else leaf_counts.astype(scalar_dtype)
    )

    levels = []
    for level in range(int(np.max(tree.level)) - 1, 0, -1):
        node_ids = np.flatnonzero((tree.level == level) & (tree.left >= 0)).astype(
            np.int32
        )
        if not len(node_ids):
            continue
        left_ids = tree.left[node_ids].astype(np.int32)
        right_ids = tree.right[node_ids].astype(np.int32)
        if area_weighted:
            left_counts = tree.area[left_ids].astype(scalar_dtype)
            right_counts = tree.area[right_ids].astype(scalar_dtype)
            parent_counts = tree.area[node_ids].astype(scalar_dtype)
        else:
            left_counts = (tree.end[left_ids] - tree.start[left_ids]).astype(scalar_dtype)
            right_counts = (tree.end[right_ids] - tree.start[right_ids]).astype(scalar_dtype)
            parent_counts = (tree.end[node_ids] - tree.start[node_ids]).astype(scalar_dtype)
        levels.append(
            (node_ids, left_ids, right_ids, left_counts, right_counts, parent_counts)
        )
    return leaf_nodes, leaf_surface_ids, leaf_weights, leaf_denominators, levels


def _device_aggregation_topology(tree, surface_area, area_weighted, device):
    leaf_nodes, leaf_ids, leaf_weights, leaf_denominators, levels = (
        _prepare_aggregation_topology(tree, surface_area, area_weighted)
    )
    return (
        jax.device_put(jnp.asarray(leaf_nodes), device),
        jax.device_put(jnp.asarray(leaf_ids), device),
        jax.device_put(jnp.asarray(leaf_weights), device),
        jax.device_put(jnp.asarray(leaf_denominators), device),
        [
            tuple(jax.device_put(jnp.asarray(array), device) for array in arrays)
            for arrays in levels
        ],
    )


def _route_buckets(routes: KdTreeRoutes) -> dict[int, np.ndarray]:
    groups: dict[int, list[int]] = defaultdict(list)
    for row_id, count in enumerate(routes.expanded_counts):
        groups[next_power_of_two(int(count))].append(row_id)
    return {
        width: np.asarray(row_ids, dtype=np.int64)
        for width, row_ids in sorted(groups.items())
    }


def make_kdtree_deformer(
    mesh,
    reference_volume_points,
    device,
    rotator,
    *,
    LdefFact=100.0,
    bucket_size=8,
    err_tol=5.0e-4,
    aExp=3.0,
    bExp=5.0,
    alpha=0.25,
    warp_eps=1.0e-6,
    normal_eps=1.0e-6,
    rotation_eps=1.0e-6,
    symmetry_tolerance=1.0e-6,
    surface_face_type=1,
    useRotations=True,
    zeroCornerRotations=True,
    cornerAngle=30.0,
    target_batch_interactions=1_000_000,
    max_batch_points=4096,
    area_weighted_nodes=True,
    route_cache="none",
    cached_routes=None,
    progress=False,
    tree=None,
    routes=None,
):
    """Construct an exact-y-symmetry fast-IDWarp prototype.

    Tree traversal and its discrete decisions are fixed preprocessing. The
    online normal, M/b, tree aggregation, sparse warp, and VJP are JAX based.
    """
    setup_started = time.perf_counter()
    reference_volume_points = np.asarray(reference_volume_points)
    jax_dtype = jnp.asarray(reference_volume_points).dtype
    host_dtype = np.dtype(jax_dtype)
    points = np.asarray(mesh["points"], dtype=host_dtype)
    reference_volume_points = np.asarray(
        reference_volume_points, dtype=host_dtype
    )
    if reference_volume_points.ndim != 2 or reference_volume_points.shape[1] != 3:
        raise ValueError("reference_volume_points must have shape (n, 3)")
    if route_cache not in {"none", "host"}:
        raise ValueError("route_cache must be 'none' or 'host'")

    wall_faces = [
        np.asarray(face)
        for face, face_type in zip(mesh["faces"]["points"], mesh["faces"]["type"])
        if face_type == surface_face_type
    ]
    face_sizes = np.asarray([len(face) for face in wall_faces], dtype=np.int32)
    global_conn = np.concatenate(wall_faces)
    wall_ids = np.unique(global_conn)
    conn = np.searchsorted(wall_ids, global_conn).astype(np.int32)
    wall_points = np.asarray(points[wall_ids], dtype=host_dtype)

    Xs0 = jax.device_put(jnp.asarray(wall_points, dtype=jax_dtype), device)
    normal_topology = normal.prepare_normal_topology(conn, face_sizes)
    normals0, Ai = normal.compute_node_normals_from_topology(
        Xs0, normal_topology, eps=normal_eps
    )
    normals0.block_until_ready()
    surface_area = np.asarray(Ai, dtype=host_dtype)
    Ldef0 = compute_idwarp_reference_length(Xs0)
    Ldef0.block_until_ready()
    ldef = float(np.asarray(Ldef0)) * float(LdefFact)

    corner_mask = normal.detect_corner_nodes_from_topology(
        Xs0,
        normal_topology,
        corner_angle=cornerAngle,
        zero_corner_rotations=zeroCornerRotations,
    )
    on_symmetry = jnp.abs(Xs0[:, 1]) < symmetry_tolerance

    tree_started = time.perf_counter()
    if tree is None:
        if progress:
            print("[setup] building surface kd-tree and error table", flush=True)
        tree = build_idwarp_tree(
            wall_points,
            surface_area,
            bucket_size=bucket_size,
            ldef=ldef,
            a_exp=aExp,
            b_exp=bExp,
            alpha=alpha,
            distance_eps=warp_eps,
        )
    tree_seconds = time.perf_counter() - tree_started
    if progress:
        print(
            f"[setup] tree complete: {len(tree.start):,} nodes, "
            f"{tree_seconds:.3f} s",
            flush=True,
        )

    routes_started = time.perf_counter()
    cache_path = None
    routes_cache_hit = False
    if routes is None and cached_routes is not None:
        cache_path = Path(cached_routes) / "routes.npz"
        fingerprint = route_fingerprint(
            reference_volume_points, wall_points, surface_area, tree,
            (True, ldef, aExp, bExp, alpha, err_tol, warp_eps),
        )
        routes = load_routes(cache_path, fingerprint)
        routes_cache_hit = routes is not None
        if progress:
            status = "loaded" if routes_cache_hit else "missing or stale"
            print(f"[setup] routes cache {status}: {cache_path}", flush=True)
    if routes is None:
        if progress:
            print("[setup] computing fixed volume-point frontiers", flush=True)
        routes = precompute_routes(
            reference_volume_points,
            wall_points,
            surface_area,
            tree,
            exact_y_symmetry=True,
            ldef=ldef,
            a_exp=aExp,
            b_exp=bExp,
            alpha=alpha,
            err_tol=err_tol,
            distance_eps=warp_eps,
            progress=progress,
        )
        if cache_path is not None:
            save_routes(cache_path, fingerprint, routes)
            if progress:
                print(f"[setup] saved routes cache: {cache_path}", flush=True)
    routes_seconds = time.perf_counter() - routes_started
    if progress:
        print(
            f"[setup] routes complete: {routes.nbytes / 2**30:.3f} GiB, "
            f"{routes_seconds:.3f} s",
            flush=True,
        )
    if routes.num_volume_points != len(reference_volume_points):
        raise ValueError("routes do not match reference_volume_points")

    route_buckets = _route_buckets(routes)
    n_surface = len(wall_points)
    n_tree = len(tree.start)
    source_X0 = jax.device_put(
        jnp.asarray(
            np.concatenate((wall_points, tree.center), axis=0), dtype=jax_dtype
        ),
        device,
    )
    source_area = jax.device_put(
        jnp.asarray(
            np.concatenate((surface_area, tree.area), axis=0), dtype=jax_dtype
        ),
        device,
    )
    reflection_sign = jax.device_put(
        jnp.asarray([1.0, -1.0, 1.0], dtype=jax_dtype), device
    )
    (
        leaf_nodes,
        leaf_surface_ids,
        leaf_weights,
        leaf_denominators,
        aggregation_levels,
    ) = _device_aggregation_topology(
        tree, surface_area, area_weighted_nodes, device
    )

    def surface_to_MB_raw(surface_points):
        Xs = surface_points.at[on_symmetry, 1].set(0.0)
        if not useRotations:
            return rotation.get_MB_no_rotation(Xs0, Xs)
        normals, _ = normal.compute_node_normals_from_topology(
            Xs, normal_topology, eps=normal_eps
        )
        return rotation.get_MB_rotation(
            Xs0,
            Xs,
            normals0,
            normals,
            eps=rotation_eps,
            corner_mask=corner_mask,
        )

    def aggregate_nodes_raw(surface_M, surface_b):
        gathered_M = surface_M[leaf_surface_ids]
        gathered_b = surface_b[leaf_surface_ids]
        leaf_M = jnp.sum(
            leaf_weights[..., None, None] * gathered_M, axis=1
        ) / leaf_denominators[:, None, None]
        leaf_b = jnp.sum(
            leaf_weights[..., None] * gathered_b, axis=1
        ) / leaf_denominators[:, None]
        node_M = jnp.zeros((n_tree, 3, 3), dtype=surface_M.dtype)
        node_b = jnp.zeros((n_tree, 3), dtype=surface_b.dtype)
        node_M = node_M.at[leaf_nodes].set(leaf_M)
        node_b = node_b.at[leaf_nodes].set(leaf_b)
        for (
            node_ids,
            left_ids,
            right_ids,
            left_sizes,
            right_sizes,
            parent_sizes,
        ) in aggregation_levels:
            parent_M = (
                node_M[left_ids] * left_sizes[:, None, None]
                + node_M[right_ids] * right_sizes[:, None, None]
            ) / parent_sizes[:, None, None]
            parent_b = (
                node_b[left_ids] * left_sizes[:, None]
                + node_b[right_ids] * right_sizes[:, None]
            ) / parent_sizes[:, None]
            node_M = node_M.at[node_ids].set(parent_M)
            node_b = node_b.at[node_ids].set(parent_b)
        return node_M, node_b

    surface_to_MB = jax.jit(surface_to_MB_raw, device=device)
    aggregate_nodes = jax.jit(aggregate_nodes_raw, device=device)

    def warp_batch_raw(source_M, source_b, volume_points, encoded_ids, pitch):
        valid = encoded_ids != 0
        mirrored = encoded_ids < 0
        source_ids = jnp.maximum(jnp.abs(encoded_ids) - 1, 0)
        selected_X0 = source_X0[source_ids]
        selected_area = source_area[source_ids]
        selected_M = source_M[source_ids]
        selected_b = source_b[source_ids]

        volume_expanded = volume_points[:, None, :]
        evaluation_points = jnp.where(
            mirrored[..., None],
            volume_expanded * reflection_sign,
            volume_expanded,
        )
        difference = evaluation_points - selected_X0
        distance = jnp.sqrt(jnp.sum(difference * difference, axis=2) + warp_eps)
        ratio = jnp.asarray(ldef, dtype=distance.dtype) / distance
        weights = selected_area * (
            ratio**aExp + jnp.asarray(alpha**bExp, dtype=distance.dtype) * ratio**bExp
        )
        weights = jnp.where(valid, weights, 0.0)

        suggested = (
            jnp.einsum("bkij,bkj->bki", selected_M, evaluation_points)
            + selected_b
            - evaluation_points
        )
        suggested = jnp.where(
            mirrored[..., None], suggested * reflection_sign, suggested
        )
        numerator = jnp.sum(weights[..., None] * suggested, axis=1)
        denominator = jnp.sum(weights, axis=1)
        deformed = volume_points + numerator / (denominator[:, None] + warp_eps)
        rotated = rotator(deformed.reshape(-1), pitch)
        return rotated.reshape((-1, 3))

    warp_batch = jax.jit(warp_batch_raw, device=device)

    def iter_batches():
        for width, bucket_rows in route_buckets.items():
            batch_size = max(
                1,
                min(
                    int(max_batch_points),
                    int(target_batch_interactions) // int(width),
                ),
            )
            for start in range(0, len(bucket_rows), batch_size):
                rows = bucket_rows[start : start + batch_size]
                yield int(width), int(batch_size), rows

    batch_plan = list(iter_batches())

    def expand_host_batch(width, batch_size, rows):
        count = len(rows)
        encoded = np.zeros((batch_size, width), dtype=np.int32)
        encoded[:count] = expand_route_rows(
            routes, tree, rows, n_surface, padded_width=width
        )
        points_batch = np.zeros((batch_size, 3), dtype=host_dtype)
        points_batch[:count] = reference_volume_points[rows]
        return count, points_batch, encoded

    host_cache_started = time.perf_counter()
    host_batch_cache = None
    if route_cache == "host":
        host_batch_cache = []
        cache_nbytes = 0
        n_batches = len(batch_plan)
        report_every = max(1, n_batches // 10)
        if progress:
            print(
                f"[cache] expanding {n_batches} fixed batches into host RAM",
                flush=True,
            )
        for batch_id, (width, batch_size, rows) in enumerate(
            batch_plan, start=1
        ):
            batch = expand_host_batch(width, batch_size, rows)
            host_batch_cache.append(batch)
            cache_nbytes += batch[1].nbytes + batch[2].nbytes
            if progress and (
                batch_id % report_every == 0 or batch_id == n_batches
            ):
                print(
                    f"[cache] batch {batch_id:>3}/{n_batches}: "
                    f"{cache_nbytes / 2**30:.3f} GiB, "
                    f"{time.perf_counter() - host_cache_started:.1f} s",
                    flush=True,
                )
    else:
        cache_nbytes = 0
    host_cache_seconds = time.perf_counter() - host_cache_started
    if progress and route_cache == "host":
        print(
            f"[cache] complete: {cache_nbytes / 2**30:.3f} GiB, "
            f"{host_cache_seconds:.3f} s",
            flush=True,
        )

    def prepare_batch(
        batch_id,
        width,
        batch_size,
        rows,
        cotangent=None,
    ):
        if host_batch_cache is None:
            count, points_batch, encoded = expand_host_batch(
                width, batch_size, rows
            )
        else:
            count, points_batch, encoded = host_batch_cache[batch_id]
        cotangent_batch = None
        if cotangent is not None:
            cotangent_batch = np.zeros((batch_size, 3), dtype=host_dtype)
            cotangent_batch[:count] = cotangent[rows]
        return (
            count,
            jax.device_put(points_batch, device),
            jax.device_put(encoded, device),
            None
            if cotangent_batch is None
            else jax.device_put(cotangent_batch, device),
        )

    deform_calls = 0

    def deform(surface_points, volume_points, pitch):
        nonlocal deform_calls
        deform_calls += 1
        call_started = time.perf_counter()
        if np.asarray(volume_points) is not reference_volume_points:
            raise ValueError("This precomputed deformer only accepts its reference volume array")
        if progress:
            print(
                f"[warp {deform_calls}] start: {len(reference_volume_points):,} "
                f"points, {len(route_buckets)} padded-width buckets",
                flush=True,
            )
        stage_started = time.perf_counter()
        surface_points_device = jax.device_put(
            jnp.asarray(surface_points, dtype=Xs0.dtype), device
        )
        pitch_device = jax.device_put(jnp.asarray(pitch, dtype=Xs0.dtype), device)
        surface_M, surface_b = surface_to_MB(surface_points_device)
        if progress:
            surface_M.block_until_ready()
            surface_b.block_until_ready()
            print(
                f"[warp {deform_calls}] surface normals + M,b: "
                f"{time.perf_counter() - stage_started:.3f} s",
                flush=True,
            )
            stage_started = time.perf_counter()
        node_M, node_b = aggregate_nodes(surface_M, surface_b)
        if progress:
            node_M.block_until_ready()
            node_b.block_until_ready()
            print(
                f"[warp {deform_calls}] tree M,b aggregation: "
                f"{time.perf_counter() - stage_started:.3f} s",
                flush=True,
            )
        source_M = jnp.concatenate((surface_M, node_M), axis=0)
        source_b = jnp.concatenate((surface_b, node_b), axis=0)
        result = np.empty(reference_volume_points.shape, dtype=host_dtype)
        active_width = None
        bucket_started = None
        bucket_prepare_seconds = 0.0
        bucket_kernel_seconds = 0.0
        bucket_batches = 0
        bucket_points = 0
        total_prepare_seconds = 0.0
        total_kernel_seconds = 0.0
        total_batches = 0

        def report_bucket():
            if not progress or active_width is None:
                return
            print(
                f"[warp {deform_calls}] width {active_width:>4}: "
                f"{bucket_points:,} points in {bucket_batches} batches, "
                f"route preparation/transfer {bucket_prepare_seconds:.3f} s, "
                f"JAX kernel {bucket_kernel_seconds:.3f} s, "
                f"total {time.perf_counter() - bucket_started:.3f} s",
                flush=True,
            )

        for batch_id, (width, batch_size, rows) in enumerate(batch_plan):
            if width != active_width:
                report_bucket()
                active_width = width
                bucket_started = time.perf_counter()
                bucket_prepare_seconds = 0.0
                bucket_kernel_seconds = 0.0
                bucket_batches = 0
                bucket_points = 0
                if progress:
                    print(
                        f"[warp {deform_calls}] entering width {width}, "
                        f"padded batch shape ({batch_size}, {width})",
                        flush=True,
                    )
            batch_started = time.perf_counter()
            count, points_batch, encoded, _ = prepare_batch(
                batch_id, width, batch_size, rows
            )
            if progress:
                points_batch.block_until_ready()
                encoded.block_until_ready()
            prepare_seconds = time.perf_counter() - batch_started
            bucket_prepare_seconds += prepare_seconds
            total_prepare_seconds += prepare_seconds
            batch_started = time.perf_counter()
            batch_result = np.asarray(
                warp_batch(source_M, source_b, points_batch, encoded, pitch_device)
            )
            kernel_seconds = time.perf_counter() - batch_started
            bucket_kernel_seconds += kernel_seconds
            total_kernel_seconds += kernel_seconds
            bucket_batches += 1
            bucket_points += count
            total_batches += 1
            result[rows] = batch_result[:count]
        report_bucket()
        if progress:
            # Sum the completed call independently of the per-bucket counters;
            # the batch path is synchronized above when profiling is enabled.
            # Reconstructing these totals from wall time would mix in printing.
            print(
                f"[warp {deform_calls}] batch total: "
                f"{total_batches} batches, route preparation/transfer "
                f"{total_prepare_seconds:.3f} s, JAX kernels "
                f"{total_kernel_seconds:.3f} s",
                flush=True,
            )
            print(
                f"[warp {deform_calls}] complete: "
                f"{time.perf_counter() - call_started:.3f} s",
                flush=True,
            )
        return result

    @jax.jit
    def warp_batch_vjp(source_M, source_b, points_batch,
                    encoded, pitch, cotangent):

        def f(M, b, p):
            return warp_batch_raw(M, b, points_batch, encoded, p)

        _, pullback = jax.vjp(f, source_M, source_b, pitch)
        return pullback(cotangent)

    def deform_vjp(surface_points, volume_points, pitch, d_volume_points):
        if np.asarray(volume_points) is not reference_volume_points:
            raise ValueError("This precomputed deformer only accepts its reference volume array")
        d_volume_points = np.asarray(d_volume_points, dtype=host_dtype)
        surface_points_device = jax.device_put(
            jnp.asarray(surface_points, dtype=Xs0.dtype), device
        )
        pitch_device = jax.device_put(jnp.asarray(pitch, dtype=Xs0.dtype), device)
        surface_M, surface_b = surface_to_MB(surface_points_device)
        node_M, node_b = aggregate_nodes(surface_M, surface_b)
        source_M = jnp.concatenate((surface_M, node_M), axis=0)
        source_b = jnp.concatenate((surface_b, node_b), axis=0)
        d_source_M = jnp.zeros_like(source_M)
        d_source_b = jnp.zeros_like(source_b)
        d_pitch = jnp.zeros_like(pitch_device)

        for batch_id, (width, batch_size, rows) in enumerate(batch_plan):
            _, points_batch, encoded, cotangent_batch = prepare_batch(
                batch_id,
                width,
                batch_size,
                rows,
                cotangent=d_volume_points,
            )

            # def batch_function(M_value, b_value, pitch_value):
            #     return warp_batch_raw(
            #         M_value, b_value, points_batch, encoded, pitch_value
            #     )

            # _, pullback = jax.vjp(batch_function, source_M, source_b, pitch_device)
            # dM_batch, db_batch, dpitch_batch = pullback(cotangent_batch)

            dM_batch, db_batch, dpitch_batch = warp_batch_vjp(
                source_M, source_b, points_batch, encoded, pitch_device, cotangent_batch
            )
            d_source_M = d_source_M + dM_batch
            d_source_b = d_source_b + db_batch
            d_pitch = d_pitch + dpitch_batch

        _, aggregate_pullback = jax.vjp(
            aggregate_nodes_raw, surface_M, surface_b
        )
        dM_tree, db_tree = aggregate_pullback(
            (d_source_M[n_surface:], d_source_b[n_surface:])
        )
        d_surface_M = d_source_M[:n_surface] + dM_tree
        d_surface_b = d_source_b[:n_surface] + db_tree
        _, surface_pullback = jax.vjp(surface_to_MB_raw, surface_points_device)
        (d_surface_points,) = surface_pullback((d_surface_M, d_surface_b))
        return np.asarray(d_surface_points), np.asarray(d_pitch).item()

    def deform_jvp(
        surface_points,
        volume_points,
        pitch,
        d_surface_points,
        d_pitch=0.0,
    ):
        """Forward derivative used for adjoint validation and forward-mode use."""
        if np.asarray(volume_points) is not reference_volume_points:
            raise ValueError("This precomputed deformer only accepts its reference volume array")
        surface_points_device = jax.device_put(
            jnp.asarray(surface_points, dtype=Xs0.dtype), device
        )
        d_surface_device = jax.device_put(
            jnp.asarray(d_surface_points, dtype=Xs0.dtype), device
        )
        pitch_device = jax.device_put(jnp.asarray(pitch, dtype=Xs0.dtype), device)
        d_pitch_device = jax.device_put(
            jnp.broadcast_to(
                jnp.asarray(d_pitch, dtype=Xs0.dtype), pitch_device.shape
            ),
            device,
        )
        surface_M, surface_b = surface_to_MB_raw(surface_points_device)
        # normal.compute_node_normals_from_topology has a custom VJP and JAX
        # therefore rejects direct forward-mode AD. Recover J*v by transposing
        # its linear VJP a second time; this checks the derivative actually used
        # by the production reverse path.
        _, surface_pullback = jax.vjp(
            surface_to_MB_raw, surface_points_device
        )

        def transposed_surface(cotangent_M, cotangent_b):
            (cotangent_surface,) = surface_pullback(
                (cotangent_M, cotangent_b)
            )
            return cotangent_surface

        zero_M = jnp.zeros_like(surface_M)
        zero_b = jnp.zeros_like(surface_b)
        _, transpose_pullback = jax.vjp(
            transposed_surface, zero_M, zero_b
        )
        d_surface_M, d_surface_b = transpose_pullback(d_surface_device)
        (node_M, node_b), (d_node_M, d_node_b) = jax.jvp(
            aggregate_nodes_raw,
            (surface_M, surface_b),
            (d_surface_M, d_surface_b),
        )
        source_M = jnp.concatenate((surface_M, node_M), axis=0)
        source_b = jnp.concatenate((surface_b, node_b), axis=0)
        d_source_M = jnp.concatenate((d_surface_M, d_node_M), axis=0)
        d_source_b = jnp.concatenate((d_surface_b, d_node_b), axis=0)
        tangent_result = np.empty(reference_volume_points.shape, dtype=host_dtype)
        for batch_id, (width, batch_size, rows) in enumerate(batch_plan):
            count, points_batch, encoded, _ = prepare_batch(
                batch_id, width, batch_size, rows
            )

            def batch_function(M_value, b_value, pitch_value):
                return warp_batch_raw(
                    M_value, b_value, points_batch, encoded, pitch_value
                )

            _, tangent_batch = jax.jvp(
                batch_function,
                (source_M, source_b, pitch_device),
                (d_source_M, d_source_b, d_pitch_device),
            )
            tangent_result[rows] = np.asarray(tangent_batch)[:count]
        return tangent_result

    aux = {
        "wall_points": wall_points,
        "wall_ids": wall_ids,
        "wall_faces": wall_faces,
        "tree": tree,
        "routes": routes,
        "route_buckets": route_buckets,
        "route_cache": route_cache,
        "cached_routes": None if cache_path is None else str(cache_path),
        "routes_cache_hit": routes_cache_hit,
        "route_cache_nbytes": cache_nbytes,
        "deform_jvp": deform_jvp,
        "surface_area": surface_area,
        "ldef": ldef,
        "area_weighted_nodes": area_weighted_nodes,
        "timings": {
            "tree_seconds": tree_seconds,
            "routes_seconds": routes_seconds,
            "route_cache_seconds": host_cache_seconds,
            "total_setup_seconds": time.perf_counter() - setup_started,
        },
    }
    return wall_points, wall_ids, wall_faces, deform, deform_vjp, aux
