"""Pure-NumPy one-time preprocessing for the JAX kd-tree volume warper."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ._kdtree_tree import (
    DODECAHEDRON_DIRECTIONS,
    RSTAR,
    FlatTree,
    _estimate_denominator_batch,
    _interpolated_errors,
    _leaf_source_table,
    build_flat_tree,
    compute_error_table,
)


@dataclass
class KdTreeRoutes:
    """Compact terminal-node frontiers for real and mirrored traversals."""

    real_offsets: np.ndarray
    real_node_ids: np.ndarray
    mirror_offsets: np.ndarray
    mirror_node_ids: np.ndarray
    expanded_counts: np.ndarray

    @property
    def num_volume_points(self) -> int:
        return len(self.expanded_counts)

    @property
    def nbytes(self) -> int:
        return int(
            self.real_offsets.nbytes
            + self.real_node_ids.nbytes
            + self.mirror_offsets.nbytes
            + self.mirror_node_ids.nbytes
            + self.expanded_counts.nbytes
        )


def build_idwarp_tree(
    surface_points: np.ndarray,
    surface_area: np.ndarray,
    *,
    bucket_size: int,
    ldef: float,
    a_exp: float,
    b_exp: float,
    alpha: float,
    distance_eps: float,
) -> FlatTree:
    """Build the fixed tree and IDWarp error table."""
    surface_points = np.asarray(surface_points, dtype=np.float64)
    surface_area = np.asarray(surface_area, dtype=np.float64)
    tree = build_flat_tree(surface_points, surface_area, bucket_size)
    permuted_points = np.ascontiguousarray(surface_points[tree.permutation])
    permuted_area = np.ascontiguousarray(surface_area[tree.permutation])
    tree.error = compute_error_table(
        permuted_points,
        permuted_area,
        tree.start,
        tree.end,
        tree.left,
        tree.center,
        tree.radius,
        tree.area,
        float(ldef),
        float(a_exp),
        float(b_exp),
        float(alpha**b_exp),
        float(distance_eps),
        RSTAR,
        DODECAHEDRON_DIRECTIONS,
    )
    return tree


def _offsets_from_counts(counts: np.ndarray) -> np.ndarray:
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    return offsets


def _frontier_batch(
    volume_points: np.ndarray,
    approximate_denominator: np.ndarray,
    tree: FlatTree,
    err_tol: float,
    distance_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row-grouped terminal node IDs and expanded source counts."""
    n_volume = len(volume_points)
    terminal_counts = np.zeros(n_volume, dtype=np.int32)
    expanded_counts = np.zeros(n_volume, dtype=np.int32)
    terminal_chunks = []
    rows = np.arange(n_volume, dtype=np.int32)
    nodes = np.zeros(n_volume, dtype=np.int32)

    def retain(retained_rows, retained_nodes, source_counts):
        if not len(retained_nodes):
            return
        retained_rows = np.ascontiguousarray(retained_rows, dtype=np.int32)
        retained_nodes = np.ascontiguousarray(retained_nodes, dtype=np.uint16)
        counts = np.bincount(retained_rows, minlength=n_volume).astype(np.int32)
        terminal_counts[:] += counts
        expanded_counts[:] += np.bincount(
            retained_rows,
            weights=source_counts,
            minlength=n_volume,
        ).astype(np.int32)
        terminal_chunks.append((retained_rows, retained_nodes, counts))

    while len(nodes):
        is_leaf = tree.left[nodes] < 0
        if np.any(is_leaf):
            leaf_rows = rows[is_leaf]
            leaf_nodes = nodes[is_leaf]
            source_counts = tree.end[leaf_nodes] - tree.start[leaf_nodes]
            retain(leaf_rows, leaf_nodes, source_counts)

        internal_rows = rows[~is_leaf]
        internal_nodes = nodes[~is_leaf]
        if not len(internal_nodes):
            break
        difference = volume_points[internal_rows] - tree.center[internal_nodes]
        distance = np.sqrt(
            np.einsum("ri,ri->r", difference, difference) + distance_eps
        )
        distance_over_radius = distance / tree.radius[internal_nodes]
        candidate = distance_over_radius >= 2.0
        accept = np.zeros(len(internal_nodes), dtype=np.bool_)
        if np.any(candidate):
            candidate_errors = _interpolated_errors(
                internal_nodes[candidate],
                distance_over_radius[candidate],
                tree.error,
                RSTAR,
            )
            accept[candidate] = candidate_errors < (
                err_tol * approximate_denominator[internal_rows[candidate]]
            )
        if np.any(accept):
            accepted_rows = internal_rows[accept]
            accepted_nodes = internal_nodes[accept]
            retain(
                accepted_rows,
                accepted_nodes,
                np.ones(len(accepted_nodes), dtype=np.int32),
            )

        expanded_rows = internal_rows[~accept]
        expanded_nodes = internal_nodes[~accept]
        rows = np.repeat(expanded_rows, 2)
        nodes = np.column_stack(
            (tree.left[expanded_nodes], tree.right[expanded_nodes])
        ).reshape(-1)

    offsets = _offsets_from_counts(terminal_counts)
    grouped_nodes = np.empty(offsets[-1], dtype=np.uint16)
    cursors = np.zeros(n_volume, dtype=np.int32)
    for chunk_rows, chunk_nodes, chunk_counts in terminal_chunks:
        chunk_offsets = _offsets_from_counts(chunk_counts)
        within_row = np.arange(len(chunk_rows), dtype=np.int64) - np.repeat(
            chunk_offsets[:-1], chunk_counts
        )
        positions = offsets[chunk_rows] + cursors[chunk_rows] + within_row
        grouped_nodes[positions] = chunk_nodes
        cursors += chunk_counts
    return terminal_counts, grouped_nodes, expanded_counts


def _denominator_batch(
    volume_points,
    permuted_points,
    permuted_area,
    tree,
    leaf_source_ids,
    leaf_source_valid,
    ldef,
    a_exp,
    b_exp,
    alpha_to_b,
    distance_eps,
):
    return _estimate_denominator_batch(
        volume_points,
        permuted_points,
        permuted_area,
        tree.left,
        tree.right,
        tree.center,
        tree.radius,
        tree.area,
        leaf_source_ids,
        leaf_source_valid,
        ldef,
        a_exp,
        b_exp,
        alpha_to_b,
        distance_eps,
    )


def precompute_routes(
    volume_points: np.ndarray,
    surface_points: np.ndarray,
    surface_area: np.ndarray,
    tree: FlatTree,
    *,
    exact_y_symmetry: bool,
    ldef: float,
    a_exp: float,
    b_exp: float,
    alpha: float,
    err_tol: float,
    distance_eps: float,
    batch_size: int = 30_000,
    progress: bool = False,
) -> KdTreeRoutes:
    """Traverse once and retain only terminal tree-node frontiers."""
    volume_points = np.ascontiguousarray(volume_points, dtype=np.float64)
    permuted_points = np.ascontiguousarray(
        np.asarray(surface_points, dtype=np.float64)[tree.permutation]
    )
    permuted_area = np.ascontiguousarray(
        np.asarray(surface_area, dtype=np.float64)[tree.permutation]
    )
    leaf_source_ids, leaf_source_valid = _leaf_source_table(
        tree.start, tree.end, tree.left
    )
    n_volume = len(volume_points)
    alpha_to_b = float(alpha**b_exp)
    real_count_chunks = []
    real_node_chunks = []
    mirror_count_chunks = []
    mirror_node_chunks = []
    expanded_count_chunks = []
    n_batches = (n_volume + batch_size - 1) // batch_size
    report_every = max(1, n_batches // 10)
    started = time.perf_counter()
    if progress:
        print(
            f"[routes] {n_volume:,} volume points in {n_batches} "
            f"batches of at most {batch_size:,}",
            flush=True,
        )

    # The denominator and terminal frontier are computed together per bounded
    # batch. Keeping compact node chunks avoids both a duplicate traversal and
    # dense padding by the maximum route length.
    for batch_id, batch_start in enumerate(range(0, n_volume, batch_size), start=1):
        batch_end = min(batch_start + batch_size, n_volume)
        batch_points = volume_points[batch_start:batch_end]
        denominator = _denominator_batch(
            batch_points,
            permuted_points,
            permuted_area,
            tree,
            leaf_source_ids,
            leaf_source_valid,
            ldef,
            a_exp,
            b_exp,
            alpha_to_b,
            distance_eps,
        )
        mirror_points = None
        if exact_y_symmetry:
            mirror_points = batch_points.copy()
            mirror_points[:, 1] *= -1.0
            denominator += _denominator_batch(
                mirror_points,
                permuted_points,
                permuted_area,
                tree,
                leaf_source_ids,
                leaf_source_valid,
                ldef,
                a_exp,
                b_exp,
                alpha_to_b,
                distance_eps,
            )
        real_counts, real_nodes, expanded = _frontier_batch(
            batch_points, denominator, tree, err_tol, distance_eps
        )
        real_count_chunks.append(real_counts)
        real_node_chunks.append(real_nodes)
        if exact_y_symmetry:
            mirror_counts, mirror_nodes, mirror_expanded = _frontier_batch(
                mirror_points, denominator, tree, err_tol, distance_eps
            )
            expanded += mirror_expanded
        else:
            mirror_counts = np.zeros(len(batch_points), dtype=np.int32)
            mirror_nodes = np.empty(0, dtype=np.uint16)
        mirror_count_chunks.append(mirror_counts)
        mirror_node_chunks.append(mirror_nodes)
        expanded_count_chunks.append(expanded)
        if progress and (batch_id % report_every == 0 or batch_id == n_batches):
            print(
                f"[routes] batch {batch_id:>3}/{n_batches}: "
                f"{batch_end:,}/{n_volume:,} points "
                f"({100.0 * batch_end / n_volume:5.1f}%), "
                f"{time.perf_counter() - started:.1f} s",
                flush=True,
            )

    real_counts = np.concatenate(real_count_chunks)
    mirror_counts = np.concatenate(mirror_count_chunks)
    expanded_counts = np.concatenate(expanded_count_chunks)
    if len(tree.start) > np.iinfo(np.uint16).max:
        raise ValueError("Tree has too many nodes for uint16 frontier IDs")
    real_offsets = _offsets_from_counts(real_counts)
    mirror_offsets = _offsets_from_counts(mirror_counts)
    real_node_ids = np.concatenate(real_node_chunks)
    mirror_node_ids = np.concatenate(mirror_node_chunks)

    return KdTreeRoutes(
        real_offsets=real_offsets,
        real_node_ids=real_node_ids,
        mirror_offsets=mirror_offsets,
        mirror_node_ids=mirror_node_ids,
        expanded_counts=expanded_counts,
    )


def _gather_frontier_rows(offsets, node_ids, row_ids):
    counts = offsets[row_ids + 1] - offsets[row_ids]
    total = int(np.sum(counts, dtype=np.int64))
    if total == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.uint16)
    local_rows = np.repeat(np.arange(len(row_ids), dtype=np.int32), counts)
    prefix = _offsets_from_counts(counts)[:-1]
    within = np.arange(total, dtype=np.int64) - np.repeat(prefix, counts)
    positions = np.repeat(offsets[row_ids], counts) + within
    return local_rows, node_ids[positions]


def _expand_pass(offsets, node_ids, row_ids, tree, n_surface, mirrored):
    frontier_rows, frontier_nodes = _gather_frontier_rows(
        offsets, node_ids, row_ids
    )
    if not len(frontier_nodes):
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.zeros(len(row_ids), dtype=np.int32),
        )
    leaf = tree.left[frontier_nodes] < 0
    source_counts = np.where(
        leaf,
        tree.end[frontier_nodes] - tree.start[frontier_nodes],
        1,
    ).astype(np.int32)
    expanded_frontier = np.repeat(
        np.arange(len(frontier_nodes), dtype=np.int32), source_counts
    )
    expanded_rows = frontier_rows[expanded_frontier]
    expanded_nodes = frontier_nodes[expanded_frontier]
    prefix = _offsets_from_counts(source_counts)[:-1]
    within = np.arange(len(expanded_frontier), dtype=np.int64) - np.repeat(
        prefix, source_counts
    )
    expanded_leaf = tree.left[expanded_nodes] < 0
    sources = n_surface + expanded_nodes.astype(np.int64)
    sources[expanded_leaf] = tree.permutation[
        tree.start[expanded_nodes[expanded_leaf]] + within[expanded_leaf]
    ]
    encoded = (sources + 1).astype(np.int32)
    if mirrored:
        encoded *= -1
    counts = np.bincount(expanded_rows, minlength=len(row_ids)).astype(np.int32)
    return expanded_rows, encoded, counts


def expand_route_rows(
    routes: KdTreeRoutes,
    tree: FlatTree,
    row_ids: np.ndarray,
    n_surface: int,
    padded_width: int,
) -> np.ndarray:
    """Return signed, padded source IDs for one JAX batch."""
    row_ids = np.ascontiguousarray(row_ids, dtype=np.int64)
    if len(row_ids) and int(np.max(routes.expanded_counts[row_ids])) > padded_width:
        raise ValueError("padded_width is smaller than a selected row")
    output = np.zeros((len(row_ids), padded_width), dtype=np.int32)
    real_rows, real_encoded, real_counts = _expand_pass(
        routes.real_offsets,
        routes.real_node_ids,
        row_ids,
        tree,
        n_surface,
        False,
    )
    if len(real_encoded):
        real_prefix = _offsets_from_counts(real_counts)[:-1]
        real_columns = np.arange(len(real_encoded), dtype=np.int64) - np.repeat(
            real_prefix, real_counts
        )
        output[real_rows, real_columns] = real_encoded

    mirror_rows, mirror_encoded, mirror_counts = _expand_pass(
        routes.mirror_offsets,
        routes.mirror_node_ids,
        row_ids,
        tree,
        n_surface,
        True,
    )
    if len(mirror_encoded):
        mirror_prefix = _offsets_from_counts(mirror_counts)[:-1]
        mirror_columns = (
            np.arange(len(mirror_encoded), dtype=np.int64)
            - np.repeat(mirror_prefix, mirror_counts)
            + real_counts[mirror_rows]
        )
        output[mirror_rows, mirror_columns] = mirror_encoded
    if not np.array_equal(real_counts + mirror_counts, routes.expanded_counts[row_ids]):
        raise RuntimeError("Expanded route width mismatch")
    return output


def next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())
