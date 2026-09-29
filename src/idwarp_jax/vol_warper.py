"""Production interface for the error-controlled JAX kd-tree volume warper."""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np

from .kdT_vol_warper._kdtree_warper import make_kdtree_deformer


def _wall_metadata(mesh, surface_face_type):
    points = np.asarray(mesh["points"])
    wall_faces = [
        np.asarray(face)
        for face, face_type in zip(
            mesh["faces"]["points"], mesh["faces"]["type"]
        )
        if face_type == surface_face_type
    ]
    if not wall_faces:
        raise ValueError(f"No mesh faces have type {surface_face_type}")
    wall_ids = np.unique(np.concatenate(wall_faces))
    return points[wall_ids], wall_ids, wall_faces


def make_deformer(
    mesh,
    device,
    rotator,
    LdefFact=1.0,
    symmetry_mode="exactsym",
    symmetry_length_scale=1.0,
    volume_chunk_size=512,
    surface_block_size=1024,
    normal_eps=1.0e-6,
    rotation_eps=1.0e-6,
    warp_eps=1.0e-6,
    symmetry_tolerance=1.0e-6,
    surface_face_type=1,
    useRotations=True,
    zeroCornerRotations=True,
    cornerAngle=30.0,
    precompute_denominator=False,
    reference_volume_points=None,
    *,
    bucket_size=8,
    err_tol=5.0e-4,
    aExp=3.0,
    bExp=5.0,
    alpha=0.25,
    target_batch_interactions=6_000_000,
    max_batch_points=4096,
    area_weighted_nodes=True,
    route_cache="host",
    cached_routes=None,
    progress=True,
):
    """Build a kd-tree deformer while preserving the former public API.

    The kd-tree topology is valid only for fixed reference volume points.  If
    ``reference_volume_points`` is omitted, initialization is deferred until
    the first forward or reverse call.  This preserves older direct callers
    that supplied volume points only to ``deform``.

    ``volume_chunk_size``, ``surface_block_size``,
    ``symmetry_length_scale``, and ``precompute_denominator`` are accepted for
    source compatibility with the former dense implementation.  Kd-tree batch
    sizing and denominator/frontier preprocessing replace those controls.
    """
    del volume_chunk_size, surface_block_size
    del symmetry_length_scale, precompute_denominator
    normalized_symmetry = str(symmetry_mode).lower()
    if normalized_symmetry not in {"exactsym", "exact-y", "exact_y"}:
        raise NotImplementedError(
            "The kd-tree warper currently supports only exact y=0 symmetry; "
            f"got symmetry_mode={symmetry_mode!r}"
        )

    wall_points, wall_ids, wall_faces = _wall_metadata(
        mesh, surface_face_type
    )
    state = {
        "reference": None,
        "deform": None,
        "deform_vjp": None,
        "aux": None,
    }

    def initialize(volume_points):
        reference = np.asarray(volume_points)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError(
                "reference volume points must have shape (n, 3); got "
                f"{reference.shape}"
            )
        if state["deform"] is not None:
            if reference is not state["reference"]:
                raise ValueError(
                    "This kd-tree deformer is fixed to the volume-point array "
                    "used for its first initialization"
                )
            return

        (
            initialized_wall_points,
            initialized_wall_ids,
            initialized_wall_faces,
            inner_deform,
            inner_deform_vjp,
            inner_aux,
        ) = make_kdtree_deformer(
            mesh,
            reference,
            device,
            rotator,
            LdefFact=LdefFact,
            bucket_size=bucket_size,
            err_tol=err_tol,
            aExp=aExp,
            bExp=bExp,
            alpha=alpha,
            warp_eps=warp_eps,
            normal_eps=normal_eps,
            rotation_eps=rotation_eps,
            symmetry_tolerance=symmetry_tolerance,
            surface_face_type=surface_face_type,
            useRotations=useRotations,
            zeroCornerRotations=zeroCornerRotations,
            cornerAngle=cornerAngle,
            target_batch_interactions=target_batch_interactions,
            max_batch_points=max_batch_points,
            area_weighted_nodes=area_weighted_nodes,
            route_cache=route_cache,
            cached_routes=cached_routes,
            progress=progress,
        )
        if not np.array_equal(initialized_wall_ids, wall_ids):
            raise RuntimeError("Wall-point ordering changed during initialization")
        if not np.array_equal(initialized_wall_points, wall_points):
            raise RuntimeError("Wall-point coordinates changed during initialization")
        if len(initialized_wall_faces) != len(wall_faces):
            raise RuntimeError("Wall-face topology changed during initialization")
        state.update(
            reference=reference,
            deform=inner_deform,
            deform_vjp=inner_deform_vjp,
            aux=inner_aux,
        )

    if reference_volume_points is not None:
        initialize(reference_volume_points)

    def deform(surface_points, volume_points, pitch):
        initialize(volume_points)
        return state["deform"](surface_points, state["reference"], pitch)

    def deform_vjp(surface_points, volume_points, pitch, d_volume_points):
        initialize(volume_points)
        return state["deform_vjp"](
            surface_points,
            state["reference"],
            pitch,
            d_volume_points,
        )

    # Used by build_volume_pts_func to expose the detailed implementation data
    # without changing make_deformer's historical five-value return.
    deform._kdtree_state = state
    return wall_points, wall_ids, wall_faces, deform, deform_vjp


def build_volume_pts_func(
    foam_mesh,
    base_volume_points,
    device,
    volume_chunk_size=512,
    surface_block_size=1024,
    surface_face_type=1,
    return_aux=False,
    print_timings=True,
    precompute_denominator=True,
    *,
    LdefFact=100.0,
    symmetry_mode="exactsym",
    normal_eps=1.0e-6,
    rotation_eps=1.0e-6,
    warp_eps=1.0e-6,
    symmetry_tolerance=1.0e-6,
    useRotations=True,
    zeroCornerRotations=True,
    cornerAngle=30.0,
    bucket_size=8,
    err_tol=5.0e-4,
    target_batch_interactions=6_000_000,
    max_batch_points=4096,
    area_weighted_nodes=True,
    route_cache="host",
    cached_routes=None,
    progress=True,
):
    """Build the volume-deformation and reverse-derivative functions.

    Parameters
    ----------
    foam_mesh : dict
        Mesh dictionary containing points and boundary faces.
    base_volume_points : array_like, shape (n, 3)
        Undeformed volume-point coordinates.
    device : jax.Device
        JAX device used for deformation kernels.
    volume_chunk_size : int, default=512
        Compatibility option; currently unused by the kd-tree warper.
    surface_block_size : int, default=1024
        Compatibility option; currently unused by the kd-tree warper.
    surface_face_type : int, default=1
        Boundary-face type that identifies the moving surface.
    return_aux : bool, default=False
        Return implementation metadata as a third result.
    print_timings : bool, default=True
        Print elapsed time for forward and reverse calls.
    precompute_denominator : bool, default=True
        Compatibility option; kd-tree preprocessing always handles this.
    LdefFact : float, default=100.0
        Multiplier for the characteristic deformation length.
    symmetry_mode : str, default="exactsym"
        Symmetry treatment; only exact symmetry at ``y=0`` is supported.
    normal_eps : float, default=1e-6
        Regularization used when computing surface normals.
    rotation_eps : float, default=1e-6
        Regularization used when computing local rotations.
    warp_eps : float, default=1e-6
        Regularization used in deformation weights.
    symmetry_tolerance : float, default=1e-6
        Distance from ``y=0`` treated as lying on the symmetry plane.
    useRotations : bool, default=True
        Include local surface rotations in the deformation.
    zeroCornerRotations : bool, default=True
        Suppress local rotations at detected corners.
    cornerAngle : float, default=30.0
        Corner-detection angle in degrees.
    bucket_size : int, default=8
        Maximum kd-tree leaf size, using IDWarp's inclusive convention.
    err_tol : float, default=5e-4
        Relative tolerance for pruning kd-tree interactions.
    target_batch_interactions : int, default=6000000
        Target interaction count per JAX batch.
    max_batch_points : int, default=4096
        Maximum volume points per JAX batch.
    area_weighted_nodes : bool, default=True
        Weight aggregate kd-tree nodes by surface area.
    route_cache : {"host", "none"}, default="host"
        Keep expanded routing batches in host memory or rebuild them.
    cached_routes : path-like or None, default=None
        Directory used to load and save compact routing data.
    progress : bool, default=True
        Print preprocessing and cache progress.

    Returns
    -------
    compute_volume_pts : callable
        Maps surface points and ``{"pitch": degrees}`` to volume points.
    d_compute_volume_pts : callable
        Applies the reverse derivative to a volume-point seed.
    aux : dict, optional
        Metadata returned only when ``return_aux=True``.
    """
    def rotate_pitch_jax(xyz_flat, pitch_degrees):
        xyz = xyz_flat.reshape((-1, 3))
        theta = pitch_degrees * jnp.pi / 180.0
        cosine = jnp.cos(theta)
        sine = jnp.sin(theta)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        return jnp.stack(
            (cosine * x + sine * z, y, -sine * x + cosine * z),
            axis=1,
        ).reshape(xyz_flat.shape)

    base_volume_points = np.asarray(base_volume_points)
    wall_points, wall_ids, wall_faces, deform, deform_vjp = make_deformer(
        foam_mesh,
        device=device,
        rotator=rotate_pitch_jax,
        LdefFact=LdefFact,
        symmetry_mode=symmetry_mode,
        volume_chunk_size=volume_chunk_size,
        surface_block_size=surface_block_size,
        normal_eps=normal_eps,
        rotation_eps=rotation_eps,
        warp_eps=warp_eps,
        symmetry_tolerance=symmetry_tolerance,
        surface_face_type=surface_face_type,
        useRotations=useRotations,
        zeroCornerRotations=zeroCornerRotations,
        cornerAngle=cornerAngle,
        precompute_denominator=precompute_denominator,
        reference_volume_points=base_volume_points,
        bucket_size=bucket_size,
        err_tol=err_tol,
        target_batch_interactions=target_batch_interactions,
        max_batch_points=max_batch_points,
        area_weighted_nodes=area_weighted_nodes,
        route_cache=route_cache,
        cached_routes=cached_routes,
        progress=progress,
    )

    def compute_volume_pts(surf_pts_val, flow_condition):
        started = time.perf_counter()
        new_vol_pts = deform(
            surf_pts_val,
            base_volume_points,
            flow_condition["pitch"],
        )
        if print_timings:
            print(
                f"* evaluated volume deformation ({new_vol_pts.shape[0]} pts) "
                f"({time.perf_counter() - started:.3f} sec)"
            )
        return new_vol_pts

    def d_compute_volume_pts(surf_pts_val, flow_condition, d_vol):
        started = time.perf_counter()
        d_surf, d_pitch = deform_vjp(
            surf_pts_val,
            base_volume_points,
            flow_condition["pitch"],
            d_vol,
        )
        if print_timings:
            print(
                f"* evaluated VJP volume deformation ({len(base_volume_points)} pts) "
                f"({time.perf_counter() - started:.3f} sec)"
            )
        return d_surf, {"pitch": d_pitch}

    if return_aux:
        state = deform._kdtree_state
        implementation_aux = state["aux"]
        aux = {
            "wall_points": wall_points,
            "wall_ids": wall_ids,
            "wall_faces": wall_faces,
            "deform": deform,
            "deform_vjp": deform_vjp,
            "precompute_denominator": False,
            "kdtree": implementation_aux,
        }
        return compute_volume_pts, d_compute_volume_pts, aux
    return compute_volume_pts, d_compute_volume_pts
