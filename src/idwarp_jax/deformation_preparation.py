"""Prepare reusable fixed data for JAX mesh deformation.

This module contains the coordinate-independent and reference-geometry work
that should be performed once for repeated mesh deformations. 
The prepared arrays remain JAX arrays and can be captured by the one-input function used by
JVP and VJP calculations.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from idwarp_jax import normal, symmetry
from idwarp_jax.symmetry import (
    SYMMETRY_APPROXIMATE,
    SYMMETRY_NONE,
)

Array = jax.Array


class PreparedDeformationData(NamedTuple):
    """Fixed mesh and reference-geometry data reused by each deformation."""

    Xv0: Array
    Xs0: Array
    surface_global_ids: Array
    warp_global_ids: Array
    normal_topology: normal.NormalTopology
    normals0: Array
    Ai: Array
    Ldef0: Array
    corner_mask: Array
    symmetry_plane_point: Array | None
    symmetry_plane_normal: Array | None
    surface_on_plane: Array | None
    symmetry_mode: str


class DeformationOptions(NamedTuple):
    """Numerical options used by one deformation evaluation."""

    LdefFact: float
    aExp: float
    bExp: float
    alpha: float
    normal_eps: float
    rotation_eps: float
    warp_eps: float
    volume_chunk_size: int
    surface_block_size: int
    symmetry_length_scale: float | None
    useRotations: bool


def _make_warp_global_ids(
    n_volume: int,
    surface_global_ids: Array,
) -> Array:
    """Return global IDs for volume points outside the prescribed surface.

    This is a fixed topology operation. NumPy performs it once on the host and
    the resulting index array is converted back to JAX for later GPU use.
    """
    surface_ids_np = np.asarray(
        jax.device_get(surface_global_ids),
        dtype=np.int64,
    ).reshape(-1)

    if np.any(surface_ids_np < 0) or np.any(surface_ids_np >= n_volume):
        raise ValueError(
            "surface_global_ids contains an index outside the volume mesh"
        )

    if np.unique(surface_ids_np).size != surface_ids_np.size:
        raise ValueError("surface_global_ids must not contain duplicates")

    prescribed_mask = np.zeros(n_volume, dtype=bool)
    prescribed_mask[surface_ids_np] = True

    warp_global_ids_np = np.flatnonzero(~prescribed_mask).astype(
        np.int32,
        copy=False,
    )
    return jnp.asarray(warp_global_ids_np, dtype=jnp.int32)


def validate_deformation_configuration(
    *,
    volume_chunk_size: int,
    surface_block_size: int,
    symmetry_mode: str | None,
    symmetry_plane_point,
    symmetry_plane_normal,
    symmetry_length_scale: float | None,
) -> str:
    """Validate public configuration and return the canonical symmetry mode."""
    mode = symmetry.normalize_symmetry_mode(symmetry_mode)

    if volume_chunk_size <= 0:
        raise ValueError("volume_chunk_size must be positive")
    if surface_block_size <= 0:
        raise ValueError("surface_block_size must be positive")

    if mode != SYMMETRY_NONE:
        if symmetry_plane_point is None or symmetry_plane_normal is None:
            raise ValueError(
                "symmetry_plane_point and symmetry_plane_normal are required "
                "for exact and approximate symmetry modes"
            )

    if mode == SYMMETRY_APPROXIMATE:
        if symmetry_length_scale is None:
            raise ValueError(
                "symmetry_length_scale is required for 'approxsym'"
            )
        if symmetry_length_scale <= 0.0:
            raise ValueError(
                "symmetry_length_scale must be positive for 'approxsym'"
            )

    return mode


def prepare_deformation_data(
    Xv0,
    Xs0,
    conn,
    face_sizes,
    surface_global_ids,
    *,
    normal_eps: float,
    warp_eps: float,
    symmetry_mode: str | None,
    symmetry_plane_point,
    symmetry_plane_normal,
    symmetry_tolerance: float,
    zeroCornerRotations: bool = True,
    cornerAngle: float = 30.0,
) -> PreparedDeformationData:
    """Prepare all fixed arrays needed by repeated deformation evaluations."""
    mode = symmetry.normalize_symmetry_mode(symmetry_mode)

    Xv0 = jnp.asarray(Xv0)
    Xs0 = jnp.asarray(Xs0, dtype=Xv0.dtype)
    conn = jnp.asarray(conn, dtype=jnp.int32)
    face_sizes = jnp.asarray(face_sizes, dtype=jnp.int32)
    surface_global_ids = jnp.asarray(
        surface_global_ids,
        dtype=jnp.int32,
    )

    warp_global_ids = _make_warp_global_ids(
        Xv0.shape[0],
        surface_global_ids,
    )

    normal_topology = normal.prepare_normal_topology(conn, face_sizes)
    normals0, Ai = normal.compute_node_normals_from_topology(
        Xs0,
        normal_topology,
        eps=normal_eps,
    )

    # Match IDWarp's reference-length normalization exactly:
    # Ldef0 = max_i ||Xs0_i - mean(Xs0)||.
    Xcen = jnp.mean(Xs0, axis=0)
    dX = Xs0 - Xcen
    Ldef0 = jnp.max(jnp.sqrt(jnp.sum(dX * dX, axis=1)))

    # IDWarp determines corners once from the reference surface and disables
    # the local normal-rotation contribution at those nodes.
    corner_mask = normal.detect_corner_nodes_from_topology(
        Xs0,
        normal_topology,
        corner_angle=cornerAngle,
        zero_corner_rotations=zeroCornerRotations,
    )

    if mode == SYMMETRY_NONE:
        plane_point = None
        plane_normal = None
        surface_on_plane = None
    else:
        plane_point = jnp.asarray(
            symmetry_plane_point,
            dtype=Xv0.dtype,
        )
        plane_normal = jnp.asarray(
            symmetry_plane_normal,
            dtype=Xv0.dtype,
        )
        surface_on_plane = symmetry.points_on_plane(
            Xs0,
            plane_point,
            plane_normal,
            symmetry_tolerance,
            eps=warp_eps,
        )

    return PreparedDeformationData(
        Xv0=Xv0,
        Xs0=Xs0,
        surface_global_ids=surface_global_ids,
        warp_global_ids=warp_global_ids,
        normal_topology=normal_topology,
        normals0=normals0,
        Ai=Ai,
        Ldef0=Ldef0,
        corner_mask=corner_mask,
        symmetry_plane_point=plane_point,
        symmetry_plane_normal=plane_normal,
        surface_on_plane=surface_on_plane,
        symmetry_mode=mode,
    )
