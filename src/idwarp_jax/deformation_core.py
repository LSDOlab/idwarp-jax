"""Coordinate-dependent core of JAX mesh deformation.

The public driver prepares fixed mesh data separately and calls
:func:`deform_prepared_mesh` for each surface displacement. 
This function contains only work that depends on the current deformation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from idwarp_jax import normal, rotation, symmetry, warp
from idwarp_jax.deformation_preparation import (
    DeformationOptions,
    PreparedDeformationData,
)
from idwarp_jax.symmetry import SYMMETRY_NONE

Array = jax.Array


def _constrain_target_surface(
    prepared: PreparedDeformationData,
    surface_displacement: Array,
    warp_eps: float,
) -> tuple[Array, Array | None, Array | None]:
    """Construct the target surface and enforce its fixed symmetry mask."""
    target_surface = prepared.Xs0 + surface_displacement

    if prepared.symmetry_mode == SYMMETRY_NONE:
        return target_surface, None, None

    projected_surface = symmetry.project_points_to_plane(
        target_surface,
        prepared.symmetry_plane_point,
        prepared.symmetry_plane_normal,
        eps=warp_eps,
    )
    constrained_surface = jnp.where(
        prepared.surface_on_plane[:, None],
        projected_surface,
        target_surface,
    )
    return (
        constrained_surface,
        prepared.symmetry_plane_point,
        prepared.symmetry_plane_normal,
    )


def deform_prepared_mesh(
    prepared: PreparedDeformationData,
    surface_displacement,
    options: DeformationOptions,
) -> Array:
    """Evaluate one deformation using previously prepared fixed mesh data."""
    surface_displacement = jnp.asarray(
        surface_displacement,
        dtype=prepared.Xv0.dtype,
    )

    Xs, plane_point, plane_normal = _constrain_target_surface(
        prepared,
        surface_displacement,
        options.warp_eps,
    )

    normals, _ = normal.compute_node_normals_from_topology(
        Xs,
        prepared.normal_topology,
        eps=options.normal_eps,
    )

    if options.useRotations:
        M, b = rotation.get_MB_rotation(
            prepared.Xs0,
            Xs,
            prepared.normals0,
            normals,
            eps=options.rotation_eps,
            corner_mask=prepared.corner_mask,
        )
    else:
        M, b = rotation.get_MB_no_rotation(prepared.Xs0, Xs)

    Xv_non_surface = warp.deformed_volume_points(
        Xv0=prepared.Xv0[prepared.warp_global_ids],
        Xs0=prepared.Xs0,
        Ai=prepared.Ai,
        M=M,
        b=b,
        LdefFact=options.LdefFact,
        Ldef0=prepared.Ldef0,
        aExp=options.aExp,
        bExp=options.bExp,
        alpha=options.alpha,
        eps=options.warp_eps,
        volume_chunk_size=options.volume_chunk_size,
        surface_block_size=options.surface_block_size,
        symmetry_mode=prepared.symmetry_mode,
        symmetry_plane_point=plane_point,
        symmetry_plane_normal=plane_normal,
        symmetry_length_scale=options.symmetry_length_scale,
    )

    Xv = prepared.Xv0.at[prepared.warp_global_ids].set(Xv_non_surface)
    return Xv.at[prepared.surface_global_ids].set(Xs)
