"""Adapter and simple wall bump for OpenFOAM-like mesh dictionaries.

This module expects the mesh dictionary described by the user:

    mesh["points"]                  -> (n_points, 3) NumPy array
    mesh["faces"]["points"]       -> list of global point-ID lists
    mesh["faces"]["type"]         -> array, where 2 means wall

The moving surface is built from wall faces only.  Symmetry boundary faces are
not included in the moving surface; exact symmetry mirrors the wall influence
about y=0 and pins moving-surface nodes already on y=0 to that plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import jax

# The supplied mesh coordinates are float64. Enable this before creating JAX arrays.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from idwarp_jax.driver import deform_mesh


@dataclass(frozen=True)
class SurfaceData:
    """Local moving-surface arrays required by IDWarp-JAX."""

    Xs0: np.ndarray
    conn: np.ndarray
    face_sizes: np.ndarray
    surface_global_ids: np.ndarray
    selected_face_ids: np.ndarray


def build_surface_data(
    mesh: dict,
    moving_face_types: Iterable[int] = (2,),
) -> SurfaceData:
    """Extract moving boundary faces and remap global point IDs to local IDs.

    For the face-type convention supplied by the user, ``moving_face_types=(2,)``
    selects wall faces. Do not include type 3 symmetry-plane faces unless the
    plane itself is intentionally a prescribed moving boundary.
    """
    points = np.asarray(mesh["points"], dtype=np.float64)
    face_points: Sequence[Sequence[int]] = mesh["faces"]["points"]
    face_types = np.asarray(mesh["faces"]["type"])

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"mesh['points'] must have shape (n, 3); got {points.shape}")
    if face_types.shape[0] != len(face_points):
        raise ValueError("faces['type'] and faces['points'] have different lengths")

    requested_types = np.asarray(tuple(moving_face_types), dtype=face_types.dtype)
    selected_face_ids = np.flatnonzero(np.isin(face_types, requested_types))
    if selected_face_ids.size == 0:
        raise ValueError(f"No faces found with type(s) {requested_types.tolist()}")

    selected_faces = [
        np.asarray(face_points[int(face_id)], dtype=np.int64)
        for face_id in selected_face_ids
    ]
    if any(face.size < 3 for face in selected_faces):
        raise ValueError("Every moving surface face must have at least 3 points")

    face_sizes = np.fromiter(
        (face.size for face in selected_faces),
        dtype=np.int32,
        count=len(selected_faces),
    )
    flat_global_conn = np.concatenate(selected_faces)

    if np.any(flat_global_conn < 0) or np.any(flat_global_conn >= points.shape[0]):
        raise ValueError("A moving face contains a point ID outside mesh['points']")

    # np.unique sorts the IDs, which lets searchsorted perform a fast global-to-local map.
    surface_global_ids = np.unique(flat_global_conn)
    if surface_global_ids[-1] > np.iinfo(np.int32).max:
        raise ValueError("Point IDs exceed the int32 indexing supported by this codebase")

    conn = np.searchsorted(surface_global_ids, flat_global_conn)
    if not np.array_equal(surface_global_ids[conn], flat_global_conn):
        raise RuntimeError("Failed to remap global surface connectivity to local IDs")

    surface_global_ids = surface_global_ids.astype(np.int32, copy=False)
    conn = conn.astype(np.int32, copy=False)
    Xs0 = points[surface_global_ids]

    return SurfaceData(
        Xs0=Xs0,
        conn=conn,
        face_sizes=face_sizes,
        surface_global_ids=surface_global_ids,
        selected_face_ids=selected_face_ids.astype(np.int32, copy=False),
    )


def make_gaussian_bump(
    Xs0: np.ndarray,
    *,
    amplitude: float,
    center_x: float | None = None,
    width_x: float | None = None,
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    symmetry_plane_normal: Sequence[float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """Create a smooth, symmetry-compatible displacement of the moving wall.

    The bump profile varies only in x. The requested direction is projected
    into the symmetry plane, guaranteeing no displacement normal to y=0.
    """
    Xs0 = np.asarray(Xs0, dtype=np.float64)
    x = Xs0[:, 0]
    x_range = float(np.ptp(x))
    if x_range <= 0.0:
        raise ValueError("The moving surface has zero x extent")

    if center_x is None:
        center_x = 0.5 * float(x.min() + x.max())
    if width_x is None:
        width_x = 0.15 * x_range
    if width_x <= 0.0:
        raise ValueError("width_x must be positive")

    normal = np.asarray(symmetry_plane_normal, dtype=np.float64)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0.0:
        raise ValueError("symmetry_plane_normal must be nonzero")
    normal /= normal_norm

    tangent = np.asarray(direction, dtype=np.float64)
    if tangent.shape != (3,):
        raise ValueError("direction must contain three components")
    tangent = tangent - np.dot(tangent, normal) * normal
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1.0e-14:
        raise ValueError("direction is normal to the symmetry plane; choose x or z")
    tangent /= tangent_norm

    profile = np.exp(-0.5 * ((x - center_x) / width_x) ** 2)
    return float(amplitude) * profile[:, None] * tangent[None, :]


def run_simple_deformation(
    mesh: dict,
    *,
    amplitude: float,
    center_x: float | None = None,
    width_x: float | None = None,
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    moving_face_types: Iterable[int] = (2,),
    symmetry_tolerance: float = 1.0e-10,
    LdefFact: float = 1.0,
    volume_chunk_size: int = 512,
    surface_block_size: int = 1024,
) -> tuple[np.ndarray, SurfaceData, np.ndarray]:
    """Warp the full volume mesh using a smooth wall bump and exact y symmetry.

    Returns
    -------
    deformed_points
        NumPy array with shape equal to mesh['points'].shape.
    surface
        Extracted/remapped moving-surface data.
    surface_displacement
        Prescribed displacement at each local moving-surface point.
    """
    Xv0 = np.asarray(mesh["points"], dtype=np.float64)
    surface = build_surface_data(mesh, moving_face_types=moving_face_types)

    surface_displacement = make_gaussian_bump(
        surface.Xs0,
        amplitude=amplitude,
        center_x=center_x,
        width_x=width_x,
        direction=direction,
        symmetry_plane_normal=(0.0, 1.0, 0.0),
    )

    n_volume = Xv0.shape[0]
    n_surface = surface.Xs0.shape[0]
    n_warp = n_volume - n_surface
    approximate_pair_evaluations = 2 * n_warp * n_surface  # factor 2 for exact symmetry

    plane_mask = np.abs(surface.Xs0[:, 1]) <= symmetry_tolerance
    max_plane_dy = (
        float(np.max(np.abs(surface_displacement[plane_mask, 1])))
        if np.any(plane_mask)
        else 0.0
    )

    print(f"Volume points:             {n_volume:,}")
    print(f"Moving wall faces:         {surface.face_sizes.size:,}")
    print(f"Unique moving wall points: {n_surface:,}")
    print(f"Wall points on y=0:        {int(plane_mask.sum()):,}")
    print(f"Maximum prescribed |dy| on y=0: {max_plane_dy:.3e}")
    print(
        "Approx. exact-symmetry point/surface interactions: "
        f"{approximate_pair_evaluations:.3e}"
    )
    if approximate_pair_evaluations > 1.0e9:
        print(
            "WARNING: this is a large global-IDW calculation. Test the code on "
            "a reduced mesh first and expect the full run to require substantial compute."
        )

    Xv = deform_mesh(
        Xv0=jnp.asarray(Xv0),
        Xs0=jnp.asarray(surface.Xs0),
        surface_displacement=jnp.asarray(surface_displacement),
        conn=jnp.asarray(surface.conn, dtype=jnp.int32),
        face_sizes=jnp.asarray(surface.face_sizes, dtype=jnp.int32),
        surface_global_ids=jnp.asarray(surface.surface_global_ids, dtype=jnp.int32),
        LdefFact=float(LdefFact),
        volume_chunk_size=int(volume_chunk_size),
        surface_block_size=int(surface_block_size),
        symmetry_mode="exactsym",
        symmetry_plane_point=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64),
        symmetry_plane_normal=jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64),
        symmetry_tolerance=float(symmetry_tolerance),
    )

    # Materialize the lazy JAX result and return it in the same host-side form as the input.
    deformed_points = np.asarray(Xv)

    # Basic checks.
    if not np.all(np.isfinite(deformed_points)):
        raise FloatingPointError("The deformed mesh contains NaN or infinite coordinates")

    prescribed_target = surface.Xs0 + surface_displacement
    prescribed_target[plane_mask, 1] = 0.0
    surface_error = np.max(
        np.abs(deformed_points[surface.surface_global_ids] - prescribed_target)
    )
    print(f"Maximum prescribed-surface error: {surface_error:.3e}")

    volume_plane_mask = np.abs(Xv0[:, 1]) <= symmetry_tolerance
    if np.any(volume_plane_mask):
        max_plane_y = float(np.max(np.abs(deformed_points[volume_plane_mask, 1])))
        print(f"Maximum final |y| for original y=0 points: {max_plane_y:.3e}")

    return deformed_points, surface, surface_displacement


# Notebook usage:
#
# from idwarp_jax.mesh_dict_adapter import run_simple_deformation
# deformed_points, surface, dXs = run_simple_deformation(
#     mesh,
#     amplitude=0.01,       # use your mesh length units
#     direction=(0, 0, 1), # z bump, tangent to the y=0 symmetry plane
# )
# np.save("deformed_points.npy", deformed_points)
