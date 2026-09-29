"""Reusable symmetry-plane geometry operations for JAX mesh deformation.

The functions in this module are shared by the public driver and the internal
exact/approximate warp algorithms. Geometry functions operate on arrays whose
last dimension is three and preserve JAX transformations and differentiation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array

SYMMETRY_NONE = "nonesym"
SYMMETRY_EXACT = "exactsym"
SYMMETRY_APPROXIMATE = "approxsym"
VALID_SYMMETRY_MODES = (
    SYMMETRY_NONE,
    SYMMETRY_EXACT,
    SYMMETRY_APPROXIMATE,
)


def normalize_symmetry_mode(mode: str | None) -> str:
    """Return a canonical public symmetry-mode name.

    Parameters
    ----------
    mode
        ``None`` or one of ``"nonesym"``, ``"exactsym"``, and ``"approxsym"``.
    """
    if mode is None:
        return SYMMETRY_NONE

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in VALID_SYMMETRY_MODES:
        valid = ", ".join(VALID_SYMMETRY_MODES)
        raise ValueError(
            f"Unknown symmetry mode {mode!r}. Expected one of: {valid}."
        )
    return normalized_mode


@jax.jit
def normalize_plane_normal(plane_normal: Array, eps: float = 1.0e-30) -> Array:
    """Normalize a symmetry-plane normal along its last axis."""
    plane_normal = jnp.asarray(plane_normal)
    norm = jnp.sqrt(
        jnp.sum(plane_normal * plane_normal, axis=-1, keepdims=True) + eps
    )
    return plane_normal / norm


@jax.jit
def signed_distance_to_plane(
    points: Array,
    plane_point: Array,
    plane_normal: Array,
    eps: float = 1.0e-30,
) -> Array:
    """Compute signed point-to-plane distance.

    ``plane_normal`` is normalized internally. ``points`` may have shape
    ``(3,)`` or ``(..., 3)``.
    """
    points = jnp.asarray(points)
    plane_point = jnp.asarray(plane_point, dtype=points.dtype)
    plane_normal = jnp.asarray(plane_normal, dtype=points.dtype)
    normal = normalize_plane_normal(plane_normal, eps=eps)
    return jnp.sum((points - plane_point) * normal, axis=-1)


@jax.jit
def points_on_plane(
    points: Array,
    plane_point: Array,
    plane_normal: Array,
    tolerance: float,
    eps: float = 1.0e-30,
) -> Array:
    """Return a boolean mask identifying points within a plane tolerance."""
    distance = signed_distance_to_plane(
        points,
        plane_point,
        plane_normal,
        eps=eps,
    )
    tolerance = jnp.asarray(tolerance, dtype=distance.dtype)
    return jnp.abs(distance) <= tolerance


@jax.jit
def project_points_to_plane(
    points: Array,
    plane_point: Array,
    plane_normal: Array,
    eps: float = 1.0e-30,
) -> Array:
    """Orthogonally project points onto a plane."""
    points = jnp.asarray(points)
    plane_normal = jnp.asarray(plane_normal, dtype=points.dtype)
    normal = normalize_plane_normal(plane_normal, eps=eps)
    distance = signed_distance_to_plane(
        points,
        plane_point,
        normal,
        eps=eps,
    )
    return points - distance[..., None] * normal


@jax.jit
def reflect_points_about_plane(
    points: Array,
    plane_point: Array,
    plane_normal: Array,
    eps: float = 1.0e-30,
) -> Array:
    """Reflect points across a plane."""
    points = jnp.asarray(points)
    plane_normal = jnp.asarray(plane_normal, dtype=points.dtype)
    normal = normalize_plane_normal(plane_normal, eps=eps)
    distance = signed_distance_to_plane(
        points,
        plane_point,
        normal,
        eps=eps,
    )
    return points - 2.0 * distance[..., None] * normal


@jax.jit
def reflect_vectors_about_plane(
    vectors: Array,
    plane_normal: Array,
    eps: float = 1.0e-30,
) -> Array:
    """Reflect free vectors across a plane through the origin."""
    vectors = jnp.asarray(vectors)
    plane_normal = jnp.asarray(plane_normal, dtype=vectors.dtype)
    normal = normalize_plane_normal(plane_normal, eps=eps)
    normal_amplitude = jnp.sum(vectors * normal, axis=-1)
    return vectors - 2.0 * normal_amplitude[..., None] * normal


# Singular aliases retain the terminology used by the exact reference warp.
reflect_point_about_plane = reflect_points_about_plane
reflect_vector_about_plane = reflect_vectors_about_plane


@jax.jit
def constrain_points_originally_on_plane(
    original_points: Array,
    target_points: Array,
    plane_point: Array,
    plane_normal: Array,
    tolerance: float,
    eps: float = 1.0e-30,
) -> Array:
    """Project target points whose original locations lie on the plane.

    The mask is determined from ``original_points`` and is therefore fixed for
    a fixed mesh topology. Points outside the mask are returned unchanged.
    """
    original_points = jnp.asarray(original_points)
    target_points = jnp.asarray(target_points, dtype=original_points.dtype)

    on_plane = points_on_plane(
        original_points,
        plane_point,
        plane_normal,
        tolerance,
        eps=eps,
    )
    projected_target = project_points_to_plane(
        target_points,
        plane_point,
        plane_normal,
        eps=eps,
    )
    return jnp.where(on_plane[..., None], projected_target, target_points)


@jax.jit
def apply_approximate_symmetry_correction_OLD(
    original_points: Array,
    deformed_points: Array,
    plane_point: Array,
    plane_normal: Array,
    length_scale: float,
    eps: float = 1.0e-30,
) -> Array:
    """Suppress normal displacement near a symmetry plane.

    The tangential displacement is unchanged. The normal displacement uses a
    linear recovery factor: zero on the plane and one at distances greater
    than or equal to ``length_scale``.
    """
    original_points = jnp.asarray(original_points)
    deformed_points = jnp.asarray(deformed_points, dtype=original_points.dtype)
    plane_normal = jnp.asarray(plane_normal, dtype=original_points.dtype)
    normal = normalize_plane_normal(plane_normal, eps=eps)

    displacement = deformed_points - original_points
    distance = jnp.abs(
        signed_distance_to_plane(
            original_points,
            plane_point,
            normal,
            eps=eps,
        )
    )

    length_scale = jnp.asarray(length_scale, dtype=original_points.dtype)
    normal_factor = jnp.clip(
        distance / jnp.maximum(length_scale, eps),
        0.0,
        1.0,
    )

    normal_amplitude = jnp.sum(displacement * normal, axis=-1)
    normal_displacement = normal_amplitude[..., None] * normal
    tangential_displacement = displacement - normal_displacement

    corrected_displacement = (
        tangential_displacement
        + normal_factor[..., None] * normal_displacement
    )
    return original_points + corrected_displacement


@jax.jit
def apply_approximate_symmetry_correction(
    original_points: Array,
    deformed_points: Array,
    plane_point: Array,
    plane_normal: Array,
    length_scale: float,
    eps: float = 1.0e-30,
) -> Array:
    """Suppress normal displacement using a local IDW-style blend.

    ``length_scale`` is retained for API compatibility but is not used.

    The transition scale is the local proposed normal displacement:
      - zero normal displacement on the symmetry plane
      - approximately unchanged when distance from the plane is much larger
        than the proposed normal motion
    """
    original_points = jnp.asarray(original_points)
    deformed_points = jnp.asarray(
        deformed_points,
        dtype=original_points.dtype,
    )
    plane_normal = jnp.asarray(
        plane_normal,
        dtype=original_points.dtype,
    )

    normal = normalize_plane_normal(plane_normal, eps=eps)

    displacement = deformed_points - original_points

    distance = jnp.abs(
        signed_distance_to_plane(
            original_points,
            plane_point,
            normal,
            eps=eps,
        )
    )

    normal_amplitude = jnp.sum(
        displacement * normal,
        axis=-1,
    )

    normal_displacement = (
        normal_amplitude[..., None] * normal
    )

    tangential_displacement = (
        displacement - normal_displacement
    )

    # Parameter-free inverse-distance-style blending.
    #
    # distance >> normal motion: factor -> 1
    # distance << normal motion: factor -> 0
    # distance == 0:             factor == 0
    distance_squared = distance * distance
    normal_motion_squared = normal_amplitude * normal_amplitude

    normal_factor = distance_squared / (
        distance_squared
        + normal_motion_squared
        + eps
    )

    corrected_displacement = (
        tangential_displacement
        + normal_factor[..., None] * normal_displacement
    )

    return original_points + corrected_displacement

# Compatibility alias matching the approximate reference implementation.
apply_symmetry_normal_correction = apply_approximate_symmetry_correction
