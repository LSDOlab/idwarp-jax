"""Automatic differentiation utilities for JAX mesh deformation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

Array = jax.Array


def mesh_deformation_jvp(
    deformation_function: Callable[[Array], Array],
    surface_displacement: Array,
    surface_direction: Array,
) -> tuple[Array, Array]:
    """Compute a Jacobian-vector product for the mesh deformation.

    Parameters
    ----------
    deformation_function
        Function mapping surface displacement to volume coordinates:

            dXs -> Xv

    surface_displacement
        Current surface displacement with shape (n_surface_nodes, 3).

    surface_direction
        Perturbation direction in surface displacement space, with the
        same shape as surface_displacement.

    Returns
    -------
    volume_coordinates
        Deformed volume coordinates.

    volume_direction
        Directional derivative:

            dXv/ddXs @ surface_direction
    """
    volume_coordinates, volume_direction = jax.jvp(
        deformation_function,
        primals=(surface_displacement,),
        tangents=(surface_direction,),
    )

    return volume_coordinates, volume_direction


def mesh_deformation_vjp(
    deformation_function: Callable[[Array], Array],
    surface_displacement: Array,
    volume_seed: Array,
) -> tuple[Array, Array]:
    """Compute a vector-Jacobian product for the mesh deformation.

    Parameters
    ----------
    deformation_function
        Function mapping surface displacement to volume coordinates:

            dXs -> Xv

    surface_displacement
        Current surface displacement with shape (n_surface_nodes, 3).

    volume_seed
        Reverse-mode seed with shape (n_volume_nodes, 3). For a scalar
        objective F, this generally corresponds to dF/dXv.

    Returns
    -------
    volume_coordinates
        Deformed volume coordinates.

    surface_sensitivity
        Reverse-mode sensitivity:

            (dXv/ddXs).T @ volume_seed
    """
    volume_coordinates, pullback = jax.vjp(
        deformation_function,
        surface_displacement,
    )

    surface_sensitivity, = pullback(volume_seed)

    return volume_coordinates, surface_sensitivity


def check_jvp_with_finite_difference(
    deformation_function: Callable[[Array], Array],
    surface_displacement: Array,
    surface_direction: Array,
    epsilon: float = 1.0e-6,
) -> dict[str, Array]:
    """Compare a JAX JVP against a centered finite difference."""
    _, jvp_result = mesh_deformation_jvp(
        deformation_function=deformation_function,
        surface_displacement=surface_displacement,
        surface_direction=surface_direction,
    )

    volume_plus = deformation_function(
        surface_displacement + epsilon * surface_direction
    )
    volume_minus = deformation_function(
        surface_displacement - epsilon * surface_direction
    )

    finite_difference = (
        volume_plus - volume_minus
    ) / (2.0 * epsilon)

    difference = jvp_result - finite_difference

    absolute_error = jnp.linalg.norm(difference)
    reference_norm = jnp.linalg.norm(finite_difference)
    relative_error = absolute_error / jnp.maximum(
        reference_norm,
        jnp.asarray(1.0e-14, dtype=reference_norm.dtype),
    )

    return {
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "jvp_norm": jnp.linalg.norm(jvp_result),
        "finite_difference_norm": reference_norm,
    }


def check_jvp_vjp_dot_product(
    deformation_function: Callable[[Array], Array],
    surface_displacement: Array,
    surface_direction: Array,
    volume_seed: Array,
) -> dict[str, Array]:
    """Check consistency between the JVP and VJP.

    This verifies:

        volume_seed . (J @ surface_direction)
        =
        (J.T @ volume_seed) . surface_direction
    """
    _, volume_direction = mesh_deformation_jvp(
        deformation_function=deformation_function,
        surface_displacement=surface_displacement,
        surface_direction=surface_direction,
    )

    _, surface_sensitivity = mesh_deformation_vjp(
        deformation_function=deformation_function,
        surface_displacement=surface_displacement,
        volume_seed=volume_seed,
    )

    forward_product = jnp.vdot(
        volume_seed,
        volume_direction,
    )
    reverse_product = jnp.vdot(
        surface_sensitivity,
        surface_direction,
    )

    scale = jnp.maximum(
        jnp.abs(forward_product),
        jnp.abs(reverse_product),
    )
    scale = jnp.maximum(
        scale,
        jnp.asarray(1.0e-14, dtype=scale.dtype),
    )

    relative_error = (
        jnp.abs(forward_product - reverse_product) / scale
    )

    return {
        "forward_product": forward_product,
        "reverse_product": reverse_product,
        "relative_error": relative_error,
    }
