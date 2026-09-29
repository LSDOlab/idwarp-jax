"""JAX IDW volume-mesh deformation with selectable symmetry handling.

The module exposes one public driver, :func:`deformed_volume_points`, and keeps three internal algorithms separate:

``nonesym``     One-sided IDW using the supplied real surface.
``exactsym``    IDWarp-style exact symmetry using real and mirrored volume-point contributions.
``approxsym``   One-sided IDW followed by a distance-weighted correction of the displacement normal to the symmetry plane.

All geometry operations associated with a symmetry plane are delegated to ``symmetry.py``. 
The numerical kernels use only JAX operations and remain compatible with JAX transformations and automatic differentiation.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from idwarp_jax.symmetry import (
    SYMMETRY_APPROXIMATE,
    SYMMETRY_EXACT,
    SYMMETRY_NONE,
    apply_approximate_symmetry_correction,
    normalize_plane_normal,
    normalize_symmetry_mode,
    reflect_points_about_plane,
    reflect_vectors_about_plane,
)

Array = jax.Array


def compute_idwarp_reference_length(Xs0: Array) -> Array:
    """Return IDWarp's reference surface length ``Ldef0``.

    IDWarp defines ``Ldef0`` as the maximum Euclidean distance from the
    centroid of the unique reference surface nodes to any reference surface
    node. The dimensional length used in the weight is then
    ``Ldef = LdefFact * Ldef0``.
    """
    Xs0 = jnp.asarray(Xs0)
    centroid = jnp.mean(Xs0, axis=0)
    offsets = Xs0 - centroid
    return jnp.max(jnp.sqrt(jnp.sum(offsets * offsets, axis=1)))


def _resolve_ldef(Xs0: Array, LdefFact: float, Ldef0=None) -> Array:
    """Convert the dimensionless IDWarp ``LdefFact`` to ``Ldef``."""
    if Ldef0 is None:
        Ldef0 = compute_idwarp_reference_length(Xs0)
    else:
        Ldef0 = jnp.asarray(Ldef0, dtype=Xs0.dtype)
    return jnp.asarray(LdefFact, dtype=Xs0.dtype) * Ldef0


@partial(
    jax.jit,
    static_argnames=("surface_block_size",),
)
def _warp_one_sided_chunk(
    Xv0_chunk: Array,
    Xs0: Array,
    Ai: Array,
    M: Array,
    b: Array,
    Ldef: float,
    aExp: float = 3.0,
    bExp: float = 5.0,
    alpha: float = 0.25,
    eps: float = 1.0e-30,
    surface_block_size: int = 1024,
    precomputed_denominator: Array | None = None,
) -> Array:
    """Warp one volume chunk using only the supplied real surface."""
    alpha_b = alpha**bExp

    n_surface = Xs0.shape[0]
    n_blocks = (
        n_surface + surface_block_size - 1
    ) // surface_block_size

    has_precomputed_denominator = precomputed_denominator is not None

    def one_volume_point(xv, cached_denominator):
        numerator0 = jnp.zeros((3,), dtype=xv.dtype)
        denominator0 = jnp.asarray(0.0, dtype=xv.dtype)

        def one_surface_block(carry, block_id):
            numerator, denominator = carry

            start = block_id * surface_block_size
            ids = start + jnp.arange(
                surface_block_size,
                dtype=jnp.int32,
            )

            valid = ids < n_surface
            safe_ids = jnp.minimum(ids, n_surface - 1)

            Xs_block = Xs0[safe_ids]
            Ai_block = Ai[safe_ids]
            M_block = M[safe_ids]
            b_block = b[safe_ids]

            difference = xv[None, :] - Xs_block
            distance_squared = (
                jnp.sum(difference * difference, axis=1) + eps
            )
            distance = jnp.sqrt(distance_squared)
            ratio = Ldef / distance

            weights = Ai_block * (
                ratio**aExp + alpha_b * ratio**bExp
            )
            weights = jnp.where(valid, weights, 0.0)

            suggested_displacement = (
                jnp.einsum("sij,j->si", M_block, xv)
                + b_block
                - xv
            )

            block_numerator = jnp.sum(
                weights[:, None] * suggested_displacement,
                axis=0,
            )
            block_denominator = (
                jnp.asarray(0.0, dtype=xv.dtype)
                if has_precomputed_denominator
                else jnp.sum(weights)
            )

            return (
                numerator + block_numerator,
                denominator + block_denominator,
            ), None

        block_ids = jnp.arange(n_blocks, dtype=jnp.int32)
        (numerator, denominator), _ = jax.lax.scan(
            one_surface_block,
            (numerator0, denominator0),
            block_ids,
        )

        if has_precomputed_denominator:
            denominator = cached_denominator

        return xv + numerator / (denominator + eps)

    if not has_precomputed_denominator:
        zero_denominator = jnp.zeros(Xv0_chunk.shape[0], dtype=Xv0_chunk.dtype)
        return jax.vmap(one_volume_point)(Xv0_chunk, zero_denominator)
    return jax.vmap(one_volume_point)(Xv0_chunk, precomputed_denominator)


@partial(
    jax.jit,
    static_argnames=("surface_block_size",),
)
def _warp_exact_symmetry_chunk(
    Xv0_chunk: Array,
    Xs0: Array,
    Ai: Array,
    M: Array,
    b: Array,
    Ldef: float,
    symmetry_plane_point: Array,
    symmetry_plane_normal: Array,
    aExp: float = 3.0,
    bExp: float = 5.0,
    alpha: float = 0.25,
    eps: float = 1.0e-30,
    surface_block_size: int = 1024,
    precomputed_denominator: Array | None = None,
) -> Array:
    """Warp one volume chunk using exact mirrored-point symmetry.

    For a volume point ``x``, the accumulated quantities are

    ``N_total = N(x) + R N(Rx)``
    ``D_total = D(x) + D(Rx)``

    where ``R`` denotes reflection across the symmetry plane. The mirrored
    numerator is a free vector and is reflected back before accumulation.
    """
    alpha_b = alpha**bExp

    n_surface = Xs0.shape[0]
    n_blocks = (
        n_surface + surface_block_size - 1
    ) // surface_block_size

    has_precomputed_denominator = precomputed_denominator is not None

    def one_volume_point(xv, cached_denominator):
        numerator0 = jnp.zeros((3,), dtype=xv.dtype)
        denominator0 = jnp.asarray(0.0, dtype=xv.dtype)

        xv_mirror = reflect_points_about_plane(
            xv,
            symmetry_plane_point,
            symmetry_plane_normal,
            eps=eps,
        )

        def one_surface_block(carry, block_id):
            numerator, denominator = carry

            start = block_id * surface_block_size
            ids = start + jnp.arange(
                surface_block_size,
                dtype=jnp.int32,
            )

            valid = ids < n_surface
            safe_ids = jnp.minimum(ids, n_surface - 1)

            Xs_block = Xs0[safe_ids]
            Ai_block = Ai[safe_ids]
            M_block = M[safe_ids]
            b_block = b[safe_ids]

            difference_real = xv[None, :] - Xs_block
            distance_squared_real = (
                jnp.sum(difference_real * difference_real, axis=1)
                + eps
            )
            distance_real = jnp.sqrt(distance_squared_real)
            ratio_real = Ldef / distance_real

            weights_real = Ai_block * (
                ratio_real**aExp + alpha_b * ratio_real**bExp
            )
            weights_real = jnp.where(valid, weights_real, 0.0)

            suggested_real = (
                jnp.einsum("sij,j->si", M_block, xv)
                + b_block
                - xv
            )
            numerator_real = jnp.sum(
                weights_real[:, None] * suggested_real,
                axis=0,
            )
            denominator_real = (
                jnp.asarray(0.0, dtype=xv.dtype)
                if has_precomputed_denominator
                else jnp.sum(weights_real)
            )

            difference_mirror = xv_mirror[None, :] - Xs_block
            distance_squared_mirror = (
                jnp.sum(
                    difference_mirror * difference_mirror,
                    axis=1,
                )
                + eps
            )
            distance_mirror = jnp.sqrt(distance_squared_mirror)
            ratio_mirror = Ldef / distance_mirror

            weights_mirror = Ai_block * (
                ratio_mirror**aExp
                + alpha_b * ratio_mirror**bExp
            )
            weights_mirror = jnp.where(valid, weights_mirror, 0.0)

            suggested_mirror = (
                jnp.einsum("sij,j->si", M_block, xv_mirror)
                + b_block
                - xv_mirror
            )
            numerator_mirror = jnp.sum(
                weights_mirror[:, None] * suggested_mirror,
                axis=0,
            )
            numerator_mirror = reflect_vectors_about_plane(
                numerator_mirror,
                symmetry_plane_normal,
                eps=eps,
            )
            denominator_mirror = (
                jnp.asarray(0.0, dtype=xv.dtype)
                if has_precomputed_denominator
                else jnp.sum(weights_mirror)
            )

            return (
                numerator + numerator_real + numerator_mirror,
                denominator + denominator_real + denominator_mirror,
            ), None

        block_ids = jnp.arange(n_blocks, dtype=jnp.int32)
        (numerator, denominator), _ = jax.lax.scan(
            one_surface_block,
            (numerator0, denominator0),
            block_ids,
        )

        if has_precomputed_denominator:
            denominator = cached_denominator

        return xv + numerator / (denominator + eps)

    if not has_precomputed_denominator:
        zero_denominator = jnp.zeros(Xv0_chunk.shape[0], dtype=Xv0_chunk.dtype)
        return jax.vmap(one_volume_point)(Xv0_chunk, zero_denominator)
    return jax.vmap(one_volume_point)(Xv0_chunk, precomputed_denominator)


@partial(
    jax.jit,
    static_argnames=("surface_block_size",),
)
def _one_sided_denominator_chunk(
    Xv0_chunk: Array,
    Xs0: Array,
    Ai: Array,
    Ldef: float,
    aExp: float = 3.0,
    bExp: float = 5.0,
    alpha: float = 0.25,
    eps: float = 1.0e-30,
    surface_block_size: int = 1024,
) -> Array:
    """Compute the deformation-independent one-sided IDW denominator."""
    alpha_b = alpha**bExp
    n_surface = Xs0.shape[0]
    n_blocks = (n_surface + surface_block_size - 1) // surface_block_size

    def one_volume_point(xv):
        def one_surface_block(denominator, block_id):
            start = block_id * surface_block_size
            ids = start + jnp.arange(surface_block_size, dtype=jnp.int32)
            valid = ids < n_surface
            safe_ids = jnp.minimum(ids, n_surface - 1)

            difference = xv[None, :] - Xs0[safe_ids]
            distance_squared = jnp.sum(difference * difference, axis=1) + eps
            distance = jnp.sqrt(distance_squared)
            ratio = Ldef / distance
            weights = Ai[safe_ids] * (
                ratio**aExp + alpha_b * ratio**bExp
            )
            weights = jnp.where(valid, weights, 0.0)
            return denominator + jnp.sum(weights), None

        denominator0 = jnp.asarray(0.0, dtype=xv.dtype)
        block_ids = jnp.arange(n_blocks, dtype=jnp.int32)
        denominator, _ = jax.lax.scan(
            one_surface_block,
            denominator0,
            block_ids,
        )
        return denominator

    return jax.vmap(one_volume_point)(Xv0_chunk)


@partial(
    jax.jit,
    static_argnames=("surface_block_size",),
)
def _exact_symmetry_denominator_chunk(
    Xv0_chunk: Array,
    Xs0: Array,
    Ai: Array,
    Ldef: float,
    symmetry_plane_point: Array,
    symmetry_plane_normal: Array,
    aExp: float = 3.0,
    bExp: float = 5.0,
    alpha: float = 0.25,
    eps: float = 1.0e-30,
    surface_block_size: int = 1024,
) -> Array:
    """Compute the deformation-independent exact-symmetry IDW denominator."""
    alpha_b = alpha**bExp
    n_surface = Xs0.shape[0]
    n_blocks = (n_surface + surface_block_size - 1) // surface_block_size

    def one_volume_point(xv):
        xv_mirror = reflect_points_about_plane(
            xv,
            symmetry_plane_point,
            symmetry_plane_normal,
            eps=eps,
        )

        def one_surface_block(denominator, block_id):
            start = block_id * surface_block_size
            ids = start + jnp.arange(surface_block_size, dtype=jnp.int32)
            valid = ids < n_surface
            safe_ids = jnp.minimum(ids, n_surface - 1)

            Xs_block = Xs0[safe_ids]
            Ai_block = Ai[safe_ids]

            difference_real = xv[None, :] - Xs_block
            distance_squared_real = (
                jnp.sum(difference_real * difference_real, axis=1) + eps
            )
            distance_real = jnp.sqrt(distance_squared_real)
            ratio_real = Ldef / distance_real
            weights_real = Ai_block * (
                ratio_real**aExp + alpha_b * ratio_real**bExp
            )
            weights_real = jnp.where(valid, weights_real, 0.0)

            difference_mirror = xv_mirror[None, :] - Xs_block
            distance_squared_mirror = (
                jnp.sum(difference_mirror * difference_mirror, axis=1) + eps
            )
            distance_mirror = jnp.sqrt(distance_squared_mirror)
            ratio_mirror = Ldef / distance_mirror
            weights_mirror = Ai_block * (
                ratio_mirror**aExp + alpha_b * ratio_mirror**bExp
            )
            weights_mirror = jnp.where(valid, weights_mirror, 0.0)

            return (
                denominator
                + jnp.sum(weights_real)
                + jnp.sum(weights_mirror)
            ), None

        denominator0 = jnp.asarray(0.0, dtype=xv.dtype)
        block_ids = jnp.arange(n_blocks, dtype=jnp.int32)
        denominator, _ = jax.lax.scan(
            one_surface_block,
            denominator0,
            block_ids,
        )
        return denominator

    return jax.vmap(one_volume_point)(Xv0_chunk)


def _warp_one_sided_chunks(
    Xv0: Array,
    Xs0: Array,
    Ai: Array,
    M: Array,
    b: Array,
    Ldef: float,
    *,
    aExp: float,
    bExp: float,
    alpha: float,
    eps: float,
    volume_chunk_size: int,
    surface_block_size: int,
    precomputed_denominator: Array | None = None,
) -> Array:
    """Apply the shared one-sided kernel to all volume chunks."""
    chunks = []
    n_volume = Xv0.shape[0]

    for start in range(0, n_volume, volume_chunk_size):
        # print(start)
        end = min(start + volume_chunk_size, n_volume)
        chunks.append(
            _warp_one_sided_chunk(
                Xv0[start:end],
                Xs0,
                Ai,
                M,
                b,
                Ldef,
                aExp=aExp,
                bExp=bExp,
                alpha=alpha,
                eps=eps,
                surface_block_size=surface_block_size,
                precomputed_denominator=(
                    None
                    if precomputed_denominator is None
                    else precomputed_denominator[start:end]
                ),
            )
        )

    return jnp.concatenate(chunks, axis=0)


def _warp_without_symmetry(
    Xv0: Array,
    Xs0: Array,
    Ai: Array,
    M: Array,
    b: Array,
    Ldef: float,
    *,
    aExp: float,
    bExp: float,
    alpha: float,
    eps: float,
    volume_chunk_size: int,
    surface_block_size: int,
    precomputed_denominator: Array | None = None,
) -> Array:
    """Warp using the supplied surface with no symmetry treatment."""
    return _warp_one_sided_chunks(
        Xv0,
        Xs0,
        Ai,
        M,
        b,
        Ldef,
        aExp=aExp,
        bExp=bExp,
        alpha=alpha,
        eps=eps,
        volume_chunk_size=volume_chunk_size,
        surface_block_size=surface_block_size,
        precomputed_denominator=precomputed_denominator,
    )


def _warp_exact_symmetry(
    Xv0: Array,
    Xs0: Array,
    Ai: Array,
    M: Array,
    b: Array,
    Ldef: float,
    *,
    symmetry_plane_point: Array,
    symmetry_plane_normal: Array,
    aExp: float,
    bExp: float,
    alpha: float,
    eps: float,
    volume_chunk_size: int,
    surface_block_size: int,
    precomputed_denominator: Array | None = None,
) -> Array:
    """Apply the exact-symmetry kernel to all volume chunks."""
    chunks = []
    n_volume = Xv0.shape[0]

    for start in range(0, n_volume, volume_chunk_size):
        print(start)
        end = min(start + volume_chunk_size, n_volume)
        chunks.append(
            _warp_exact_symmetry_chunk(
                Xv0[start:end],
                Xs0,
                Ai,
                M,
                b,
                Ldef,
                symmetry_plane_point,
                symmetry_plane_normal,
                aExp=aExp,
                bExp=bExp,
                alpha=alpha,
                eps=eps,
                surface_block_size=surface_block_size,
                precomputed_denominator=(
                    None
                    if precomputed_denominator is None
                    else precomputed_denominator[start:end]
                ),
            )
        )

    return jnp.concatenate(chunks, axis=0)


def _warp_approximate_symmetry(
    Xv0: Array,
    Xs0: Array,
    Ai: Array,
    M: Array,
    b: Array,
    Ldef: float,
    *,
    symmetry_plane_point: Array,
    symmetry_plane_normal: Array,
    symmetry_length_scale: float,
    aExp: float,
    bExp: float,
    alpha: float,
    eps: float,
    volume_chunk_size: int,
    surface_block_size: int,
    precomputed_denominator: Array | None = None,
) -> Array:
    """Apply one-sided IDW and then approximate symmetry correction."""
    Xv_raw = _warp_one_sided_chunks(
        Xv0,
        Xs0,
        Ai,
        M,
        b,
        Ldef,
        aExp=aExp,
        bExp=bExp,
        alpha=alpha,
        eps=eps,
        volume_chunk_size=volume_chunk_size,
        surface_block_size=surface_block_size,
        precomputed_denominator=precomputed_denominator,
    )

    return apply_approximate_symmetry_correction(
        original_points=Xv0,
        deformed_points=Xv_raw,
        plane_point=symmetry_plane_point,
        plane_normal=symmetry_plane_normal,
        length_scale=symmetry_length_scale,
        eps=eps,
    )


def _prepare_symmetry_plane(
    Xv0: Array,
    symmetry_plane_point,
    symmetry_plane_normal,
    eps: float,
) -> tuple[Array, Array]:
    """Convert and normalize the symmetry-plane definition."""
    if symmetry_plane_point is None or symmetry_plane_normal is None:
        raise ValueError(
            "A symmetry plane point and normal are required for "
            "'exactsym' and 'approxsym'."
        )

    plane_point = jnp.asarray(
        symmetry_plane_point,
        dtype=Xv0.dtype,
    )
    plane_normal = jnp.asarray(
        symmetry_plane_normal,
        dtype=Xv0.dtype,
    )
    plane_normal = normalize_plane_normal(plane_normal, eps=eps)

    return plane_point, plane_normal


def precompute_idw_denominator(
    Xv0: Array,
    Xs0: Array,
    Ai: Array,
    LdefFact: float = 1.0,
    Ldef0=None,
    aExp: float = 3.0,
    bExp: float = 5.0,
    alpha: float = 0.25,
    eps: float = 1.0e-30,
    volume_chunk_size: int = 512,
    surface_block_size: int = 1024,
    symmetry_mode: str | None = SYMMETRY_NONE,
    symmetry_plane_point=None,
    symmetry_plane_normal=None,
) -> Array:
    """Precompute the fixed IDW denominator for a reference volume mesh.

    The denominator depends only on the undeformed volume and surface geometry,
    area weights, and IDW options. It can therefore be reused for every surface
    deformation of the same prepared mesh.
    """
    Xv0 = jnp.asarray(Xv0)
    Xs0 = jnp.asarray(Xs0, dtype=Xv0.dtype)
    Ai = jnp.asarray(Ai, dtype=Xv0.dtype)
    mode = normalize_symmetry_mode(symmetry_mode)
    Ldef = _resolve_ldef(Xs0, LdefFact, Ldef0)

    if Xv0.shape[0] == 0:
        return jnp.zeros((0,), dtype=Xv0.dtype)

    if mode != SYMMETRY_NONE:
        plane_point, plane_normal = _prepare_symmetry_plane(
            Xv0,
            symmetry_plane_point,
            symmetry_plane_normal,
            eps,
        )

    chunks = []
    for start in range(0, Xv0.shape[0], volume_chunk_size):
        end = min(start + volume_chunk_size, Xv0.shape[0])
        Xv0_chunk = Xv0[start:end]

        if mode == SYMMETRY_EXACT:
            denominator_chunk = _exact_symmetry_denominator_chunk(
                Xv0_chunk,
                Xs0,
                Ai,
                Ldef,
                plane_point,
                plane_normal,
                aExp=aExp,
                bExp=bExp,
                alpha=alpha,
                eps=eps,
                surface_block_size=surface_block_size,
            )
        else:
            denominator_chunk = _one_sided_denominator_chunk(
                Xv0_chunk,
                Xs0,
                Ai,
                Ldef,
                aExp=aExp,
                bExp=bExp,
                alpha=alpha,
                eps=eps,
                surface_block_size=surface_block_size,
            )

        chunks.append(denominator_chunk)

    return jnp.concatenate(chunks, axis=0)


def deformed_volume_points(
    Xv0: Array,
    Xs0: Array,
    Ai: Array,
    M: Array,
    b: Array,
    LdefFact: float = 1.0,
    Ldef0=None,
    aExp: float = 3.0,
    bExp: float = 5.0,
    alpha: float = 0.25,
    eps: float = 1.0e-30,
    volume_chunk_size: int = 512,
    surface_block_size: int = 1024,
    symmetry_mode: str | None = SYMMETRY_NONE,
    symmetry_plane_point=None,
    symmetry_plane_normal=None,
    symmetry_length_scale: float | None = None,
    precomputed_denominator: Array | None = None,
) -> Array:
    """Deform volume coordinates using one selected symmetry algorithm.

    Parameters
    ----------
    Xv0                         Original volume coordinates, shape ``(n_volume, 3)``.
    Xs0                         Original surface coordinates, shape ``(n_surface, 3)``.
    Ai                          Area weight for each surface point, shape ``(n_surface,)``.
    M                           Local transformation matrix for each surface point, shape        ``(n_surface, 3, 3)``.
    b                           Local translation vector for each surface point, shape        ``(n_surface, 3)``.
    LdefFact                    Dimensionless IDWarp deformation-length factor.
                                The dimensional length is ``LdefFact * Ldef0``.
    Ldef0                       Optional precomputed IDWarp reference length. If omitted,
                                it is computed from ``Xs0`` as the maximum distance from
                                the reference-surface centroid to any surface node.
    symmetry_mode               One of ``"nonesym"``, ``"exactsym"``, or ``"approxsym"``.
    symmetry_plane_point        Required by exact and approximate symmetry modes.
    symmetry_plane_normal       Required by exact and approximate symmetry modes.
    symmetry_length_scale       Required by approximate symmetry. It controls the distance over which the normal displacement recovers from zero to its one-sided value.
    precomputed_denominator     Optional result from :func:`precompute_idw_denominator`
                                for the same volume points and IDW configuration.

    Returns
    -------
    Array
        Deformed volume coordinates with the same shape as ``Xv0``.
    """
    mode = normalize_symmetry_mode(symmetry_mode)
    Ldef = _resolve_ldef(Xs0, LdefFact, Ldef0)
    if precomputed_denominator is not None:
        precomputed_denominator = jnp.asarray(
            precomputed_denominator,
            dtype=Xv0.dtype,
        )
        if precomputed_denominator.shape != (Xv0.shape[0],):
            raise ValueError(
                "precomputed_denominator must have shape "
                f"{(Xv0.shape[0],)}; got {precomputed_denominator.shape}"
            )

    if mode == SYMMETRY_NONE:
        return _warp_without_symmetry(
            Xv0,
            Xs0,
            Ai,
            M,
            b,
            Ldef,
            aExp=aExp,
            bExp=bExp,
            alpha=alpha,
            eps=eps,
            volume_chunk_size=volume_chunk_size,
            surface_block_size=surface_block_size,
            precomputed_denominator=precomputed_denominator,
        )

    plane_point, plane_normal = _prepare_symmetry_plane(
        Xv0,
        symmetry_plane_point,
        symmetry_plane_normal,
        eps,
    )

    if mode == SYMMETRY_EXACT:
        return _warp_exact_symmetry(
            Xv0,
            Xs0,
            Ai,
            M,
            b,
            Ldef,
            symmetry_plane_point=plane_point,
            symmetry_plane_normal=plane_normal,
            aExp=aExp,
            bExp=bExp,
            alpha=alpha,
            eps=eps,
            volume_chunk_size=volume_chunk_size,
            surface_block_size=surface_block_size,
            precomputed_denominator=precomputed_denominator,
        )

    if symmetry_length_scale is None:
        raise ValueError(
            "symmetry_length_scale is required for 'approxsym'."
        )

    return _warp_approximate_symmetry(
        Xv0,
        Xs0,
        Ai,
        M,
        b,
        Ldef,
        symmetry_plane_point=plane_point,
        symmetry_plane_normal=plane_normal,
        symmetry_length_scale=symmetry_length_scale,
        aExp=aExp,
        bExp=bExp,
        alpha=alpha,
        eps=eps,
        volume_chunk_size=volume_chunk_size,
        surface_block_size=surface_block_size,
        precomputed_denominator=precomputed_denominator,
    )
