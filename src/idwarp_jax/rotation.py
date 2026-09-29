"""
Local rotation and translation calculations for JAX mesh deformation. 
This module computes the local rigid transformation associated with each surface node. 
The transformation is represented by x -> M_i x + b_i, 
where ``M_i`` rotates the original surface normal to the deformed surface normal, 
and ``b_i`` ensures that the original surface point is mapped exactly to its deformed position. 

All numerical operations use JAX and remain compatible with JAX transformations and automatic differentiation. 
"""

import jax
import jax.numpy as jnp
from idwarp_jax.normal import get_normals_Ai

#Use below function when useRotations is false
def get_MB_no_rotation(Xs0, Xs):
    """
    Compute local transformations without surface-normal rotations. 
    This version uses M_i = I b_i = Xs_i - Xs0_i so that every surface point contributes translation only. 
    
    Parameters 
    Xs0         Original surface coordinates with shape ``(n_surface, 3)``. 
    Xs          Deformed surface coordinates with shape ``(n_surface, 3)``. 
    
    Returns 
    M           Identity transformation matrix for each surface point, with shape ``(n_surface, 3, 3)``. 
    b           Translation vector for each surface point, with shape ``(n_surface, 3)``. 
    """
    n_surf = Xs0.shape[0]
    dtype = Xs0.dtype

    M = jnp.tile(jnp.eye(3, dtype=dtype)[None, :, :], (n_surf, 1, 1))
    b = Xs - Xs0

    return M, b


def compute_b_from_M(Xs0, Xs, M):
    """
    Compute translation vectors from surface coordinates and rotations. 
    The translation is defined by b_i = Xs_i - M_i Xs0_i, which ensures that the local transformation maps each original surface point exactly to its deformed position. 
    
    Parameters 
    Xs0         Original surface coordinates with shape ``(n_surface, 3)``. 
    Xs          Deformed surface coordinates with shape ``(n_surface, 3)``. 
    M           Local rotation matrices with shape ``(n_surface, 3, 3)``. 

    Returns
    b           Local translation vectors with shape ``(n_surface, 3)``. 
    """
    MXs0 = jnp.einsum("sij,sj->si", M, Xs0)
    b = Xs - MXs0
    return b


def _cross_product_matrix(v):
    """
    Construct the skew-symmetric cross-product matrix of a vector. 
    The returned matrix ``K`` satisfies K @ x = v cross x. This matrix is used in Rodrigues' rotation formula. 
    
    Parameters
    v           Three-dimensional vector with shape ``(3,)``. 
    
    Returns
    K           Skew-symmetric matrix with shape ``(3, 3)``. 
    """
    vx, vy, vz = v[0], v[1], v[2]

    return jnp.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ],
        dtype=v.dtype,
    )


def _normalize(vec, eps=1e-30): 
    """Normalize a vector using a small regularization term."""    
    return vec / jnp.sqrt(jnp.sum(vec * vec) + eps)


def _orthogonal_axis(n, eps=1e-12):
    """
    Choose a stable unit axis approximately orthogonal to ``n``. 
    This axis is used to define the rotation axis when the original and deformed normals point in opposite directions and a 180-degree rotation is required. 
    
    Parameters
    n           Unit or approximately unit vector with shape ``(3,)``. 
    eps         Small positive regularization used during normalization. 
    
    Returns
    axis        Unit vector approximately orthogonal to ``n``. 
    """
    dtype = n.dtype

    ex = jnp.array([1.0, 0.0, 0.0], dtype=dtype)
    ey = jnp.array([0.0, 1.0, 0.0], dtype=dtype)

    # Use the y-axis when n is close to the x-axis. Otherwise, use the x-axis. This avoids taking the cross product with a nearly parallel reference vector.
    base = jnp.where(jnp.abs(n[0]) > 0.9, ey, ex)

    axis = jnp.cross(n, base)
    return _normalize(axis, eps=eps)


# def rodrigues_formula(n0, n, eps=1e-12):
#     """
#     Compute a rotation matrix that maps ``n0`` to ``n``. 
#     Rodrigues' rotation formula is used for the general case. 
#     Nearly parallel and nearly opposite normals are handled separately to avoid numerical instability. 
    
#     Parameters
#     n0          Original normal vector with shape ``(3,)``. 
#     n           Deformed normal vector with shape ``(3,)``. 
#     eps         Numerical tolerance used for normalization and direction tests. 
    
#     Returns 
#     R           Rotation matrix with shape ``(3, 3)``. 
#     """
#     dtype = n0.dtype

#     a = _normalize(n0, eps=eps)
#     b = _normalize(n, eps=eps)

#     I = jnp.eye(3, dtype=dtype)

#     # Rotation-axis information from the original and deformed normals.
#     v = jnp.cross(a, b)
#     c = jnp.dot(a, b) 
#     s2 = jnp.dot(v, v) 

#     K = _cross_product_matrix(v)

#     # General Rodrigues formula:
#     # R = I + K + K^2 * (1 - c) / s^2
#     R_general = I + K + (K @ K) * ((1.0 - c) / (s2 + eps))

#     # Nearly identical normals require no rotation.
#     R_same = I

#     # Nearly opposite normals require a 180-degree rotation around an axis orthogonal to the original normal.
#     u = _orthogonal_axis(a, eps=eps)
#     R_opposite = -I + 2.0 * jnp.outer(u, u)

#     R = jnp.where(
#         c > 1.0 - eps,
#         R_same,
#         jnp.where(c < -1.0 + eps, R_opposite, R_general),
#     )

#     return R


def rodrigues_formula(n0, n, eps=1e-6):
    dtype = n0.dtype
    tiny = jnp.asarray(1e-7, dtype=dtype)

    a = n0 / jnp.sqrt(jnp.sum(n0 * n0) + tiny)
    b = n  / jnp.sqrt(jnp.sum(n * n) + tiny)

    I = jnp.eye(3, dtype=dtype)
    v = jnp.cross(a, b)
    c = jnp.clip(jnp.dot(a, b), -1.0, 1.0)
    K = _cross_product_matrix(v)

    # Smooth and well behaved when a and b are nearly identical.
    R_general = I + K + (K @ K) / (1.0 + c + tiny)

    # Only retain a special branch for nearly opposite normals.
    u = _orthogonal_axis(a, eps=tiny)
    R_opposite = -I + 2.0 * jnp.outer(u, u)

    return jnp.where(c < -1.0 + eps, R_opposite, R_general)

def rotation_matrices(normals0, normals, eps=1e-12):
    """
    Compute one rotation matrix for each pair of surface normals. 
    
    Parameters 
    normals0    Original unit node normals with shape ``(n_surface, 3)``. 
    normals     Deformed unit node normals with shape ``(n_surface, 3)``. 
    eps         Numerical tolerance passed to ``rodrigues_formula``. 
    
    Returns
    M           Local rotation matrices with shape ``(n_surface, 3, 3)``. 
    """
    return jax.vmap(
        lambda n0, n:rodrigues_formula(n0, n, eps=eps)
    )(normals0, normals)


def get_MB_rotation(
    Xs0,
    Xs,
    normals0,
    normals,
    eps=1e-12,
    corner_mask=None,
):
    """
    Compute local rotation matrices and translation vectors. 
    Each matrix ``M_i`` maps the original node normal ``normals0[i]`` to the corresponding deformed node normal ``normals[i]``. 
    The translation vector is then computed as b_i = Xs_i - M_i Xs0_i. 
    
    Parameters 
    Xs0         Original surface coordinates with shape ``(n_surface, 3)``. 
    Xs          Deformed surface coordinates with shape ``(n_surface, 3)``. 
    normals0    Original unit node normals with shape ``(n_surface, 3)``. 
    normals     Deformed unit node normals with shape ``(n_surface, 3)``. 
    eps         Numerical tolerance used by the rotation calculations.
    corner_mask Optional boolean mask with shape ``(n_surface,)``. Where true,
                the local rotation is forced to identity, matching IDWarp's
                ``zeroCornerRotations`` behavior. Translation is still retained.
    
    Returns 
    M           Local rotation matrices with shape ``(n_surface, 3, 3)``. 
    b           Local translation vectors with shape ``(n_surface, 3)``. 
    """
    M = rotation_matrices(normals0, normals, eps=eps)

    if corner_mask is not None:
        corner_mask = jnp.asarray(corner_mask, dtype=jnp.bool_)
        identity = jnp.eye(3, dtype=Xs0.dtype)
        M = jnp.where(corner_mask[:, None, None], identity[None, :, :], M)

    b = compute_b_from_M(Xs0, Xs, M)

    return M, b


def compute_rotation_MB(Xs0, Xs, conn, faceSizes):
    """
    Compute area weights and local surface transformations. 
    Original and deformed node normals are first computed from the surface topology. 
    The normals are then used to construct local rotation matrices and translation vectors. 
    
    Parameters
    Xs0         Original surface coordinates with shape ``(n_surface, 3)``. 
    Xs          Deformed surface coordinates with shape ``(n_surface, 3)``. 
    conn        Flattened local surface-face connectivity. 
    faceSizes   Number of nodes in each surface face. 
    
    Returns 
    Ai          Node area weights computed from the original surface, with shape ``(n_surface,)``. 
    M           Local rotation matrices with shape ``(n_surface, 3, 3)``. 
    b           Local translation vectors with shape ``(n_surface, 3)``. 
    """
    normals0, normals, Ai = get_normals_Ai(
        Xs0,
        Xs,
        conn,
        faceSizes,
    )

    M, b = get_MB_rotation(
        Xs0,
        Xs,
        normals0,
        normals,
    )

    return Ai, M, b

