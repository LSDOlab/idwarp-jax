"""Surface-node normal calculations with reusable fixed topology.

The surface connectivity does not change during deformation. 
This module therefore separates one-time topology preparation 
from coordinate-dependent normal and area calculations.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


class NormalTopology(NamedTuple):
    """Fixed indexing arrays used by the surface-normal calculation."""

    face_ptr: Array
    corner_ids: Array
    face_ids: Array
    face_start: Array
    face_end: Array
    next_corner_ids: Array
    node_ids: Array
    next_node_ids: Array
    face_sizes: Array

    # Deterministic gather/reduction adjacency.
    face_corner_ids: Array
    face_corner_mask: Array
    node_corner_ids: Array
    node_corner_mask: Array
    node_endpoint_ids: Array
    node_endpoint_mask: Array

def create_face_pointer(face_sizes) -> Array:
    """Return the flattened-connectivity pointer for each face."""
    face_sizes = jnp.asarray(face_sizes, dtype=jnp.int32)
    return jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(face_sizes, dtype=jnp.int32),
        )
    )


# Backward-compatible alias used by older code.
create_face_pointer.__name__ = "create_face_pointer"

def _padded_groups(group_ids: np.ndarray, n_group: int):
    group_ids = np.asarray(group_ids, dtype=np.int32)

    counts = np.bincount(group_ids, minlength=n_group)
    width = max(1, int(counts.max(initial=0)))

    ids = np.zeros((n_group, width), dtype=np.int32)
    mask = np.zeros((n_group, width), dtype=np.bool_)
    offsets = np.zeros(n_group, dtype=np.int32)

    for item, group in enumerate(group_ids):
        slot = offsets[group]
        ids[group, slot] = item
        mask[group, slot] = True
        offsets[group] += 1

    return jnp.asarray(ids), jnp.asarray(mask)

def prepare_normal_topology(conn, face_sizes) -> NormalTopology:
    conn_np = np.asarray(conn, dtype=np.int32)
    face_sizes_np = np.asarray(face_sizes, dtype=np.int32)

    conn = jnp.asarray(conn_np, dtype=jnp.int32)
    face_sizes = jnp.asarray(face_sizes_np, dtype=jnp.int32)

    n_conn = conn_np.shape[0]
    n_face = face_sizes_np.shape[0]
    n_node = int(conn_np.max()) + 1 if n_conn else 0

    face_ptr_np = np.concatenate(
        (
            np.zeros(1, dtype=np.int32),
            np.cumsum(face_sizes_np, dtype=np.int32),
        )
    )

    corner_ids_np = np.arange(n_conn, dtype=np.int32)

    face_ids_np = np.searchsorted(
        face_ptr_np[1:],
        corner_ids_np,
        side="right",
    ).astype(np.int32)

    face_start_np = face_ptr_np[face_ids_np]
    face_end_np = face_ptr_np[face_ids_np + 1]

    next_corner_ids_np = np.where(
        corner_ids_np + 1 < face_end_np,
        corner_ids_np + 1,
        face_start_np,
    ).astype(np.int32)

    next_node_ids_np = conn_np[next_corner_ids_np]

    face_corner_ids, face_corner_mask = _padded_groups(
        face_ids_np,
        n_face,
    )

    node_corner_ids, node_corner_mask = _padded_groups(
        conn_np,
        n_node,
    )

    endpoint_node_ids = np.concatenate(
        (conn_np, next_node_ids_np)
    )

    node_endpoint_ids, node_endpoint_mask = _padded_groups(
        endpoint_node_ids,
        n_node,
    )

    return NormalTopology(
        face_ptr=jnp.asarray(face_ptr_np),
        corner_ids=jnp.asarray(corner_ids_np),
        face_ids=jnp.asarray(face_ids_np),
        face_start=jnp.asarray(face_start_np),
        face_end=jnp.asarray(face_end_np),
        next_corner_ids=jnp.asarray(next_corner_ids_np),
        node_ids=conn,
        next_node_ids=jnp.asarray(next_node_ids_np),
        face_sizes=face_sizes,
        face_corner_ids=face_corner_ids,
        face_corner_mask=face_corner_mask,
        node_corner_ids=node_corner_ids,
        node_corner_mask=node_corner_mask,
        node_endpoint_ids=node_endpoint_ids,
        node_endpoint_mask=node_endpoint_mask,
    )

def _idwarp_face_properties(pts, topology: NormalTopology, eps=1.0e-15):
    """Compute face areas/normals using IDWarp's ``getElementProps`` rule.

    This is intentionally separate from the differentiable nodal-normal kernel:
    it is used for the one-time reference-surface corner classification.
    """
    pts = jnp.asarray(pts)
    corner_points = pts[topology.node_ids]

    face_point_sum = _group_sum(
        corner_points,
        topology.face_corner_ids,
        topology.face_corner_mask,
    )
    face_sizes = topology.face_sizes.astype(pts.dtype)
    centers = face_point_sum / face_sizes[:, None]

    radial = corner_points - centers[topology.face_ids]
    radial_next = pts[topology.next_node_ids] - centers[topology.face_ids]
    cross = jnp.cross(radial, radial_next)
    cross_norm = jnp.sqrt(jnp.sum(cross * cross, axis=1) + eps)

    unit_cross = cross / cross_norm[:, None]
    raw_face_normal = _group_sum(
        unit_cross,
        topology.face_corner_ids,
        topology.face_corner_mask,
    )
    face_normal_norm = jnp.sqrt(
        jnp.sum(raw_face_normal * raw_face_normal, axis=1) + eps
    )
    face_normals = raw_face_normal / face_normal_norm[:, None]

    cross_norm_sum = _group_sum(
        cross_norm,
        topology.face_corner_ids,
        topology.face_corner_mask,
    )
    face_areas = 0.5 * cross_norm_sum
    return face_areas, face_normals


def detect_corner_nodes_from_topology(
    pts,
    topology: NormalTopology,
    corner_angle: float = 30.0,
    zero_corner_rotations: bool = True,
    eps: float = 1.0e-15,
):
    """Return the fixed reference-surface corner mask used by IDWarp.

    The implementation mirrors IDWarp's ``determineCorners`` logic: build
    reference face normals with ``getElementProps``; build each reference nodal
    normal from face-area-per-corner weights; then flag a node if any attached
    face satisfies ``abs(face_normal dot node_normal) < cos(corner_angle)``.

    The mask is computed from the undeformed surface and should be reused for
    all subsequent deformations.
    """
    pts = jnp.asarray(pts)
    if not zero_corner_rotations:
        return jnp.zeros((pts.shape[0],), dtype=jnp.bool_)

    face_areas, face_normals = _idwarp_face_properties(pts, topology, eps=eps)
    face_sizes = topology.face_sizes.astype(pts.dtype)
    area_per_corner_by_face = face_areas / face_sizes
    corner_area = area_per_corner_by_face[topology.face_ids]
    corner_normal_contribution = corner_area[:, None] * face_normals[topology.face_ids]

    node_normal_sum = _group_sum(
        corner_normal_contribution,
        topology.node_corner_ids,
        topology.node_corner_mask,
    )
    node_area_sum = _group_sum(
        corner_area,
        topology.node_corner_ids,
        topology.node_corner_mask,
    )
    node_normals = node_normal_sum / jnp.maximum(node_area_sum[:, None], eps)

    node_face_ids = topology.face_ids[topology.node_corner_ids]
    attached_face_normals = face_normals[node_face_ids]
    dots = jnp.sum(attached_face_normals * node_normals[:, None, :], axis=2)

    threshold = jnp.cos(
        jnp.asarray(corner_angle, dtype=pts.dtype) * jnp.pi / 180.0
    )
    corner_local = jnp.any(
        topology.node_corner_mask & (jnp.abs(dots) < threshold),
        axis=1,
    )
    return _pad_nodes(corner_local, pts.shape[0]).astype(jnp.bool_)


def _group_sum(values, ids, mask):
    gathered = values[ids]

    if values.ndim == 2:
        mask = mask[..., None]

    return jnp.sum(
        jnp.where(mask, gathered, 0),
        axis=1,
    )


def _pad_nodes(values, n_surface):
    n_missing = n_surface - values.shape[0]

    if values.ndim == 2:
        return jnp.pad(
            values,
            ((0, n_missing), (0, 0)),
        )

    return jnp.pad(
        values,
        ((0, n_missing),),
    )


def _compute_node_normals_impl(pts, topology, eps):
    pts = jnp.asarray(pts)
    n_surface = pts.shape[0]

    x0 = pts[topology.node_ids]
    x1 = pts[topology.next_node_ids]

    edge_cross = jnp.cross(x0, x1)

    # Deterministic face reduction. Replaces segment_sum.
    raw_face_normal = _group_sum(
        edge_cross,
        topology.face_corner_ids,
        topology.face_corner_mask,
    )

    raw_norm = jnp.sqrt(
        jnp.sum(
            raw_face_normal * raw_face_normal,
            axis=1,
        )
        + eps
    )

    face_area = 0.5 * raw_norm
    face_normal = raw_face_normal / raw_norm[:, None]

    corners_per_face = topology.face_sizes.astype(pts.dtype)
    valid_face = topology.face_sizes >= 3

    area_per_corner = jnp.where(
        valid_face,
        face_area / corners_per_face,
        0.0,
    )

    corner_area = area_per_corner[topology.face_ids]
    corner_face_normal = face_normal[topology.face_ids]

    corner_normal_contribution = (
        corner_area[:, None] * corner_face_normal
    )

    # Deterministic node reductions. Replaces .at[].add().
    normal_sum = _group_sum(
        corner_normal_contribution,
        topology.node_corner_ids,
        topology.node_corner_mask,
    )

    area_sum = _group_sum(
        corner_area,
        topology.node_corner_ids,
        topology.node_corner_mask,
    )

    normal_sum = _pad_nodes(normal_sum, n_surface)
    area_sum = _pad_nodes(area_sum, n_surface)

    normals_pre = normal_sum / (
        area_sum[:, None] + eps
    )

    normal_norm = jnp.sqrt(
        jnp.sum(
            normals_pre * normals_pre,
            axis=1,
            keepdims=True,
        )
        + eps
    )

    normals = normals_pre / normal_norm

    residual = (
        pts,
        raw_face_normal,
        raw_norm,
        face_normal,
        corner_area,
        normal_sum,
        area_sum,
        normals_pre,
        normal_norm,
    )

    return (normals, area_sum), residual


@jax.custom_vjp
def _compute_node_normals_deterministic(
    pts,
    topology,
    eps,
):
    out, _ = _compute_node_normals_impl(
        pts,
        topology,
        eps,
    )
    return out


def _compute_node_normals_fwd(
    pts,
    topology,
    eps,
):
    out, residual = _compute_node_normals_impl(
        pts,
        topology,
        eps,
    )

    return out, (topology, eps, residual)


def _compute_node_normals_bwd(
    saved,
    cotangents,
):
    topology, eps, residual = saved

    (
        pts,
        raw_face_normal,
        raw_norm,
        face_normal,
        corner_area,
        normal_sum,
        area_sum,
        normals_pre,
        normal_norm,
    ) = residual

    d_normals, d_area_sum_out = cotangents

    # normals = normals_pre / ||normals_pre||
    dot = jnp.sum(
        d_normals * normals_pre,
        axis=1,
        keepdims=True,
    )

    d_normals_pre = (
        d_normals / normal_norm
        - normals_pre
        * dot
        / (
            normal_norm
            * normal_norm
            * normal_norm
        )
    )

    # normals_pre = normal_sum / (area_sum + eps)
    denom = area_sum + eps

    d_normal_sum = (
        d_normals_pre / denom[:, None]
    )

    d_area_sum = (
        d_area_sum_out
        - jnp.sum(
            d_normals_pre * normals_pre,
            axis=1,
        )
        / denom
    )

    n_node = topology.node_corner_ids.shape[0]

    d_normal_sum = d_normal_sum[:n_node]
    d_area_sum = d_area_sum[:n_node]

    # Reverse of deterministic node reduction.
    d_corner_normal = (
        d_normal_sum[topology.node_ids]
    )

    d_corner_area = (
        d_area_sum[topology.node_ids]
    )

    corner_face_normal = (
        face_normal[topology.face_ids]
    )

    d_corner_area = (
        d_corner_area
        + jnp.sum(
            d_corner_normal * corner_face_normal,
            axis=1,
        )
    )

    d_corner_face_normal = (
        d_corner_normal
        * corner_area[:, None]
    )

    # Reverse accumulation to faces, again deterministic.
    d_face_normal = _group_sum(
        d_corner_face_normal,
        topology.face_corner_ids,
        topology.face_corner_mask,
    )

    d_area_per_corner = _group_sum(
        d_corner_area,
        topology.face_corner_ids,
        topology.face_corner_mask,
    )

    corners_per_face = (
        topology.face_sizes.astype(pts.dtype)
    )

    valid_face = topology.face_sizes >= 3

    d_face_area = jnp.where(
        valid_face,
        d_area_per_corner / corners_per_face,
        0.0,
    )

    # face_normal = raw_face_normal / raw_norm
    dot_face = jnp.sum(
        d_face_normal * raw_face_normal,
        axis=1,
    )

    d_raw_face_normal = (
        d_face_normal / raw_norm[:, None]
        - raw_face_normal
        * (
            dot_face
            / (
                raw_norm
                * raw_norm
                * raw_norm
            )
        )[:, None]
    )

    # face_area = 0.5 * raw_norm
    # raw_norm = sqrt(raw_face_normal**2 + eps)
    d_raw_norm = 0.5 * d_face_area

    d_raw_face_normal = (
        d_raw_face_normal
        + (
            d_raw_norm / raw_norm
        )[:, None]
        * raw_face_normal
    )

    # Reverse of face edge reduction.
    d_edge_cross = (
        d_raw_face_normal[topology.face_ids]
    )

    x0 = pts[topology.node_ids]
    x1 = pts[topology.next_node_ids]

    # edge_cross = cross(x0, x1)
    dx0 = jnp.cross(
        x1,
        d_edge_cross,
    )

    dx1 = jnp.cross(
        d_edge_cross,
        x0,
    )

    # Deterministic accumulation of both edge endpoints
    # back to surface vertices.
    endpoint_grads = jnp.concatenate(
        (dx0, dx1),
        axis=0,
    )

    d_pts = _group_sum(
        endpoint_grads,
        topology.node_endpoint_ids,
        topology.node_endpoint_mask,
    )

    d_pts = _pad_nodes(
        d_pts,
        pts.shape[0],
    )

    return d_pts, None, None


_compute_node_normals_deterministic.defvjp(
    _compute_node_normals_fwd,
    _compute_node_normals_bwd,
)


@jax.jit
def compute_node_normals_from_topology(
    pts,
    topology: NormalTopology,
    eps: float = 1.0e-30,
):
    return _compute_node_normals_deterministic(
        pts,
        topology,
        eps,
    )

def compute_node_normals(
    pts,
    conn,
    faceSizes,
    eps: float = 1.0e-30,
):
    """Backward-compatible self-contained normal calculation.

    Repeated deformation calls should instead prepare topology once with
    :func:`prepare_normal_topology` and call
    :func:`compute_node_normals_from_topology`.
    """
    topology = prepare_normal_topology(conn, faceSizes)
    return compute_node_normals_from_topology(pts, topology, eps=eps)


def get_normals_Ai(
    pts0,
    pts,
    conn,
    faceSizes,
    eps: float = 1.0e-30,
):
    """Compute original/deformed normals and original area weights."""
    topology = prepare_normal_topology(conn, faceSizes)
    normals0, Ai = compute_node_normals_from_topology(
        pts0,
        topology,
        eps=eps,
    )
    normals, _ = compute_node_normals_from_topology(
        pts,
        topology,
        eps=eps,
    )
    return normals0, normals, Ai
