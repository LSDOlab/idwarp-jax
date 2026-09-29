"""Pure-NumPy IDWarp-compatible kd-tree geometry and error estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


RSTAR = np.asarray(
    [2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0],
    dtype=np.float64,
)

_PHI = 0.5 * (1.0 + math.sqrt(5.0))
DODECAHEDRON_DIRECTIONS = np.asarray(
    [
        (-1, -1, -1),
        (1, -1, -1),
        (-1, 1, -1),
        (1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (-1, 1, 1),
        (1, 1, 1),
        (0, -1 / _PHI, -_PHI),
        (0, 1 / _PHI, -_PHI),
        (0, -1 / _PHI, _PHI),
        (0, 1 / _PHI, _PHI),
        (-1 / _PHI, -_PHI, 0),
        (1 / _PHI, -_PHI, 0),
        (-1 / _PHI, _PHI, 0),
        (1 / _PHI, _PHI, 0),
        (-_PHI, 0, -1 / _PHI),
        (-_PHI, 0, 1 / _PHI),
        (_PHI, 0, -1 / _PHI),
        (_PHI, 0, 1 / _PHI),
    ],
    dtype=np.float64,
) / math.sqrt(3.0)


@dataclass
class FlatTree:
    permutation: np.ndarray
    start: np.ndarray
    end: np.ndarray
    left: np.ndarray
    right: np.ndarray
    level: np.ndarray
    center: np.ndarray
    radius: np.ndarray
    area: np.ndarray
    error: np.ndarray


def build_flat_tree(
    points: np.ndarray,
    area: np.ndarray,
    bucket_size: int,
) -> FlatTree:
    """Build IDWarp's balanced widest-coordinate median tree as flat arrays."""
    permutation = np.arange(len(points), dtype=np.int32)
    starts = []
    ends = []
    lefts = []
    rights = []
    levels = []

    def build(start: int, end: int, level: int) -> int:
        node_id = len(starts)
        starts.append(start)
        ends.append(end)
        lefts.append(-1)
        rights.append(-1)
        levels.append(level)
        count = end - start
        # IDWarp uses inclusive Fortran bounds, hence bucket_size + 1.
        if count > bucket_size + 1:
            owned = permutation[start:end]
            owned_points = points[owned]
            coordinate = int(np.argmax(np.ptp(owned_points, axis=0)))
            left_count = (count + 1) // 2
            order = np.argpartition(
                owned_points[:, coordinate], left_count - 1
            )
            permutation[start:end] = owned[order]
            middle = start + left_count
            lefts[node_id] = build(start, middle, level + 1)
            rights[node_id] = build(middle, end, level + 1)
        return node_id

    build(0, len(points), 1)
    start_array = np.asarray(starts, dtype=np.int32)
    end_array = np.asarray(ends, dtype=np.int32)
    left_array = np.asarray(lefts, dtype=np.int32)
    right_array = np.asarray(rights, dtype=np.int32)
    level_array = np.asarray(levels, dtype=np.int16)
    permuted_points = np.ascontiguousarray(points[permutation], dtype=np.float64)
    permuted_area = np.ascontiguousarray(area[permutation], dtype=np.float64)
    center = np.empty((len(starts), 3), dtype=np.float64)
    radius = np.empty(len(starts), dtype=np.float64)
    node_area = np.empty(len(starts), dtype=np.float64)
    for node_id, (start, end) in enumerate(zip(start_array, end_array)):
        owned = permuted_points[start:end]
        center[node_id] = np.mean(owned, axis=0)
        radius[node_id] = np.sqrt(
            np.max(np.sum((owned - center[node_id]) ** 2, axis=1))
        )
        node_area[node_id] = np.sum(permuted_area[start:end])

    return FlatTree(
        permutation=permutation,
        start=start_array,
        end=end_array,
        left=left_array,
        right=right_array,
        level=level_array,
        center=center,
        radius=radius,
        area=node_area,
        error=np.zeros((len(starts), len(RSTAR)), dtype=np.float64),
    )


def _weight(ratio, area, a_exp, b_exp, alpha_to_b):
    if a_exp == 3.0 and b_exp == 5.0:
        ratio2 = ratio * ratio
        return area * (ratio2 * ratio + alpha_to_b * ratio2 * ratio2 * ratio)
    return area * (ratio**a_exp + alpha_to_b * ratio**b_exp)


def compute_error_table(
    points,
    point_area,
    start,
    end,
    left,
    center,
    radius,
    node_area,
    ldef,
    a_exp,
    b_exp,
    alpha_to_b,
    distance_eps,
    rstar,
    directions,
):
    errors = np.zeros((len(start), len(rstar)), dtype=np.float64)
    sample_count = len(rstar) * len(directions)
    max_distance_entries = 2_000_000
    for node_id in range(len(start)):
        if left[node_id] < 0 or radius[node_id] <= 0.0:
            continue
        owned_points = points[start[node_id] : end[node_id]]
        owned_area = point_area[start[node_id] : end[node_id]]
        sample_radii = radius[node_id] * rstar
        samples = (
            center[node_id][None, None, :]
            + sample_radii[:, None, None] * directions[None, :, :]
        ).reshape(sample_count, 3)
        exact = np.empty(sample_count, dtype=np.float64)
        chunk_size = max(
            1,
            min(sample_count, max_distance_entries // len(owned_points)),
        )
        for sample_start in range(0, sample_count, chunk_size):
            sample_end = min(sample_start + chunk_size, sample_count)
            difference = (
                samples[sample_start:sample_end, None, :]
                - owned_points[None, :, :]
            )
            distance = np.sqrt(
                np.einsum("sni,sni->sn", difference, difference)
                + distance_eps
            )
            weights = _weight(
                ldef / distance,
                owned_area[None, :],
                a_exp,
                b_exp,
                alpha_to_b,
            )
            exact[sample_start:sample_end] = np.sum(weights, axis=1)
        approximate = _weight(
            ldef / sample_radii,
            node_area[node_id],
            a_exp,
            b_exp,
            alpha_to_b,
        )
        difference = exact.reshape(len(rstar), len(directions)) - approximate[:, None]
        errors[node_id] = np.sqrt(np.mean(difference * difference, axis=1))
    return errors


def _leaf_source_table(start, end, left):
    leaf_nodes = np.flatnonzero(left < 0)
    width = int(np.max(end[leaf_nodes] - start[leaf_nodes]))
    source_ids = np.zeros((len(start), width), dtype=np.int32)
    valid = np.zeros((len(start), width), dtype=np.bool_)
    for node_id in leaf_nodes:
        count = int(end[node_id] - start[node_id])
        source_ids[node_id, :count] = np.arange(
            start[node_id], end[node_id], dtype=np.int32
        )
        valid[node_id, :count] = True
    return source_ids, valid


def _estimate_denominator_batch(
    volume_points,
    points,
    point_area,
    left,
    right,
    center,
    radius,
    node_area,
    leaf_source_ids,
    leaf_source_valid,
    ldef,
    a_exp,
    b_exp,
    alpha_to_b,
    distance_eps,
):
    n_volume = len(volume_points)
    denominator = np.zeros(n_volume, dtype=np.float64)
    rows = np.arange(n_volume, dtype=np.int32)
    nodes = np.zeros(n_volume, dtype=np.int32)
    while len(nodes):
        is_leaf = left[nodes] < 0
        if np.any(is_leaf):
            leaf_rows = rows[is_leaf]
            leaf_nodes = nodes[is_leaf]
            source_ids = leaf_source_ids[leaf_nodes]
            valid = leaf_source_valid[leaf_nodes]
            difference = volume_points[leaf_rows, None, :] - points[source_ids]
            distance = np.sqrt(
                np.einsum("rsi,rsi->rs", difference, difference)
                + distance_eps
            )
            weights = _weight(
                ldef / distance,
                point_area[source_ids],
                a_exp,
                b_exp,
                alpha_to_b,
            )
            values = np.sum(np.where(valid, weights, 0.0), axis=1)
            denominator += np.bincount(
                leaf_rows, weights=values, minlength=n_volume
            )

        internal_rows = rows[~is_leaf]
        internal_nodes = nodes[~is_leaf]
        if not len(internal_nodes):
            break
        difference = volume_points[internal_rows] - center[internal_nodes]
        distance = np.sqrt(
            np.einsum("ri,ri->r", difference, difference) + distance_eps
        )
        far_enough = distance / radius[internal_nodes] > 5.0
        if np.any(far_enough):
            accepted_rows = internal_rows[far_enough]
            accepted_nodes = internal_nodes[far_enough]
            values = _weight(
                ldef / distance[far_enough],
                node_area[accepted_nodes],
                a_exp,
                b_exp,
                alpha_to_b,
            )
            denominator += np.bincount(
                accepted_rows, weights=values, minlength=n_volume
            )
        expanded_rows = internal_rows[~far_enough]
        expanded_nodes = internal_nodes[~far_enough]
        rows = np.repeat(expanded_rows, 2)
        nodes = np.column_stack(
            (left[expanded_nodes], right[expanded_nodes])
        ).reshape(-1)
    return denominator


def _interpolated_errors(nodes, distance_over_radius, errors, rstar):
    result = np.empty(len(nodes), dtype=np.float64)
    far = distance_over_radius >= rstar[-1]
    result[far] = errors[nodes[far], -1]
    if np.any(~far):
        local_nodes = nodes[~far]
        local_distance = distance_over_radius[~far]
        indices = np.searchsorted(rstar[1:], local_distance, side="right")
        fraction = (
            (local_distance - rstar[indices])
            / (rstar[indices + 1] - rstar[indices])
        )
        result[~far] = (
            (1.0 - fraction) * errors[local_nodes, indices]
            + fraction * errors[local_nodes, indices + 1]
        )
    return result
