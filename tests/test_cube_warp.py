"""Small numerical checks using the tutorial's cube-in-farfield geometry."""

import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose

from idwarp_jax.vol_warper import build_volume_pts_func


_spec = importlib.util.spec_from_file_location(
    "cube_farfield_mesh",
    Path(__file__).resolve().parents[1]
    / "tutorials/basic_tutorials/cube_farfield_mesh.py",
)
_tutorial = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tutorial)

CASES = {
    "f32": (np.float32, {}),
    "f64": (np.float64, {}),
    "no-rotations": (np.float64, {"useRotations": False, "LdefFact": 1.0}),
    "corner-rotations": (np.float64, {"zeroCornerRotations": False}),
    "routing": (np.float64, {
        "route_cache": "none", "bucket_size": 4,
        "err_tol": 0.0, "max_batch_points": 32,
    }),
}


@pytest.fixture(scope="module")
def deformers():
    # Lazy caching shares initialization and compiled kernels across checks.
    cache = {}

    def get(name):
        if name not in cache:
            dtype, options = CASES[name]
            mesh, extra = _tutorial.make_cube_farfield_mesh(
                dtype=dtype,
                coordinates=(
                    [-100, -10, 0, 10, 100],
                    [0, 10, 30, 100],
                    [-100, -10, 0, 10, 100],
                ),
            )
            compute, reverse, aux = build_volume_pts_func(
                mesh, mesh["points"], jax.devices("cpu")[0],
                return_aux=True, print_timings=False, progress=False, **options,
            )
            cache[name] = mesh, extra, compute, reverse, aux
        return cache[name]

    return get


@pytest.fixture(scope="module", params=CASES)
def case(request, deformers):
    return deformers(request.param)


def stretched_surface(aux):
    surface = aux["wall_points"].copy()
    surface[:, 0] *= 2.0
    return pitch_rotate(surface, 10.0)


def pitch_rotate(points, degrees):
    theta = np.deg2rad(points.dtype.type(degrees))
    result = points.copy()
    result[:, 0] = np.cos(theta) * points[:, 0] + np.sin(theta) * points[:, 2]
    result[:, 2] = -np.sin(theta) * points[:, 0] + np.cos(theta) * points[:, 2]
    return result


def test_tiny_mesh_and_identity(case):
    mesh, extra, compute, _, aux = case
    points = mesh["points"]
    assert points.shape == (99, 3)
    assert extra["cell_points"].shape == (44, 8)
    assert len(points) - len(np.unique(mesh["faces"]["points"])) == 9
    assert extra["boundary_face_counts"] == {"cube": 12, "farfield": 64, "symmetry": 12}
    assert aux["wall_points"].shape == (17, 3)
    actual = compute(aux["wall_points"], {"pitch": 0.0})
    assert actual.dtype == points.dtype
    assert_allclose(actual, points, rtol=0, atol=2e-5 if points.dtype == np.float32 else 1e-10)


def test_stretch_surface_symmetry_and_pitch(case):
    mesh, _, compute, _, aux = case
    surface = stretched_surface(aux)
    warped = compute(surface, {"pitch": 0.0})
    assert np.isfinite(warped).all()
    assert_allclose(warped[aux["wall_ids"]], surface, rtol=0, atol=2e-5)
    assert_allclose(warped[mesh["points"][:, 1] == 0, 1], 0, rtol=0, atol=1e-12)
    interior = np.setdiff1d(np.arange(99), np.unique(mesh["faces"]["points"]))
    assert np.linalg.norm(warped[interior] - mesh["points"][interior]) > 1.0
    assert_allclose(compute(surface, {"pitch": 5.0}), pitch_rotate(warped, 5.0), rtol=2e-6, atol=2e-5)


def test_surface_and_pitch_reverse_derivatives(case):
    mesh, _, compute, reverse, aux = case
    surface = stretched_surface(aux)
    dtype = surface.dtype
    rng = np.random.default_rng(7)
    direction = rng.normal(scale=0.1, size=surface.shape).astype(dtype)
    direction[aux["wall_points"][:, 1] == 0, 1] = 0
    seed = rng.normal(size=mesh["points"].shape).astype(dtype)
    seed /= np.linalg.norm(seed)
    flow = {"pitch": dtype.type(5.0)}
    d_surface, d_flow = reverse(surface, flow, seed)
    assert d_surface.dtype == dtype
    assert np.isfinite(d_surface).all()
    assert np.isfinite(d_flow["pitch"])
    step = dtype.type(0.1 if dtype == np.float32 else 1e-4)
    rtol, atol = (3e-3, 2e-4) if dtype == np.float32 else (2e-5, 2e-8)
    fd_surface = (
        compute(surface + step * direction, flow)
        - compute(surface - step * direction, flow)
    ) / (2 * step)
    assert_allclose(np.vdot(d_surface, direction), np.vdot(seed, fd_surface), rtol=rtol, atol=atol)
    fd_pitch = (
        compute(surface, {"pitch": flow["pitch"] + step})
        - compute(surface, {"pitch": flow["pitch"] - step})
    ) / (2 * step)
    assert_allclose(d_flow["pitch"], np.vdot(seed, fd_pitch), rtol=rtol, atol=atol)


@pytest.mark.parametrize("other", ["f32", "routing"])
def test_precision_and_routing_agree(deformers, other):
    outputs = []
    for name in ("f64", other):
        mesh, _, compute, reverse, aux = deformers(name)
        surface = stretched_surface(aux)
        seed = np.random.default_rng(8).normal(size=mesh["points"].shape).astype(surface.dtype)
        flow = {"pitch": 5.0}
        ds, dp = reverse(surface, flow, seed)
        outputs.append((compute(surface, flow), ds, dp["pitch"]))
    for expected, actual in zip(*outputs):
        assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

