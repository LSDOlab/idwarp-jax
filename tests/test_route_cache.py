"""Persistence must reuse a consistent preprocessing bundle and reject changed inputs."""
from dataclasses import fields

import jax
import numpy as np
import pytest

from idwarp_jax.kdT_vol_warper import _kdtree_warper as warper
from idwarp_jax.kdT_vol_warper._route_cache import route_fingerprint, load_routes


def inputs():
    points = np.array([[0, 1, 0], [1, 1, 0], [0, 2, 0], [0, 1, 1]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    mesh = {'points': points, 'faces': {'points': faces, 'type': np.ones(4)}}
    return mesh, np.array([[2, 2, 2], [3, 1, 1]], dtype=np.float32)


def build(mesh, volume, path, **kwargs):
    return warper.make_kdtree_deformer(
        mesh, volume, jax.devices('cpu')[0], lambda x, pitch: x,
        cached_routes=path, progress=False, bucket_size=1, **kwargs,
    )


def test_hit_reuses_bundle_without_tree_or_route_recomputation(tmp_path, monkeypatch):
    mesh, volume = inputs()
    first = build(mesh, volume, tmp_path)
    surface = mesh['points'].copy()
    surface[:, 0] *= 1.1
    expected = first[3](surface, volume, np.array([0.0], dtype=np.float32))

    def forbidden(*args, **kwargs):
        raise AssertionError('cache hit must bypass preprocessing')
    monkeypatch.setattr(warper, 'build_idwarp_tree', forbidden)
    monkeypatch.setattr(warper, 'precompute_routes', forbidden)
    monkeypatch.setattr(warper, 'compute_idwarp_reference_length', forbidden)
    second = build(mesh, volume, tmp_path)
    assert second[-1]['routes_cache_hit']
    for name in ('tree', 'routes'):
        for field in fields(first[-1][name]):
            np.testing.assert_array_equal(getattr(first[-1][name], field.name),
                                          getattr(second[-1][name], field.name))
    np.testing.assert_array_equal(second[-1]['surface_area'], first[-1]['surface_area'])
    np.testing.assert_allclose(second[3](surface, volume, np.array([0.0], dtype=np.float32)), expected)


@pytest.mark.parametrize('change', ['volume', 'wall', 'connectivity', 'tolerance', 'precision'])
def test_changed_inputs_invalidate(tmp_path, change):
    mesh, volume = inputs()
    build(mesh, volume, tmp_path)
    options = {}
    if change == 'volume': volume[0, 0] += 0.1
    if change == 'wall': mesh['points'][0, 0] += 0.1
    if change == 'connectivity': mesh['faces']['points'][0] = [0, 1, 2]
    if change == 'tolerance': options['err_tol'] = 0.001
    if change == 'precision': volume = volume.astype(np.float64)
    assert not build(mesh, volume, tmp_path, **options)[-1]['routes_cache_hit']


def test_old_and_damaged_archives_are_misses(tmp_path):
    path = tmp_path / 'routes.npz'
    np.savez(path, fingerprint=np.asarray('same'))
    assert load_routes(path, 'same') is None
    path.write_bytes(b'broken archive')
    assert load_routes(path, 'same') is None


def test_fingerprint_settings_order_and_precision():
    arrays = (np.zeros((2, 3)), np.zeros((3, 3)), np.array([0, 1, 2]), np.array([3]))
    first = route_fingerprint(*arrays, {'dtype': '<f4', 'bucket': 8})
    assert first == route_fingerprint(*arrays, {'bucket': 8, 'dtype': '<f4'})
    assert first != route_fingerprint(*arrays, {'bucket': 8, 'dtype': '<f8'})


def test_effective_precision_invalidates_with_same_reference_arrays(tmp_path):
    mesh, volume = inputs()
    mesh['points'] = mesh['points'].astype(np.float64)
    volume = volume.astype(np.float64)
    with jax.enable_x64(False):
        assert not build(mesh, volume, tmp_path)[-1]['routes_cache_hit']
        assert build(mesh, volume, tmp_path)[-1]['routes_cache_hit']
    with jax.enable_x64(True):
        assert not build(mesh, volume, tmp_path)[-1]['routes_cache_hit']
