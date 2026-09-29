import numpy as np

from idwarp_jax import vol_warper


def _capture_make_deformer(monkeypatch):
    captured = {}

    def fake_make_deformer(mesh, **kwargs):
        captured.update(kwargs)
        wall_points = np.zeros((1, 3))
        wall_ids = np.zeros(1, dtype=np.int32)
        wall_faces = [np.array([0, 1, 2], dtype=np.int32)]

        def deform(surface_points, volume_points, pitch):
            return volume_points

        def deform_vjp(surface_points, volume_points, pitch, d_volume_points):
            return np.zeros_like(surface_points), 0.0

        return wall_points, wall_ids, wall_faces, deform, deform_vjp

    monkeypatch.setattr(vol_warper, "make_deformer", fake_make_deformer)
    return captured


def _mesh_and_points():
    points = np.zeros((3, 3))
    mesh = {
        "points": points,
        "faces": {
            "points": np.array([[0, 1, 2]], dtype=np.int32),
            "type": np.array([1], dtype=np.int32),
        },
    }
    return mesh, points


def test_build_volume_pts_func_forwards_deformer_options(monkeypatch):
    captured = _capture_make_deformer(monkeypatch)
    mesh, points = _mesh_and_points()

    vol_warper.build_volume_pts_func(
        mesh,
        points,
        device="cpu",
        LdefFact=12.5,
        symmetry_mode="exact-y",
        normal_eps=1.0e-5,
        rotation_eps=2.0e-5,
        warp_eps=3.0e-5,
        symmetry_tolerance=4.0e-5,
        useRotations=False,
        zeroCornerRotations=False,
        cornerAngle=45.0,
    )

    assert captured["LdefFact"] == 12.5
    assert captured["symmetry_mode"] == "exact-y"
    assert captured["normal_eps"] == 1.0e-5
    assert captured["rotation_eps"] == 2.0e-5
    assert captured["warp_eps"] == 3.0e-5
    assert captured["symmetry_tolerance"] == 4.0e-5
    assert captured["useRotations"] is False
    assert captured["zeroCornerRotations"] is False
    assert captured["cornerAngle"] == 45.0


def test_build_volume_pts_func_preserves_deformer_defaults(monkeypatch):
    captured = _capture_make_deformer(monkeypatch)
    mesh, points = _mesh_and_points()

    vol_warper.build_volume_pts_func(mesh, points, device="cpu")

    assert captured["LdefFact"] == 100.0
    assert captured["symmetry_mode"] == "exactsym"
    assert captured["normal_eps"] == 1.0e-6
    assert captured["rotation_eps"] == 1.0e-6
    assert captured["warp_eps"] == 1.0e-6
    assert captured["symmetry_tolerance"] == 1.0e-6
    assert captured["useRotations"] is True
    assert captured["zeroCornerRotations"] is True
    assert captured["cornerAngle"] == 30.0
