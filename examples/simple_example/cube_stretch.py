"""Stretch a structured half-cube, rotate it, and check its derivatives."""

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from idwarp_jax.vol_warper import build_volume_pts_func

# Change these two settings to select precision and hardware.
DTYPE = "f32"  # "f32" or "f64"
DEVICE = "cpu"  # "cpu" or "gpu"


def make_half_cube(nx=20, ny=14, nz=20, dtype=np.float32):
    """Return 5,600 structured cells plus their wall/symmetry faces."""
    x = np.linspace(-1.0, 1.0, nx + 1, dtype=dtype)
    y = np.linspace(0.0, 1.0, ny + 1, dtype=dtype)
    z = np.linspace(-1.0, 1.0, nz + 1, dtype=dtype)
    xyz = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack(xyz, axis=-1).reshape(-1, 3)
    node = np.arange(len(points)).reshape(nx + 1, ny + 1, nz + 1)
    faces, face_types = [], []

    def add(ids, face_type=1):
        faces.append(ids)
        face_types.append(face_type)

    for j in range(ny):
        for k in range(nz):
            add([node[0, j, k], node[0, j, k + 1],
                 node[0, j + 1, k + 1], node[0, j + 1, k]])
            add([node[nx, j, k], node[nx, j + 1, k],
                 node[nx, j + 1, k + 1], node[nx, j, k + 1]])
    for i in range(nx):
        for k in range(nz):
            add([node[i, ny, k], node[i, ny, k + 1],
                 node[i + 1, ny, k + 1], node[i + 1, ny, k]])
            add([node[i, 0, k], node[i + 1, 0, k],
                 node[i + 1, 0, k + 1], node[i, 0, k + 1]], 2)
    for i in range(nx):
        for j in range(ny):
            add([node[i, j, 0], node[i, j + 1, 0],
                 node[i + 1, j + 1, 0], node[i + 1, j, 0]])
            add([node[i, j, nz], node[i + 1, j, nz],
                 node[i + 1, j + 1, nz], node[i, j + 1, nz]])

    return {
        "points": points,
        "faces": {
            "points": np.asarray(faces, dtype=np.int32),
            "type": np.asarray(face_types, dtype=np.int32),
        },
    }


def plot_state(axis, points, wall_ids, title, color):
    axis.scatter(*points[::4].T, s=1, c="0.75", alpha=0.25)
    axis.scatter(*points[wall_ids][::2].T, s=2, c=color, alpha=0.7)
    axis.set(title=title, xlabel="x", ylabel="y", zlabel="z")
    axis.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1.0e-12))
    axis.view_init(elev=20, azim=-55)


def main():
    if DTYPE not in {"f32", "f64"} or DEVICE not in {"cpu", "gpu"}:
        raise ValueError("DTYPE must be f32/f64 and DEVICE must be cpu/gpu")
    jax.config.update("jax_enable_x64", DTYPE == "f64")
    dtype = np.float64 if DTYPE == "f64" else np.float32
    try:
        device = jax.devices(DEVICE)[0]
    except RuntimeError as error:
        raise RuntimeError(f"No JAX {DEVICE} device is available") from error

    mesh = make_half_cube(dtype=dtype)
    compute, compute_vjp, aux = build_volume_pts_func(
        mesh, mesh["points"], device, return_aux=True,
        print_timings=True, progress=False,
    )
    surface = aux["wall_points"].copy()
    surface[:, 0] *= 2.0
    flow = {"pitch": np.asarray(10.0, dtype=dtype)}
    deformed = compute(surface, flow)

    rng = np.random.default_rng(7)
    direction = rng.normal(scale=0.1, size=surface.shape).astype(dtype)
    seed = rng.normal(size=deformed.shape).astype(dtype)
    seed /= np.linalg.norm(seed)
    d_pitch = dtype(1.0)
    step = dtype(2.0e-3 if DTYPE == "f32" else 1.0e-5)
    plus = compute(surface + step * direction,
                   {"pitch": flow["pitch"] + step * d_pitch})
    minus = compute(surface - step * direction,
                    {"pitch": flow["pitch"] - step * d_pitch})
    finite_difference = (plus - minus) / (2.0 * step)
    d_surface, d_flow = compute_vjp(surface, flow, seed)
    forward = np.vdot(seed, finite_difference)
    reverse = np.vdot(d_surface, direction) + d_flow["pitch"] * d_pitch
    relative_error = abs(forward - reverse) / max(abs(forward), abs(reverse))

    print(f"device={device.platform}, dtype={deformed.dtype}")
    print(f"cells=5,600, points={len(mesh['points']):,}")
    print(f"derivative relative error={relative_error:.3e}")

    figure = plt.figure(figsize=(10, 4))
    before = figure.add_subplot(121, projection="3d")
    after = figure.add_subplot(122, projection="3d")
    plot_state(before, mesh["points"], aux["wall_ids"], "Before", "C0")
    plot_state(after, deformed, aux["wall_ids"], "2x stretch + 10° y rotation", "C1")
    output = Path(__file__).with_name("cube_stretch.png")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
