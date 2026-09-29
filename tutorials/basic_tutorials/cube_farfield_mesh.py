"""Uniform half-domain mesh with a cube-shaped solid removed."""

from __future__ import annotations

import numpy as np


def make_cube_farfield_mesh(
    nx=20,
    ny=20,
    nz=20,
    cube_half=10.0,
    farfield=100.0,
    dtype=np.float32,
):
    """Return the warper mesh and plotting metadata for the cube example."""
    x = np.linspace(-farfield, farfield, nx + 1, dtype=dtype)
    y = np.linspace(0.0, farfield, ny + 1, dtype=dtype)
    z = np.linspace(-farfield, farfield, nz + 1, dtype=dtype)
    for name, coordinates in (("x", x), ("y", y), ("z", z)):
        if not np.any(np.isclose(coordinates, cube_half)):
            raise ValueError(
                f"the {name} grid does not align with the cube surface"
            )
    if not np.any(np.isclose(x, -cube_half)) or not np.any(
        np.isclose(z, -cube_half)
    ):
        raise ValueError("the x/z grid does not align with both cube sides")

    xyz = np.meshgrid(x, y, z, indexing="ij")
    background_points = np.stack(xyz, axis=-1).reshape(-1, 3)
    node = np.arange(len(background_points)).reshape(
        nx + 1, ny + 1, nz + 1
    )

    fluid_cells_old = []
    removed_cell_count = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                inside_cube = (
                    abs(0.5 * (x[i] + x[i + 1])) < cube_half
                    and 0.5 * (y[j] + y[j + 1]) < cube_half
                    and abs(0.5 * (z[k] + z[k + 1])) < cube_half
                )
                if inside_cube:
                    removed_cell_count += 1
                    continue

                n000, n100 = node[i, j, k], node[i + 1, j, k]
                n010, n110 = node[i, j + 1, k], node[i + 1, j + 1, k]
                n001, n101 = node[i, j, k + 1], node[i + 1, j, k + 1]
                n011, n111 = (
                    node[i, j + 1, k + 1],
                    node[i + 1, j + 1, k + 1],
                )
                fluid_cells_old.append(
                    [n000, n100, n110, n010, n001, n101, n111, n011]
                )

    fluid_cells_old = np.asarray(fluid_cells_old, dtype=np.int32)
    used = np.unique(fluid_cells_old)
    old_to_new = np.full(len(background_points), -1, dtype=np.int32)
    old_to_new[used] = np.arange(len(used), dtype=np.int32)
    points = background_points[used]
    cells = old_to_new[fluid_cells_old]

    # Faces appearing once lie on the fluid-domain boundary.
    boundary = {}
    for cell in fluid_cells_old:
        n000, n100, n110, n010, n001, n101, n111, n011 = cell
        cell_faces = (
            [n000, n001, n011, n010],
            [n100, n110, n111, n101],
            [n000, n100, n101, n001],
            [n010, n011, n111, n110],
            [n000, n010, n110, n100],
            [n001, n101, n111, n011],
        )
        for face in cell_faces:
            key = tuple(sorted(face))
            boundary[key] = None if key in boundary else face

    faces = []
    face_types = []
    cube_face_connectivity = []
    boundary_face_counts = {"cube": 0, "farfield": 0, "symmetry": 0}
    atol = 20.0 * np.finfo(dtype).eps

    for face in boundary.values():
        if face is None:
            continue
        face = old_to_new[np.asarray(face, dtype=np.int32)]
        coordinates = points[face]
        center = coordinates.mean(axis=0)
        on_symmetry = np.all(
            np.isclose(coordinates[:, 1], 0.0, atol=atol)
        )
        on_farfield = (
            np.all(
                np.isclose(np.abs(coordinates[:, 0]), farfield, atol=atol)
            )
            or np.all(
                np.isclose(coordinates[:, 1], farfield, atol=atol)
            )
            or np.all(
                np.isclose(np.abs(coordinates[:, 2]), farfield, atol=atol)
            )
        )
        on_cube = (
            np.all(
                np.isclose(np.abs(coordinates[:, 0]), cube_half, atol=atol)
            )
            and center[1] <= cube_half + atol
            and abs(center[2]) <= cube_half + atol
        ) or (
            np.all(np.isclose(coordinates[:, 1], cube_half, atol=atol))
            and abs(center[0]) <= cube_half + atol
            and abs(center[2]) <= cube_half + atol
        ) or (
            np.all(
                np.isclose(np.abs(coordinates[:, 2]), cube_half, atol=atol)
            )
            and center[1] <= cube_half + atol
            and abs(center[0]) <= cube_half + atol
        )

        if on_cube:
            face_type = 1
            cube_face_connectivity.append(face)
            boundary_face_counts["cube"] += 1
        elif on_farfield:
            face_type = 3
            boundary_face_counts["farfield"] += 1
        elif on_symmetry:
            face_type = 2
            boundary_face_counts["symmetry"] += 1
        else:
            raise RuntimeError(f"unclassified boundary face at {center}")

        faces.append(face)
        face_types.append(face_type)

    mesh = {
        "points": points,
        "faces": {
            "points": np.asarray(faces, dtype=np.int32),
            "type": np.asarray(face_types, dtype=np.int32),
        },
    }
    extra = {
        "cell_points": cells,
        "cube_faces": np.asarray(cube_face_connectivity, dtype=np.int32),
        "cube_half": cube_half,
        "farfield": farfield,
        "background_cell_count": nx * ny * nz,
        "removed_cell_count": removed_cell_count,
        "boundary_face_counts": boundary_face_counts,
    }
    return mesh, extra
