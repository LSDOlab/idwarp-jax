# IDWarp-JAX

IDWarp-JAX is a JAX implementation of
[IDWarp](https://github.com/mdolab/idwarp), MDO Lab's mesh deformation
package. It propagates prescribed boundary motion through a volume mesh using
inverse-distance-weighted interpolation and provides both the deformed volume
coordinates and a reverse-mode derivative for propagating volume
sensitivities back to the surface and flow inputs.

The original method and software are described in N. Secco, G. K. W. Kenway,
P. He, C. A. Mader, and J. R. R. A. Martins, “Efficient Mesh Generation and
Deformation for Aerodynamic Shape Optimization,” *AIAA Journal*, 2021,
[doi:10.2514/1.J059491](https://doi.org/10.2514/1.J059491).

## Workflow

1. Create a mesh dictionary containing volume points and boundary faces.
2. Call `build_volume_pts_func` once to prepare the fixed kd-tree routes.
3. Pass the target coordinates of the moving surface to the forward callable.
4. Use the VJP callable when reverse derivatives are required.

```python
compute, compute_vjp, aux = build_volume_pts_func(
    mesh,
    base_volume_points,
    device,
    return_aux=True,
)

surface_points = aux["wall_points"]
volume_points = compute(surface_points, {"pitch": 0.0})
d_surface, d_flow = compute_vjp(
    surface_points,
    {"pitch": 0.0},
    volume_seed,
)
```

The moving surface consists of faces matching `surface_face_type`. The
`aux["wall_points"]` array gives their reference coordinates in the ordering
expected by both callables.

`flow_condition["pitch"]` is a temporary feature that post-rotates the entire
deformed volume mesh, in degrees, about the global y-axis through the origin
`(0, 0, 0)`; use `0.0` for no rotation. The VJP returns its corresponding
pitch sensitivity in `d_flow["pitch"]`.

## `build_volume_pts_func` settings

The signature, defaults, and short descriptions below are generated directly
from the source docstring.

```{eval-rst}
.. autofunction:: idwarp_jax.vol_warper.build_volume_pts_func
```

```{toctree}
:maxdepth: 1
:caption: Documentation

self
Cube deformation tutorial <_tutorial/cube_stretch>
```
