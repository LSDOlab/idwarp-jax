# idwarp-jax

A JAX implementation of inverse-distance-weighted volume-mesh deformation.

## Installation

From this directory, install the package in editable mode:

```bash
python -m pip install -e .
```

The array-based API used by the existing warper code is available as:

```python
from idwarp_jax.driver import deform_mesh, make_deformation_function
from idwarp_jax.vol_warper import build_volume_pts_func, make_deformer
```

## Simple example

The half-cube example stretches the mesh by 2x in the x-direction, rotates it
10 degrees about the y-axis, checks its derivatives, and saves a PNG:

```bash
python simple_example/cube_stretch.py
```

Select the numerical precision and JAX device by changing `DTYPE` and
`DEVICE` near the top of the script.

```python
DTYPE = "f64"
DEVICE = "gpu"
```
