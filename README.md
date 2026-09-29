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

## Tutorial

The [cube-in-farfield deformation tutorial](tutorials/basic_tutorials/cube_stretch.ipynb)
walks through mesh construction, deformation, reverse-derivative verification,
and 2D/3D visualization.

To run it locally, launch Jupyter from the repository root:

```bash
jupyter lab tutorials/basic_tutorials/cube_stretch.ipynb
```

Select the numerical precision and JAX device by changing `DTYPE` and
`DEVICE` near the top of the notebook.

```python
DTYPE = "f64"
DEVICE = "gpu"
```
