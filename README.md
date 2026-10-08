# idwarp-jax

IDWarp-JAX is a JAX implementation of
[IDWarp](https://github.com/mdolab/idwarp), MDO Lab's
inverse-distance-weighted volume-mesh deformation package.

The original method and software are described in N. Secco, G. K. W. Kenway,
P. He, C. A. Mader, and J. R. R. A. Martins, “Efficient Mesh Generation and
Deformation for Aerodynamic Shape Optimization,” *AIAA Journal*, 2021,
[doi:10.2514/1.J059491](https://doi.org/10.2514/1.J059491).

IDWarp-JAX is independently developed and maintained by LSDO Lab and is not officially affiliated with or endorsed by MDO Lab.

Link to documentation: https://idwarp-jax.readthedocs.io/en/latest/

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
