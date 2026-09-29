"""JAX-based inverse-distance-weighted volume mesh deformation."""

from .driver import deform_mesh, make_deformation_function
from .vol_warper import build_volume_pts_func, make_deformer

__all__ = [
    "build_volume_pts_func",
    "deform_mesh",
    "make_deformer",
    "make_deformation_function",
]

__version__ = "0.1.0"

