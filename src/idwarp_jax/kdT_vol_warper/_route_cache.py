"""Versioned, geometry-checked persistence for compact kd-tree routes."""

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import fields

import numpy as np

from ._kdtree_preprocessing import KdTreeRoutes
from ._kdtree_tree import FlatTree

# Increment when traversal semantics or the stored route representation change.
_CACHE_VERSION = 2


def route_fingerprint(volume_points, wall_points, conn, face_sizes, settings):
    """Hash original reference inputs, never computed floating-point results."""
    digest = hashlib.sha256()
    digest.update(json.dumps((_CACHE_VERSION, settings), sort_keys=True).encode())
    for value in (volume_points, wall_points, conn, face_sizes):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def load_routes(path, fingerprint):
    """Return a consistent tree/routes/area/length bundle, or None on a miss."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive['version'].item() != _CACHE_VERSION or archive['fingerprint'].item() != fingerprint:
                return None
            routes = KdTreeRoutes(**{
                field.name: archive[field.name] for field in fields(KdTreeRoutes)
            })
            tree = FlatTree(**{
                field.name: archive["tree_" + field.name] for field in fields(FlatTree)
            })
            return tree, routes, archive["surface_area"], float(archive["ldef"].item())
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def save_routes(path, fingerprint, routes, tree, surface_area, ldef, settings):
    """Publish atomically so interrupted writes cannot replace a good cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz',
                                         delete=False) as stream:
            temporary = stream.name
            # Uncompressed: fast loading and disk size close to routes.nbytes.
            np.savez(stream, version=np.asarray(_CACHE_VERSION),
                     fingerprint=np.asarray(fingerprint),
                     settings=np.asarray(json.dumps(settings, sort_keys=True)),
                     surface_area=surface_area, ldef=np.asarray(ldef),
                     **{"tree_" + field.name: getattr(tree, field.name)
                        for field in fields(FlatTree)}, **{
                field.name: getattr(routes, field.name)
                for field in fields(KdTreeRoutes)
            })
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
