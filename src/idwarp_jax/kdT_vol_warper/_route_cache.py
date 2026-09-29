"""Versioned, geometry-checked persistence for compact kd-tree routes."""

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import fields

import numpy as np

from ._kdtree_preprocessing import KdTreeRoutes

# Increment when traversal semantics or the stored route representation change.
_CACHE_VERSION = 1


def route_fingerprint(volume_points, wall_points, surface_area, tree, settings):
    digest = hashlib.sha256()
    digest.update(json.dumps((_CACHE_VERSION, settings)).encode())
    for value in (volume_points, wall_points, surface_area,
                  *(getattr(tree, field.name) for field in fields(tree))):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def load_routes(path, fingerprint):
    """Return matching routes, or None for a missing/stale/damaged archive."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive['fingerprint'].item() != fingerprint:
                return None
            return KdTreeRoutes(**{
                field.name: archive[field.name] for field in fields(KdTreeRoutes)
            })
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def save_routes(path, fingerprint, routes):
    """Publish atomically so interrupted writes cannot replace a good cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz',
                                         delete=False) as stream:
            temporary = stream.name
            # Uncompressed: fast loading and disk size close to routes.nbytes.
            np.savez(stream, fingerprint=np.asarray(fingerprint), **{
                field.name: getattr(routes, field.name)
                for field in fields(KdTreeRoutes)
            })
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
