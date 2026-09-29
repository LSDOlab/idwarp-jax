"""Sphinx configuration for the two-page IDWarp-JAX documentation."""

from pathlib import Path
import shutil
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "idwarp_jax"
copyright = "2026, Mark"
author = "Mark"
version = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx_copybutton",
    "myst_nb",
]

root_doc = "index"
exclude_patterns = ["README.md", "_build", "Thumbs.db", ".DS_Store"]

myst_title_to_header = True
myst_enable_extensions = ["dollarmath", "amsmath"]
nb_execution_mode = "off"

# Keep the reference page focused on the public function and its docstring.
autodoc_typehints = "description"
numpydoc_show_class_members = False

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 2,
    "prev_next_buttons_location": "bottom",
    "sticky_navigation": True,
    "titles_only": True,
}

_DOCS_DIR = Path(__file__).resolve().parent
_TUTORIAL_SOURCE = _DOCS_DIR.parent / "tutorials" / "basic_tutorials"
_TUTORIAL_BUILD_SOURCE = _DOCS_DIR / "_tutorial"


def _copy_tutorial(app):
    """Expose the runnable notebook to Sphinx without duplicating it in git."""
    shutil.rmtree(_TUTORIAL_BUILD_SOURCE, ignore_errors=True)
    shutil.copytree(
        _TUTORIAL_SOURCE,
        _TUTORIAL_BUILD_SOURCE,
        ignore=shutil.ignore_patterns("__pycache__"),
    )


def _clean_tutorial(app, exception):
    shutil.rmtree(_TUTORIAL_BUILD_SOURCE, ignore_errors=True)


def setup(app):
    app.connect("builder-inited", _copy_tutorial)
    app.connect("build-finished", _clean_tutorial)
