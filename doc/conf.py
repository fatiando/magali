# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import datetime

import magali

# Project information
# -----------------------------------------------------------------------------
project = "magali"
copyright = f"{datetime.date.today().year}, The {project} Developers"
is_dev_version = len(magali.__version__.split(".")) > 3
version = "dev" if is_dev_version else magali.__version__

# General configuration
# -----------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "jupyter_sphinx",
]

# Configuration to include links to other project docs when referencing
# functions/classes
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "ensaio": ("https://www.fatiando.org/ensaio/latest/", None),
}

# Autosummary pages will be generated by sphinx-autogen instead of sphinx-build
autosummary_generate = []

# Create cross-references for the parameter types in the Parameters, Other
# Returns and Yields sections of the docstring
numpydoc_xref_param_type = True
# Format the Attributes like the Parameters section.
numpydoc_attributes_as_param_list = True
# Disable the creation of a toctree for class members to avoid missing stub
# file warnings. See https://stackoverflow.com/a/73294408
numpydoc_class_members_toctree = False

# Sphinx project configuration
templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
source_suffix = ".rst"
# The encoding of source files
source_encoding = "utf-8"
master_doc = "index"
pygments_style = "default"
add_function_parentheses = False

# HTML output configuration
# -----------------------------------------------------------------------------
html_title = f'{project} <span class="project-version">{version}</span>'
html_logo = ""
html_favicon = "_static/favicon.png"
html_last_updated_fmt = "%b %d, %Y"
html_copy_source = True
html_static_path = ["_static"]
# CSS files are relative to the static path
html_css_files = ["custom.css"]
html_extra_path = []
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/fatiando/magali",
    "repository_branch": "main",
    "path_to_docs": "doc",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "home_page_in_toc": False,
}
