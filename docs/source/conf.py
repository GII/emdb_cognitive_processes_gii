# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os

sys.path.insert(0, os.path.abspath("../../cognitive_processes"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "e-MDB Cognitive processes implemented by the GII"
copyright = "2025, GII"
author = "GII"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

# The constructors document their parameters: show them with the class
autoclass_content = "both"

templates_path = ["_templates"]
exclude_patterns = []

autodoc_mock_imports = [
    "rclpy",
    "std_msgs",
    "core",
    "cognitive_nodes",
    "cognitive_processes_interfaces",
    "core_interfaces",
    "cognitive_node_interfaces",
    "numpy",
    "yaml",
    "rclpy.node",
    "rclpy.executors",
    "rclpy.callback_groups",
    "rclpy.time",
    "scipy",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
