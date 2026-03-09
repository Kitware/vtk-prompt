"""
File Handlers Module.

This module provides file handling utilities for the VTK Prompt UI application,
including JavaScript loading and file operations.
"""

from pathlib import Path
from typing import Any


def load_js(server: Any) -> None:
    """Load JavaScript utilities for VTK Prompt UI."""
    js_file = Path(__file__).parent.parent.with_name("utils.js")
    server.enable_module(
        {
            "serve": {"vtk_prompt": str(js_file.parent)},
            "scripts": [f"vtk_prompt/{js_file.name}"],
        }
    )
