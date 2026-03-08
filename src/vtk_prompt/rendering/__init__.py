"""
VTK Rendering Package.

This package provides VTK rendering utilities and isolates VTK-specific logic from the UI
layer for better modularity.
"""

from .code_executor import execute_vtk_code
from .scene_manager import add_default_scene, clear_scene, reset_camera, setup_vtk_renderer

__all__ = [
    "setup_vtk_renderer",
    "add_default_scene",
    "clear_scene",
    "reset_camera",
    "execute_vtk_code",
]
