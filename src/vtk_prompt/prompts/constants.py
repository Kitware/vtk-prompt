"""
Constants for the VTK Prompt system.

This module defines version constants used throughout the prompt system.
"""

import vtk
from pathlib import Path

# Version constants used in prompt variable substitution
PYTHON_VERSION = ">=3.10"
VTK_VERSION = vtk.__version__

# Path to the prompts directory
PROMPTS_DIR = Path(__file__).parent
