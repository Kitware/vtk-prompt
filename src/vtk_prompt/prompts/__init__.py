"""
VTK Prompt System.

This module provides a component-based prompt system for VTK code generation.

The system supports two approaches:
1. Component assembly (primary): Modular, file-based components that can be
   composed programmatically to create prompts for different scenarios
2. YAML prompt loading (for custom prompts): Direct loading of user-defined
   YAML prompt files with variable substitution

Component types include:
- Base system messages and VTK coding instructions
- UI-specific renderer instructions for web interface
- RAG context injection for retrieval-augmented generation
- Output formatting and model parameter defaults

Components are stored as YAML files and assembled at runtime with support for
conditional inclusion, variable substitution, and message composition.
"""

from pathlib import Path

from .constants import PYTHON_VERSION, VTK_VERSION
from .yaml_prompt_loader import YAMLPromptLoader
from .prompt_component_assembler import (
    PromptComponentLoader,
    VTKPromptAssembler,
    assemble_vtk_prompt,
)

# Path to the prompts directory
PROMPTS_DIR = Path(__file__).parent

# Export classes and functions for public API
__all__ = [
    # Component assembly (default system)
    "assemble_vtk_prompt",
    "PromptComponentLoader",
    "VTKPromptAssembler",
    # YAML prompt loader (for custom prompts)
    "YAMLPromptLoader",
    # Constants
    "PYTHON_VERSION",
    "VTK_VERSION",
    "PROMPTS_DIR",
]
