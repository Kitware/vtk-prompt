"""
VTK Prompt System.

This module provides a modern YAML-based prompt system for VTK code generation.
It includes functions for loading and formatting YAML prompts:

- Standard VTK Python code generation prompts
- RAG-enhanced prompts with context from VTK examples
- VTK XML file generation prompts
- RAG chat assistant prompts
- UI-specific prompts with embedded instructions

Prompts are stored as YAML files following GitHub's prompt schema and can be dynamically
formatted with runtime values like VTK version, Python version, user requests,
and context snippets.
"""

from pathlib import Path
from typing import Any, Dict, List
import vtk

from .yaml_prompt_loader import YAMLPromptLoader
from .prompt_component_assembler import (
    PromptComponentLoader,
    VTKPromptAssembler,
    assemble_vtk_prompt,
)

PYTHON_VERSION = ">=3.10"
VTK_VERSION = vtk.__version__

# Path to the prompts directory
PROMPTS_DIR = Path(__file__).parent

# Global singleton instance
_loader = YAMLPromptLoader()


# Public API functions that delegate to the singleton
def substitute_yaml_variables(content: str, variables: Dict[str, Any]) -> str:
    """Substitute {{variable}} placeholders in YAML content."""
    return _loader.substitute_yaml_variables(content, variables)


def load_yaml_prompt(prompt_name: str, **variables: Any) -> Dict[str, Any]:
    """Load a YAML prompt file and substitute variables."""
    return _loader.load_yaml_prompt(prompt_name, **variables)


def format_messages_for_client(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Format messages from YAML prompt for LLM client."""
    return _loader.format_messages_for_client(messages)


def get_yaml_prompt(prompt_name: str, **variables: Any) -> List[Dict[str, str]]:
    """Get a YAML prompt and format it for the LLM client."""
    return _loader.get_yaml_prompt(prompt_name, **variables)


# Export classes and functions for public API
__all__ = [
    # YAML prompt functions
    "load_yaml_prompt",
    "get_yaml_prompt",
    "substitute_yaml_variables",
    "format_messages_for_client",
    # Component assembly functions
    "assemble_vtk_prompt",
    # Classes
    "YAMLPromptLoader",
    "PromptComponentLoader",
    "VTKPromptAssembler",
]
