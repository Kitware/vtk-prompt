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

# Global instance
_loader = YAMLPromptLoader()


# Public API functions that delegate to the singleton
def substitute_yaml_variables(content: str, variables: dict[str, Any]) -> str:
    """Substitute {{variable}} placeholders in YAML content."""
    return _loader.substitute_yaml_variables(content, variables)


def load_yaml_prompt(prompt_name: str, **variables: Any) -> dict[str, Any]:
    """Load a YAML prompt file and substitute variables."""
    return _loader.load_yaml_prompt(prompt_name, **variables)


def format_messages_for_client(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Format messages from YAML prompt for LLM client."""
    return _loader.format_messages_for_client(messages)


def get_yaml_prompt(prompt_name: str, **variables: Any) -> list[dict[str, str]]:
    """Get a YAML prompt and format it for the LLM client."""
    return _loader.get_yaml_prompt(prompt_name, **variables)


__all__ = [
    "load_yaml_prompt",
    "get_yaml_prompt",
    "substitute_yaml_variables",
    "format_messages_for_client",
    "assemble_vtk_prompt",
    "YAMLPromptLoader",
    "PromptComponentLoader",
    "VTKPromptAssembler",
    "PYTHON_VERSION",
    "VTK_VERSION",
    "PROMPTS_DIR",
]
