"""
VTK Prompt Template System.

This module provides a template system for generating prompts used in VTK code generation.
It includes functions for loading and formatting various template types:

- Base context templates with VTK and Python version information
- RAG (Retrieval-Augmented Generation) context templates
- No-RAG context templates for direct queries
- Role-based templates for Python and XML generation
- UI-specific templates for post-processing

Templates are stored as text files in the prompts directory and can be dynamically
formatted with runtime values like VTK version, Python version, user requests,
and context snippets.
"""

from pathlib import Path

import vtk

PYTHON_VERSION = ">=3.10"
VTK_VERSION = vtk.__version__

# Path to the prompts directory
PROMPTS_DIR = Path(__file__).parent


def load_template(template_name: str) -> str:
    """Load a template file from the prompts directory.

    Args:
        template_name: Name of the template file (without .txt extension)

    Returns:
        The template content as a string
    """
    template_path = PROMPTS_DIR / f"{template_name}.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Template {template_name} not found at {template_path}")

    return template_path.read_text()


def get_base_context() -> str:
    """Get the base context template with version variables filled in."""
    template = load_template("base_context")
    return template.format(VTK_VERSION=VTK_VERSION, PYTHON_VERSION=PYTHON_VERSION)


def get_no_rag_context(request: str) -> str:
    """Get the no-RAG context template with request filled in."""
    base_context = get_base_context()
    template = load_template("no_rag_context")
    return template.format(BASE_CONTEXT=base_context, request=request)


def get_rag_context(request: str, context_snippets: str) -> str:
    """Get the RAG context template with request and snippets filled in."""
    base_context = get_base_context()
    template = load_template("rag_context")
    return template.format(
        BASE_CONTEXT=base_context, request=request, context_snippets=context_snippets
    )


def get_python_role() -> str:
    """Get the Python role template with version filled in."""
    template = load_template("python_role")
    return template.format(PYTHON_VERSION=PYTHON_VERSION)


def get_vtk_xml_context(description: str) -> str:
    """Get the VTK XML context template with description filled in."""
    template = load_template("vtk_xml_context")
    return template.format(VTK_VERSION=VTK_VERSION, description=description)


def get_xml_role() -> str:
    """Get the XML role template."""
    return load_template("xml_role")


def get_ui_post_prompt() -> str:
    """Get the UI post prompt template."""
    return load_template("ui_post_prompt")


def get_rag_chat_context(context: str, query: str) -> str:
    """Get the RAG chat context template with context and query filled in."""
    template = load_template("rag_chat_context")
    return template.format(CONTEXT=context, QUERY=query)
