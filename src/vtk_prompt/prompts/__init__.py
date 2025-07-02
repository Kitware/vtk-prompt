#!/usr/bin/env python3

from pathlib import Path
import vtk
from ..yaml_prompt_loader import GitHubModelYAMLLoader

PYTHON_VERSION = ">=3.10"
VTK_VERSION = vtk.__version__

# Path to the prompts directory
PROMPTS_DIR = Path(__file__).parent

# Initialize YAML loader for current directory (src/vtk_prompt/prompts)
_yaml_loader = GitHubModelYAMLLoader(PROMPTS_DIR)


# Legacy functions for backward compatibility with rag_chat_wrapper
def get_rag_chat_context(context: str, query: str) -> str:
    """Get the RAG chat context template with context and query filled in."""
    # Use YAML version
    messages = _yaml_loader.build_messages(
        "rag_chat_context", {"CONTEXT": context, "QUERY": query}
    )
    # Return combined system + user content for backward compatibility
    system_content = ""
    user_content = ""
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] == "user":
            user_content = msg["content"]
    return f"{system_content}\n\n{user_content}"


def get_vtk_xml_context(description: str) -> str:
    """Get the VTK XML context template with description filled in."""
    # Use YAML version
    messages = _yaml_loader.build_messages(
        "vtk_xml_context", {"description": description}
    )
    # Return combined system + user content for backward compatibility
    system_content = ""
    user_content = ""
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] == "user":
            user_content = msg["content"]
    return f"{system_content}\n\n{user_content}"
