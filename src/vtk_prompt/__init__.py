"""VTK-Prompt - CLI tool for generating VTK visualizations using LLMs.

This package provides tools to generate VTK Python code and XML files using
LLMs (Anthropic Claude, OpenAI GPT, or NVIDIA NIM models). It also includes
Retrieval-Augmented Generation (RAG) capabilities to improve code generation
by providing relevant examples from the VTK examples corpus.

Main components:
- vtk-prompt: Generate and run VTK Python code
- gen-vtk-file: Generate VTK XML files
- vtk-build-rag: Build a RAG database from VTK examples
- vtk-test-rag: Test the RAG database with queries
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("vtk-prompt")
except PackageNotFoundError:
    __version__ = "unknown"
__author__ = "Vicente Adolfo Bolea Sanchez"
__email__ = "vicente.bolea@kitware.com"

from pathlib import Path


def get_project_root():
    """Get the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "db"
RAG_COMPONENTS_DIR = PROJECT_ROOT / "rag-components"
