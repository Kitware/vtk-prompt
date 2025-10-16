"""
RAG Database Builder for VTK Examples.

This module provides functionality to build a RAG (Retrieval-Augmented Generation) database
from VTK example files. It processes Python code examples, extracts metadata, and creates
embeddings for semantic search capabilities.

The module integrates with the rag-components submodule to handle database creation,
document processing, and embedding generation for VTK code examples.

Example:
    >>> vtk-build-rag --examples-dir ./examples --collection vtk-examples
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from . import get_logger

logger = get_logger(__name__)


def setup_rag_path() -> str:
    """Add rag-components to the Python path.

    Returns:
        The path to the rag-components directory
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    rag_path = str(project_root / "rag-components")

    if rag_path not in sys.path:
        sys.path.append(rag_path)

    return rag_path


def check_dependencies() -> bool:
    """Check if required dependencies are installed.

    Returns:
        True if all dependencies are installed, False otherwise
    """
    required_modules = ["chromadb", "sentence_transformers", "tree_sitter_languages"]

    for module in required_modules:
        if importlib.util.find_spec(module) is None:
            logger.error("Missing required dependency: %s", module)
            return False

    return True


def main() -> None:
    """Build a RAG database from VTK example files."""
    parser = argparse.ArgumentParser(description="Build RAG database for VTK examples")
    parser.add_argument(
        "--examples-dir",
        type=str,
        default="data/examples",
        help="Directory containing VTK examples (default: data/examples)",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="./db/codesage-codesage-large-v2",
        help="Database path (default: ./db/codesage-codesage-large-v2)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="vtk-examples",
        help="Collection name in the database (default: vtk-examples)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="codesage/codesage-large-v2",
        help="Embedding model name (default: codesage/codesage-large-v2)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        help="Language of the examples (default: python)",
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        logger.error("Please install the required dependencies with:")
        logger.error('pip install -e ".[rag]"')
        sys.exit(1)

    # Setup RAG path
    rag_path = setup_rag_path()

    # Import populate_db from rag-components
    try:
        sys.path.insert(0, rag_path)
        from populate_db import fill_database  # type: ignore[import-not-found]
    except ImportError as e:
        logger.error("Failed to import from rag-components: %s", e)
        logger.error(
            "Make sure the rag-components directory exists and contains the required files."
        )
        sys.exit(1)

    # Check if examples directory exists
    examples_dir = Path(args.examples_dir)
    if not examples_dir.exists() or not examples_dir.is_dir():
        logger.error(
            "Examples directory '%s' does not exist or is not a directory",
            args.examples_dir,
        )
        sys.exit(1)

    # Get all Python files in the examples directory
    files = list(examples_dir.glob("**/*.py"))
    if not files:
        logger.error("No Python files found in '%s'", args.examples_dir)
        sys.exit(1)

    logger.info("Found %d Python files in '%s'", len(files), args.examples_dir)

    # Create database directory if it doesn't exist
    database_dir = Path(args.database).parent
    database_dir.mkdir(parents=True, exist_ok=True)

    # Build the RAG database
    logger.info(
        "Building RAG database at '%s' using embedding model '%s'...",
        args.database,
        args.model,
    )
    try:
        fill_database(
            files=files,
            database_path=args.database,
            embedding_model=args.model,
            language=args.language,
            collection_name=args.collection_name,
        )

        logger.info("Successfully built RAG database at '%s'", args.database)
        logger.info("You can now use the RAG database with vtk-prompt by running:")
        logger.info(
            'vtk-prompt "your query" -r --database %s --collection %s',
            args.database,
            args.collection_name,
        )

    except Exception as e:
        logger.error("Error building RAG database: %s", e)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
