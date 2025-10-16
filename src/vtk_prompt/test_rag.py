#!/usr/bin/env python3
"""
RAG Database Testing Utilities.

This module provides utilities for testing the RAG (Retrieval-Augmented Generation) database
functionality. It includes functions to set up the RAG components path, query the database,
and test various RAG operations.

The module is designed to work with the rag-components submodule and provides a CLI interface
for testing database queries and validating RAG functionality.

Example:
    >>> vtk-test-rag --query "lighting effects" --top-k 10
"""

import argparse
import sys
from pathlib import Path
from typing import Any

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


def display_results(results: dict[str, Any], top_k: int) -> None:
    """Display the results from the RAG database query.

    Args:
        results: The results from the query
        top_k: The number of top results to display
    """
    # Display code snippets
    logger.info("Top %d most similar code snippets:", top_k)
    for i, (doc, metadata, score) in enumerate(
        zip(results["code_documents"], results["code_metadata"], results["code_scores"])
    ):
        logger.info("\n--- Result %d (Score: %.4f) ---", i + 1, score)
        logger.info("Source: %s", metadata["original_id"])
        logger.info("Snippet:\n%s", doc)
        logger.info("-" * 80)

    # Display text explanations
    logger.info("\nText explanations:")
    for i, (doc, metadata, score) in enumerate(
        zip(results["text_documents"], results["text_metadata"], results["text_scores"])
    ):
        logger.info("\n--- Text %d (Score: %.4f) ---", i + 1, score)
        logger.info("Source: %s", metadata["original_id"])
        if "code" in metadata:
            logger.info("Related code: %s", metadata["code"])
        logger.info("Content:\n%s", doc)
        logger.info("-" * 80)


def main() -> None:
    """Test the RAG database with a query."""
    parser = argparse.ArgumentParser(description="Test RAG functionality for VTK examples")
    parser.add_argument("query", type=str, help="Query to test the RAG database with")
    parser.add_argument(
        "--database",
        type=str,
        default="./db/codesage-codesage-large-v2",
        help="Database path (default: ./db/codesage-codesage-large-v2)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="vtk-examples",
        help="Collection name in the database (default: vtk-examples)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)",
    )

    args = parser.parse_args()

    # Setup RAG path
    rag_path = setup_rag_path()

    # Import query_db from rag-components
    try:
        sys.path.insert(0, rag_path)
        from query_db import query_db_interactive
    except ImportError as e:
        logger.error("Failed to import from rag-components: %s", e)
        logger.error("Make sure you have installed the required dependencies:")
        logger.error('pip install -e ".[rag]"')
        sys.exit(1)

    # Check if database directory exists
    database_path = Path(args.database)
    if not database_path.parent.exists():
        logger.error("Database directory '%s' does not exist", database_path.parent)
        logger.error("Have you built the RAG database? Run:")
        logger.error("vtk-build-rag")
        sys.exit(1)

    # Query the RAG database
    logger.info(
        "Querying RAG database at '%s' with collection '%s'",
        args.database,
        args.collection,
    )
    logger.info("Query: '%s'", args.query)

    try:
        # Query the database
        results = query_db_interactive(args.query, args.database, args.collection, args.top_k)

        # Display the results
        display_results(results, args.top_k)

    except Exception as e:
        logger.error("Error querying the RAG database: %s", e)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
