#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

from rag_components.query_db import query_db_interactive


def display_results(results, top_k):
    """Display the results from the RAG database query.

    Args:
        results: The results from the query
        top_k: The number of top results to display
    """
    # Display code snippets
    print(f"Top {top_k} most similar code snippets:")
    for i, (doc, metadata, score) in enumerate(
        zip(results["code_documents"], results["code_metadata"], results["code_scores"])
    ):
        print(f"\n--- Result {i + 1} (Score: {score:.4f}) ---")
        print(f"Source: {metadata['original_id']}")
        print(f"Snippet:\n{doc}")
        print("-" * 80)

    # Display text explanations
    print("\nText explanations:")
    for i, (doc, metadata, score) in enumerate(
        zip(results["text_documents"], results["text_metadata"], results["text_scores"])
    ):
        print(f"\n--- Text {i + 1} (Score: {score:.4f}) ---")
        print(f"Source: {metadata['original_id']}")
        if "code" in metadata:
            print(f"Related code: {metadata['code']}")
        print(f"Content:\n{doc}")
        print("-" * 80)


def main():
    """Test the RAG database with a query."""
    parser = argparse.ArgumentParser(
        description="Test RAG functionality for VTK examples"
    )
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

    # Check if database directory exists
    database_path = Path(args.database)
    if not database_path.parent.exists():
        print(f"Error: Database directory '{database_path.parent}' does not exist")
        print("Have you built the RAG database? Run:")
        print("vtk-build-rag")
        sys.exit(1)

    # Query the RAG database
    print(
        f"Querying RAG database at '{args.database}' with collection '{args.collection}'"
    )
    print(f"Query: '{args.query}'")
    print()

    try:
        # Query the database
        results = query_db_interactive(
            args.query, args.database, args.collection, args.top_k
        )

        # Display the results
        display_results(results, args.top_k)

    except Exception as e:
        print(f"Error querying the RAG database: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
