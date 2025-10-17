"""
VTK Prompt Command Line Interface.

This module provides the CLI interface for VTK code generation using LLMs.
It handles argument parsing, validation, and orchestrates the VTKPromptClient.

Example:
    >>> vtk-prompt "create sphere" --rag --model gpt-5
"""

import sys
from typing import Optional

import click

from . import get_logger
from .client import VTKPromptClient
from .provider_utils import supports_temperature

logger = get_logger(__name__)


@click.command()
@click.argument("input_string")
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "gemini", "nim"]),
    default="openai",
    help="LLM provider to use",
)
@click.option("-m", "--model", default="gpt-5", help="Model name to use")
@click.option("-k", "--max-tokens", type=int, default=1000, help="Max # of tokens to generate")
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature for generation (0.0-2.0)",
)
@click.option("-t", "--token", required=True, help="API token for the selected provider")
@click.option("--base-url", help="Base URL for API (auto-detected or custom)")
@click.option("-r", "--rag", is_flag=True, help="Use RAG to improve code generation")
@click.option("-v", "--verbose", is_flag=True, help="Show generated source code")
@click.option("--collection", default="vtk-examples", help="Collection name for RAG")
@click.option(
    "--database",
    default="./db/codesage-codesage-large-v2",
    help="Database path for RAG",
)
@click.option("--top-k", type=int, default=5, help="Number of examples to retrieve from RAG")
@click.option(
    "--retry-attempts",
    type=int,
    default=1,
    help="Number of times to retry if AST validation fails",
)
@click.option(
    "--conversation",
    help="Path to conversation file for maintaining chat history",
)
def main(
    input_string: str,
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
    token: str,
    base_url: Optional[str],
    rag: bool,
    verbose: bool,
    collection: str,
    database: str,
    top_k: int,
    retry_attempts: int,
    conversation: Optional[str],
) -> None:
    """
    Generate and execute VTK code using LLMs.

    INPUT_STRING: The code description to generate VTK code for
    """
    # Set default base URLs
    if base_url is None:
        base_urls = {
            "anthropic": "https://api.anthropic.com/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "nim": "https://integrate.api.nvidia.com/v1",
        }
        base_url = base_urls.get(provider)

    # Set default models based on provider
    if model == "gpt-5":
        default_models = {
            "anthropic": "claude-opus-4-1-20250805",
            "gemini": "gemini-2.5-pro",
            "nim": "meta/llama3-70b-instruct",
        }
        model = default_models.get(provider, model)

    # Handle temperature override for unsupported models
    if not supports_temperature(model):
        logger.warning(
            "Model %s does not support temperature control. "
            "Temperature parameter will be ignored (using 1.0).",
            model,
        )
        temperature = 1.0

    try:
        client = VTKPromptClient(
            collection_name=collection,
            database_path=database,
            verbose=verbose,
            conversation_file=conversation,
        )
        result = client.query(
            input_string,
            api_key=token,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            rag=rag,
            retry_attempts=retry_attempts,
        )

        if isinstance(result, tuple) and len(result) == 3:
            explanation, generated_code, usage = result
            if verbose and usage:
                logger.info(
                    "Used tokens: input=%d output=%d",
                    usage.prompt_tokens,
                    usage.completion_tokens,
                )
            client.run_code(generated_code)
        else:
            # Handle string result
            logger.info("Result: %s", result)

    except ValueError as e:
        if "RAG components" in str(e):
            logger.error("RAG components not found")
            sys.exit(1)
        elif "Failed to load RAG snippets" in str(e):
            logger.error("Failed to load RAG snippets")
            sys.exit(2)
        elif "max_tokens" in str(e):
            logger.error("Error: %s", e)
            logger.error("Current max_tokens: %d", max_tokens)
            logger.error("Try increasing with: --max-tokens <higher_number>")
            sys.exit(3)
        else:
            logger.error("Error: %s", e)
            sys.exit(4)


if __name__ == "__main__":
    main()
