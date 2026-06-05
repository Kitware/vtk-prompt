"""
VTK Prompt Command Line Interface.

This module provides the CLI interface for VTK code generation using LLMs.
It handles argument parsing, validation, and orchestrates the VTKPromptClient.

Example:
    >>> vtk-prompt "create sphere" --mcp-url http://localhost:8000 --model claude-sonnet-4-6
"""

import sys

import click

from . import get_logger
from .client import VTKPromptClient
from .provider_utils import DEFAULT_MODEL, DEFAULT_PROVIDER, get_default_model, supports_temperature

logger = get_logger(__name__)


@click.command()
@click.argument("input_string")
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "gemini", "nim"]),
    default=DEFAULT_PROVIDER,
    help="LLM provider to use",
)
@click.option("-m", "--model", default=DEFAULT_MODEL, help="Model name to use")
@click.option("-k", "--max-tokens", type=int, default=1000, help="Max # of tokens to generate")
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature for generation (0.0-2.0)",
)
@click.option("-t", "--token", required=True, help="API token for the selected provider")
@click.option("--base-url", help="Base URL for API (auto-detected or custom)")
@click.option("-v", "--verbose", is_flag=True, help="Show generated source code")
@click.option("--mcp-url", default=None, help="vtk-mcp server URL")
@click.option("--top-k", type=int, default=5, help="Number of examples to retrieve from vtk-mcp")
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
@click.option(
    "--prompt-file",
    help="Path to custom YAML prompt file (overrides built-in prompts and defaults)",
)
def main(
    input_string: str,
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
    token: str,
    base_url: str | None,
    verbose: bool,
    mcp_url: str | None,
    top_k: int,
    retry_attempts: int,
    conversation: str | None,
    prompt_file: str | None,
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
    if model == DEFAULT_MODEL:
        model = get_default_model(provider)

    # Load custom prompt file if provided
    custom_prompt_data = None
    if prompt_file:
        try:
            from pathlib import Path

            custom_file_path = Path(prompt_file)
            if not custom_file_path.exists():
                logger.error("Custom prompt file not found: %s", prompt_file)
                sys.exit(1)

            # Load the custom YAML prompt file manually
            import yaml

            with open(custom_file_path, "r") as f:
                custom_prompt_data = yaml.safe_load(f)

            logger.info("Loaded custom prompt file: %s", custom_file_path.name)

            # Override defaults with custom prompt parameters, but preserve CLI overrides
            # Only override if CLI argument is still the default value
            if custom_prompt_data and isinstance(custom_prompt_data, dict):
                # Override model if CLI didn't specify a custom one
                if model == DEFAULT_MODEL and custom_prompt_data.get("model"):
                    model = custom_prompt_data.get("model") or model
                    logger.info("Using model from prompt file: %s", model)

                # Override model parameters if CLI used defaults
                model_params = custom_prompt_data.get("modelParameters", {})
                if temperature == 0.7 and "temperature" in model_params:
                    temperature = model_params["temperature"]
                    logger.info("Using temperature from prompt file: %s", temperature)

                if max_tokens == 1000 and "max_tokens" in model_params:
                    max_tokens = model_params["max_tokens"]
                    logger.info("Using max_tokens from prompt file: %s", max_tokens)

        except Exception as e:
            logger.error("Failed to load custom prompt file %s: %s", prompt_file, e)
            sys.exit(1)

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
            verbose=verbose,
            conversation_file=conversation,
            mcp_url=mcp_url,
        )
        result = client.query(
            input_string,
            api_key=token,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            retry_attempts=retry_attempts,
            _provider=provider,
            custom_prompt=custom_prompt_data,
        )

        # Handle result with optional validation warnings
        if isinstance(result, tuple):
            if len(result) == 4:
                # Result includes validation warnings
                explanation, generated_code, usage, validation_warnings = result
                # Display validation warnings
                for warning in validation_warnings:
                    logger.warning("Custom prompt validation: %s", warning)
            elif len(result) == 3:
                explanation, generated_code, usage = result
            else:
                logger.info("Result: %s", result)
                return

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
        if "max_tokens" in str(e):
            logger.error("Error: %s", e)
            logger.error("Current max_tokens: %d", max_tokens)
            logger.error("Try increasing with: --max-tokens <higher_number>")
            sys.exit(3)
        else:
            logger.error("Error: %s", e)
            sys.exit(4)


if __name__ == "__main__":
    main()
