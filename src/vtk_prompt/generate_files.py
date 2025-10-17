"""
VTK XML File Generator.

This module provides functionality for generating VTK XML files using OpenAI's language models.
It includes the VTKXMLGenerator class which handles OpenAI API communication and XML file
generation based on natural language descriptions.

The module supports:
- XML file generation from text descriptions
- Template-based prompt construction for VTK XML context
- Error handling and logging for generation processes
- CLI interface for standalone XML generation

Example:
    >>> vtk-generate-xml --description "sphere" --output sphere.xml
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import click
import openai

from . import get_logger

# Import our template system
from .prompts import (
    get_vtk_xml_context,
    get_xml_role,
)

logger = get_logger(__name__)


class VTKXMLGenerator:
    """OpenAI client for VTK XML file generation."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        """Initialize the VTK XML generator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Optional custom API endpoint URL
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("No API key provided. Set OPENAI_API_KEY or pass api_key parameter.")

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_xml(
        self, message: str, model: str, max_tokens: int = 4000, temperature: float = 0.7
    ) -> str:
        """Generate VTK XML content from a description."""
        examples_path = Path("data/examples/index.json")
        if examples_path.exists():
            _ = " ".join(json.loads(examples_path.read_text()).keys())
        else:
            _ = ""

        context = get_vtk_xml_context(message)

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": get_xml_role()},
                {"role": "user", "content": context},
            ],
            max_completion_tokens=max_tokens,
            # max_tokens=max_tokens,
            temperature=temperature,
        )

        if hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content or "No content in response"
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "length":
                raise ValueError(
                    f"Output was truncated due to max_tokens limit ({max_tokens}). "
                    "Please increase max_tokens."
                )

            return content

        return "No response generated"


# Legacy function wrapper for backwards compatibility
def openai_query(
    message: str,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float = 0.7,
    base_url: Optional[str] = None,
) -> str:
    """Legacy wrapper for VTK XML generation."""
    generator = VTKXMLGenerator(api_key, base_url)
    return generator.generate_xml(message, model, max_tokens, temperature)


@click.command()
@click.argument("input_string")
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "gemini", "nim"]),
    default="openai",
    help="LLM provider to use",
)
@click.option("-m", "--model", default="gpt-5", help="Model to use for generation")
@click.option("-t", "--token", required=True, help="API token for the selected provider")
@click.option("--base-url", help="Base URL for API (auto-detected or custom)")
@click.option(
    "-k",
    "--max-tokens",
    type=int,
    default=4000,
    help="Maximum number of tokens to generate",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature for generation (0.0-2.0)",
)
@click.option("-o", "--output", help="Output file path (if not specified, output to stdout)")
def main(
    input_string: str,
    provider: str,
    model: str,
    token: str,
    base_url: Optional[str],
    max_tokens: int,
    temperature: float,
    output: Optional[str],
) -> None:
    """
    Generate VTK XML file content using LLMs.

    INPUT_STRING: Description of the VTK file to generate
    """
    # Set default base URLs
    if not base_url:
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

    # Initialize the VTK XML generator
    try:
        generator = VTKXMLGenerator(token, base_url)
    except ValueError as e:
        logger.error("Error: %s", e)
        sys.exit(1)

    # Generate the VTK XML content
    try:
        xml_content = generator.generate_xml(input_string, model, max_tokens, temperature)
    except ValueError as e:
        if "max_tokens" in str(e):
            logger.error("Error: %s", e)
            logger.error("Current max_tokens: %d", max_tokens)
            logger.error("Try increasing with: --max-tokens <higher_number>")
        else:
            logger.error("Error: %s", e)
        sys.exit(1)

    # Validate XML structure (basic check)
    if xml_content.strip().startswith("<?xml") and "</VTKFile>" in xml_content:
        # Output to file or stdout
        if output:
            with open(output, "w") as f:
                f.write(xml_content)
            logger.info("VTK XML content written to %s", output)
        else:
            print(xml_content)
    else:
        logger.warning("Generated content may not be valid VTK XML")
        if output:
            with open(output, "w") as f:
                f.write(xml_content)
            logger.info("Content written to %s (please verify)", output)
        else:
            print(xml_content)


if __name__ == "__main__":
    main()
