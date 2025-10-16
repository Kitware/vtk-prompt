#!/usr/bin/env python3

from pathlib import Path
import os
import json
import sys
import openai
import click

# Import our template system
from .prompts import (
    get_vtk_xml_context,
    get_xml_role,
)

from . import DATA_DIR


class VTKXMLGenerator:
    """OpenAI client for VTK XML file generation."""

    def __init__(self, api_key=None, base_url=None):
        """Initialize the VTK XML generator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Optional custom API endpoint URL
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY or pass api_key parameter."
            )

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_xml(self, message, model, max_tokens=4000, temperature=0.7):
        """Generate VTK XML content from a description."""
        examples_path = DATA_DIR / "examples" / "index.json"
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
            max_tokens=max_tokens,
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
def openai_query(message, model, api_key, max_tokens, temperature=0.7, base_url=None):
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
@click.option("-m", "--model", default="gpt-4o", help="Model to use for generation")
@click.option(
    "-t", "--token", required=True, help="API token for the selected provider"
)
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
@click.option(
    "-o", "--output", help="Output file path (if not specified, output to stdout)"
)
def main(
    input_string, provider, model, token, base_url, max_tokens, temperature, output
):
    """Generate VTK XML file content using LLMs.

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
    if model == "gpt-4o":
        default_models = {
            "anthropic": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-1.5-pro",
            "nim": "meta/llama3-70b-instruct",
        }
        model = default_models.get(provider, model)

    # Initialize the VTK XML generator
    try:
        generator = VTKXMLGenerator(token, base_url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate the VTK XML content
    try:
        xml_content = generator.generate_xml(
            input_string, model, max_tokens, temperature
        )
    except ValueError as e:
        if "max_tokens" in str(e):
            print(f"\nError: {e}", file=sys.stderr)
            print(f"Current max_tokens: {max_tokens}", file=sys.stderr)
            print("Try increasing with: --max-tokens <higher_number>", file=sys.stderr)
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate XML structure (basic check)
    if xml_content.strip().startswith("<?xml") and "</VTKFile>" in xml_content:
        # Output to file or stdout
        if output:
            with open(output, "w") as f:
                f.write(xml_content)
            print(f"VTK XML content written to {output}")
        else:
            print(xml_content)
    else:
        print("Warning: Generated content may not be valid VTK XML", file=sys.stderr)
        if output:
            with open(output, "w") as f:
                f.write(xml_content)
            print(f"Content written to {output} (please verify)")
        else:
            print(xml_content)


if __name__ == "__main__":
    main()
