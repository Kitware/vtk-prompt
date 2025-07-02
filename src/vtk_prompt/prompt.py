#!/usr/bin/env python3

import ast
import os
import re
import sys
import json
import openai
import click
from dataclasses import dataclass
from pathlib import Path

# Using YAML system exclusively
from .yaml_prompt_loader import GitHubModelYAMLLoader


@dataclass
class VTKPromptClient:
    """OpenAI client for VTK code generation."""

    collection_name: str = "vtk-examples"
    database_path: str = "./db/codesage-codesage-large-v2"
    verbose: bool = False
    conversation_file: str = None
    conversation: list = None

    def load_conversation(self):
        """Load conversation history from file."""
        if not self.conversation:
            return []

        if not self.conversation_file or not Path(self.conversation_file).exists():
            return []

        try:
            with open(self.conversation_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error: Could not load conversation file: {e}")

    def save_conversation(self):
        """Save conversation history to file."""
        if not self.conversation_file or not self.conversation:
            return

        try:
            # Ensure directory exists
            Path(self.conversation_file).parent.mkdir(parents=True, exist_ok=True)

            with open(self.conversation_file, "w") as f:
                json.dump(self.conversation, f, indent=2)
        except Exception as e:
            print(f"Error: Could not save conversation file: {e}")

    def validate_code_syntax(self, code_string):
        """Validate Python code syntax using AST."""
        try:
            ast.parse(code_string)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e.msg} at line {e.lineno}"
        except Exception as e:
            return False, f"AST parsing error: {str(e)}"

    def run_code(self, code_string):
        """Execute VTK code using exec() after AST validation."""
        is_valid, error_msg = self.validate_code_syntax(code_string)
        if not is_valid:
            print(f"Code validation failed: {error_msg}")
            if self.verbose:
                print("Generated code:")
                print(code_string)
            return None

        if self.verbose:
            print(code_string)

        try:
            exec(code_string, globals(), {})
        except Exception as e:
            print(f"Error executing code: {e}")
            if not self.verbose:
                print(code_string)
            return None

    def query_yaml(
        self,
        message,
        api_key,
        prompt_name="vtk_python_code_generation",
        base_url=None,
        rag=False,
        top_k=5,
        retry_attempts=1,
        override_model=None,
        override_temperature=None,
        override_max_tokens=None,
    ):
        """Generate VTK code using YAML prompt templates.

        Args:
            message: The user query
            api_key: API key for the service
            prompt_name: Name of the YAML prompt file to use
            base_url: API base URL
            rag: Whether to use RAG enhancement
            top_k: Number of RAG examples to retrieve
            retry_attempts: Number of retry attempts for failed generations

        Returns:
            Generated code string or None if failed
        """
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY or pass api_key parameter."
            )

        # Create client with current parameters
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # Load YAML prompt configuration
        from pathlib import Path

        prompts_dir = Path(__file__).parent / "prompts"
        yaml_loader = GitHubModelYAMLLoader(prompts_dir)
        model_params = yaml_loader.get_model_parameters(prompt_name)
        model = override_model or yaml_loader.get_model_name(prompt_name)

        # Prepare variables for template substitution
        variables = {"request": message}

        # Handle RAG if requested
        if rag:
            from .rag_chat_wrapper import (
                check_rag_components_available,
                get_rag_snippets,
            )

            if not check_rag_components_available():
                raise ValueError("RAG components not available")

            rag_snippets = get_rag_snippets(
                message,
                collection_name=self.collection_name,
                database_path=self.database_path,
                top_k=top_k,
            )

            if not rag_snippets:
                raise ValueError("Failed to load RAG snippets")

            context_snippets = "\n\n".join(rag_snippets["code_snippets"])
            variables["context_snippets"] = context_snippets

            if self.verbose:
                references = rag_snippets.get("references")
                if references:
                    print("Using examples from:")
                    for ref in references:
                        print(f"- {ref}")

        # Load existing conversation or start fresh
        conversation_messages = self.load_conversation()

        # Build base messages from YAML template
        base_messages = yaml_loader.build_messages(prompt_name, variables)

        # If conversation exists, extend it with new user message
        if conversation_messages:
            # Add the current request as a new user message
            conversation_messages.append({"role": "user", "content": message})
            self.conversation = conversation_messages
        else:
            # Use YAML template as starting point
            self.conversation = base_messages

        # Extract parameters with overrides
        temperature = override_temperature or model_params.get("temperature", 0.3)
        max_tokens = override_max_tokens or model_params.get("max_tokens", 2000)

        # Retry loop for AST validation
        for attempt in range(retry_attempts):
            if self.verbose:
                if attempt > 0:
                    print(f"Retry attempt {attempt + 1}/{retry_attempts}")
                print(f"Making request with model: {model}, temperature: {temperature}")
                for i, msg in enumerate(self.conversation):
                    print(f"Message {i} ({msg['role']}): {msg['content'][:100]}...")

            response = client.chat.completions.create(
                model=model,
                messages=self.conversation,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if hasattr(response, "choices") and len(response.choices) > 0:
                content = (
                    response.choices[0].message.content or "No content in response"
                )
                finish_reason = response.choices[0].finish_reason

                if finish_reason == "length":
                    raise ValueError(
                        f"Output was truncated due to max_tokens limit ({max_tokens}). Please increase max_tokens."
                    )

                generated_explanation = re.findall(
                    "<explanation>(.*?)</explanation>", content, re.DOTALL
                )[0]
                generated_code = re.findall("<code>(.*?)</code>", content, re.DOTALL)[0]
                if "import vtk" not in generated_code:
                    generated_code = f"import vtk\n{generated_code}"

                is_valid, error_msg = self.validate_code_syntax(generated_code)
                if is_valid:
                    # Save conversation with assistant response
                    self.conversation.append({"role": "assistant", "content": content})
                    self.save_conversation()

                    if self.verbose:
                        print("Code validation successful!")
                    return generated_code, generated_explanation
                else:
                    if self.verbose:
                        print(
                            f"Code validation failed on attempt {attempt + 1}: {error_msg}"
                        )
                        print("Generated code:")
                        print(generated_code)

                    if attempt < retry_attempts - 1:
                        # Add error feedback to messages for retry
                        error_feedback = (
                            f"The previous code had a syntax error: {error_msg}. "
                            "Please fix the syntax and try again."
                        )
                        self.conversation.append({"role": "user", "content": error_feedback})
                    else:
                        # Save conversation even if final attempt failed
                        self.conversation.append({"role": "assistant", "content": content})
                        self.save_conversation()
                        print(
                            f"All {retry_attempts} attempts failed. Final error: {error_msg}"
                        )
                        return generated_code, generated_explanation  # Return anyway, let caller handle
            else:
                print("No response content received")
                return None

        return None


@click.command()
@click.argument("input_string")
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "gemini", "nim"]),
    default="openai",
    help="LLM provider to use",
)
@click.option("-m", "--model", default="gpt-4o-mini", help="Model name to use")
@click.option(
    "-k", "--max-tokens", type=int, default=1000, help="Max # of tokens to generate"
)
@click.option(
    "--temperature",
    type=float,
    default=0.1,
    help="Temperature for generation (0.0-2.0)",
)
@click.option(
    "-t", "--token", required=True, help="API token for the selected provider"
)
@click.option("--base-url", help="Base URL for API (auto-detected or custom)")
@click.option("-r", "--rag", is_flag=True, help="Use RAG to improve code generation")
@click.option("-v", "--verbose", is_flag=True, help="Show generated source code")
@click.option("--collection", default="vtk-examples", help="Collection name for RAG")
@click.option(
    "--database",
    default="./db/codesage-codesage-large-v2",
    help="Database path for RAG",
)
@click.option(
    "--top-k", type=int, default=5, help="Number of examples to retrieve from RAG"
)
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
    input_string,
    provider,
    model,
    max_tokens,
    temperature,
    token,
    base_url,
    rag,
    verbose,
    collection,
    database,
    top_k,
    retry_attempts,
    conversation,
):
    """Generate and execute VTK code using LLMs with YAML prompts.

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
    if model == "gpt-4o":
        default_models = {
            "anthropic": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-1.5-pro",
            "nim": "meta/llama3-70b-instruct",
        }
        model = default_models.get(provider, model)

    try:
        client = VTKPromptClient(
            collection_name=collection,
            database_path=database,
            verbose=verbose,
            conversation_file=conversation,
        )

        # Use YAML system directly
        prompt_name = "rag_context" if rag else "no_rag_context"
        generated_code = client.query_yaml(
            input_string,
            api_key=token,
            prompt_name=prompt_name,
            base_url=base_url,
            rag=rag,
            top_k=top_k,
            retry_attempts=retry_attempts,
            # Override parameters if specified in CLI
            override_model=model if model != "gpt-4o-mini" else None,
            override_temperature=temperature if temperature != 0.1 else None,
            override_max_tokens=max_tokens if max_tokens != 1000 else None,
        )

        # Usage tracking not yet implemented for YAML system
        if verbose:
            print("Token usage tracking not available in YAML mode")

        client.run_code(generated_code)

    except ValueError as e:
        if "RAG components" in str(e):
            print("rag_components not found", file=sys.stderr)
            sys.exit(1)
        elif "Failed to load RAG snippets" in str(e):
            print("failed to load rag snippets", file=sys.stderr)
            sys.exit(2)

        elif "max_tokens" in str(e):
            print(f"\nError: {e}", file=sys.stderr)
            print(f"Current max_tokens: {max_tokens}", file=sys.stderr)
            print("Try increasing with: --max-tokens <higher_number>", file=sys.stderr)
            sys.exit(3)

        else:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(4)


if __name__ == "__main__":
    main()
