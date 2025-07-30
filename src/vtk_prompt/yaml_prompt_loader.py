#!/usr/bin/env python3

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import vtk
import re

PYTHON_VERSION = ">=3.10"
VTK_VERSION = vtk.__version__


class GitHubModelYAMLLoader:
    """Loader for GitHub Models YAML prompt files."""

    def __init__(self, prompts_dir: Optional[Path] = None):
        """Initialize with prompts directory path."""
        if prompts_dir is None:
            # Default to prompts directory in repository root
            prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        self.prompts_dir = Path(prompts_dir)

    def load_prompt(self, prompt_name: str) -> Dict[str, Any]:
        """Load a YAML prompt file.

        Args:
            prompt_name: Name of the prompt file (with or without .prompt.yml extension)

        Returns:
            Parsed YAML content as dictionary
        """
        # Handle both with and without extension
        if not prompt_name.endswith(".prompt.yml"):
            prompt_name = f"{prompt_name}.prompt.yml"

        prompt_path = self.prompts_dir / prompt_name
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_data = yaml.load(f, Loader=get_loader())

        return prompt_data

    def substitute_variables(self, content: str, variables: Dict[str, str]) -> str:
        """Substitute template variables in content using GitHub Models format.

        Args:
            content: Template content with {{variable}} placeholders
            variables: Dictionary of variable name -> value mappings

        Returns:
            Content with variables substituted
        """
        # Add default variables
        default_vars = {"VTK_VERSION": VTK_VERSION, "PYTHON_VERSION": PYTHON_VERSION}
        variables = {**default_vars, **variables}

        # Handle conditional blocks like {{#if variable}}...{{/if}}
        def handle_conditionals(text: str) -> str:
            # Simple conditional handling for {{#if variable}}...{{/if}}
            conditional_pattern = r"\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}"

            def replace_conditional(match):
                var_name = match.group(1)
                block_content = match.group(2)
                # Include block if variable exists and is truthy
                if var_name in variables and variables[var_name]:
                    return block_content
                return ""

            return re.sub(
                conditional_pattern, replace_conditional, text, flags=re.DOTALL
            )

        # First handle conditionals
        content = handle_conditionals(content)

        # Then substitute regular variables
        for var_name, var_value in variables.items():
            placeholder = f"{{{{{var_name}}}}}"
            content = content.replace(placeholder, str(var_value))

        return content

    def build_messages(
        self,
        prompt_name: str,
        variables: Dict[str, str] = None,
        system_only: bool = False,
    ) -> List[Dict[str, str]]:
        """Build messages list from YAML prompt with variable substitution.

        Args:
            prompt_name: Name of the prompt file
            variables: Variables to substitute in the template
            system_only: If True, return only the first system message content as string

        Returns:
            List of message dictionaries compatible with OpenAI API, or string if system_only=True
        """
        if variables is None:
            variables = {}

        prompt_data = self.load_prompt(prompt_name)
        messages = prompt_data.get("messages", [])

        # Substitute variables in each message
        processed_messages = []
        for message in messages:
            processed_message = {
                "role": message["role"],
                "content": self.substitute_variables(message["content"], variables),
            }
            processed_messages.append(processed_message)

        # If system_only is True, return only the first system message content
        if system_only:
            for message in processed_messages:
                if message["role"] == "system":
                    return message["content"]
            return ""  # No system message found

        return processed_messages

    def get_model_parameters(self, prompt_name: str) -> Dict[str, Any]:
        """Get model parameters from YAML prompt.

        Args:
            prompt_name: Name of the prompt file

        Returns:
            Dictionary of model parameters
        """
        prompt_data = self.load_prompt(prompt_name)
        return prompt_data.get("modelParameters", {})

    def get_model_name(self, prompt_name: str) -> str:
        """Get model name from YAML prompt.

        Args:
            prompt_name: Name of the prompt file

        Returns:
            Model name string
        """
        prompt_data = self.load_prompt(prompt_name)
        return prompt_data.get("model", "gpt-4o-mini")

    def list_available_prompts(self) -> List[str]:
        """List all available prompt files.

        Returns:
            List of prompt file names (without extension)
        """
        if not self.prompts_dir.exists():
            return []

        prompt_files = list(self.prompts_dir.glob("*.prompt.yml"))
        return [f.stem.replace(".prompt", "") for f in prompt_files]


# Convenience functions for backward compatibility
def get_yaml_prompt_messages(
    prompt_name: str, variables: Dict[str, str] = None
) -> List[Dict[str, str]]:
    """Get messages from a YAML prompt file.

    Args:
        prompt_name: Name of the prompt file
        variables: Variables to substitute in the template

    Returns:
        List of message dictionaries
    """
    loader = GitHubModelYAMLLoader()
    return loader.build_messages(prompt_name, variables)


def get_yaml_prompt_parameters(prompt_name: str) -> Dict[str, Any]:
    """Get model parameters from a YAML prompt file.

    Args:
        prompt_name: Name of the prompt file

    Returns:
        Dictionary of model parameters
    """
    loader = GitHubModelYAMLLoader()
    return loader.get_model_parameters(prompt_name)


def include_constructor(loader, node):
    data = loader.construct_scalar(node)

    match = re.search(r'\s', data)
    if match:
        file = data[:match.start()].strip()
        other_text = data[match.start():].strip()
    else:
        file = data.strip()
        other_text = ""

    with Path(file).resolve().open() as f:
        file_content = yaml.safe_load(f)

    return file_content + "\n" + other_text

def get_loader():
    loader = yaml.SafeLoader
    loader.add_constructor("!include", include_constructor)
    return loader
