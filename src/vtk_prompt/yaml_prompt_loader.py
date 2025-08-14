#!/usr/bin/env python3

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import vtk
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

PYTHON_VERSION = ">=3.10"
VTK_VERSION = vtk.__version__


@dataclass
class LoaderConfig:
    prompts_dir: Path
    default_extension: str = ".prompt.yml"
    encoding: str = "utf-8"


class PromptSource(ABC):
    @abstractmethod
    def get_content(self) -> str:
        """Get the raw content as string."""
        pass


class FilePromptSource(PromptSource):
    # CLI provides a filename
    def __init__(self, filename: str, config: LoaderConfig):
        self.filename = filename
        self.config = config

    def get_content(self) -> str:
        filename = self.filename
        if not filename.endswith(self.config.default_extension):
            filename = f"{filename}{self.config.default_extension}"

        prompt_path = self.config.prompts_dir / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        try:
            with open(prompt_path, "r", encoding=self.config.encoding) as f:
                return f.read()
        except IOError as e:
            raise IOError(f"Failed to read prompt file {prompt_path}: {e}")


class ContentPromptSource(PromptSource):
    # UI provides a content blob
    def __init__(self, content: Union[str, bytes]):
        if isinstance(content, bytes):
            self.content = content.decode('utf-8')
        else:
            self.content = content

    def get_content(self) -> str:
        return self.content


class GitHubModelYAMLLoader:
    """Loader for GitHub Models YAML prompt files."""

    def __init__(self, prompts_dir: Optional[Path] = None):
        """Initialize with prompts directory path."""
        if prompts_dir is None:
            # Default to prompts directory in repository root
            prompts_dir = Path(__file__).parent.parent.parent / "prompts"

        self.config = LoaderConfig(prompts_dir=Path(prompts_dir))

    @classmethod
    def from_file(cls, filename: str, prompts_dir: Optional[Path] = None) -> 'GitHubModelYAMLLoader':
        loader = cls(prompts_dir)
        loader._current_source = FilePromptSource(filename, loader.config)
        return loader

    @classmethod
    def from_content(cls, content: Union[str, bytes]) -> 'GitHubModelYAMLLoader':
        loader = cls()
        loader._current_source = ContentPromptSource(content)
        return loader

    def _get_prompt_source(self, prompt: Union[str, bytes]) -> PromptSource:
        if (
            isinstance(prompt, (str, bytes)) and
            not isinstance(prompt, str) or
            isinstance(prompt, bytes)
        ):
            return ContentPromptSource(prompt)
        else:
            return FilePromptSource(prompt, self.config)

    def _parse_yaml_content(self, content: str) -> Dict[str, Any]:
        try:
            return yaml.load(content, Loader=get_loader())
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML content: {e}")

    def load_prompt(self, prompt: Union[str, bytes]) -> Dict[str, Any]:
        """Load a YAML prompt file.

        Args:
            prompt: Name of the prompt file (with or without .prompt.yml extension) or binary blob
                    of the prompt file content

        Returns:
            Parsed YAML content as dictionary
        """
        source = self._get_prompt_source(prompt)
        content = source.get_content()
        return self._parse_yaml_content(content)

    def save_prompt(self, prompt: str) -> None:
        # TODO
        return

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
        prompt: str | bytes,
        variables: Dict[str, str] = None,
        system_only: bool = False,
    ) -> List[Dict[str, str]]:
        """Build messages list from YAML prompt with variable substitution.

        Args:
            prompt: Name of the prompt file or binary blob of the prompt file content
            variables: Variables to substitute in the template
            system_only: If True, return only the first system message content as string

        Returns:
            List of message dictionaries compatible with OpenAI API, or string if system_only=True
        """
        if variables is None:
            variables = {}

        prompt_data = self.load_prompt(prompt)
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

    def get_model_parameters(self, prompt: str | bytes) -> Dict[str, Any]:
        """Get model parameters from YAML prompt.

        Args:
            prompt: Name of the prompt file or binary blob of the prompt file content

        Returns:
            Dictionary of model parameters
        """
        prompt_data = self.load_prompt(prompt)
        return prompt_data.get("modelParameters", {})

    def get_model_name(self, prompt: str | bytes) -> str:
        """Get model name from YAML prompt.

        Args:
            prompt: Name of the prompt file or binary blob of the prompt file content

        Returns:
            Model name string
        """
        prompt_data = self.load_prompt(prompt)
        return prompt_data.get("model", "openai/gpt-4o-mini").split("/")[1]

    def get_model_provider(self, prompt: str | bytes) -> str:
        """Get model provider from YAML prompt.

        Args:
            prompt: Name of the prompt file or binary blob of the prompt file content

        Returns:
            Model name string
        """
        prompt_data = self.load_prompt(prompt)
        return prompt_data.get("model", "openai/gpt-4o-mini").split("/")[0]

    def list_available_prompts(self) -> List[str]:
        """List all available prompt files.

        Returns:
            List of prompt file names (without extension)
        """
        if not self.config.prompts_dir.exists():
            return []

        prompt_files = list(self.config.prompts_dir.glob("*.prompt.yml"))
        return [f.stem.replace(".prompt", "") for f in prompt_files]


# Convenience functions for backward compatibility
def get_yaml_prompt_messages(
    prompt: str | bytes, variables: Dict[str, str] = None
) -> List[Dict[str, str]]:
    """Get messages from a YAML prompt file.

    Args:
        prompt: Name of the prompt file or binary blob of the prompt file content
        variables: Variables to substitute in the template

    Returns:
        List of message dictionaries
    """
    loader = GitHubModelYAMLLoader()
    return loader.build_messages(prompt, variables)


def get_yaml_prompt_parameters(prompt: str | bytes) -> Dict[str, Any]:
    """Get model parameters from a YAML prompt file.

    Args:
        prompt: Name of the prompt file or binary blob of the prompt file content

    Returns:
        Dictionary of model parameters
    """
    loader = GitHubModelYAMLLoader()
    return loader.get_model_parameters(prompt)


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
