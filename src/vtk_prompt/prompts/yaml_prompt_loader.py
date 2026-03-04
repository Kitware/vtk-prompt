"""
YAML Prompt Loader for VTK Prompt System.

This module provides a singleton class for loading and processing YAML prompts
used in VTK code generation. It supports variable substitution and message
formatting for LLM clients.
"""

import re
from pathlib import Path
from typing import Any, Dict, List
import yaml
import vtk

PYTHON_VERSION = ">=3.10"
VTK_VERSION = vtk.__version__

# Path to the prompts directory
PROMPTS_DIR = Path(__file__).parent


class YAMLPromptLoader:
    """Class for loading and processing YAML prompts."""

    def __init__(self) -> None:
        """Initialize the loader."""
        self.prompts_dir = PROMPTS_DIR
        self.vtk_version = VTK_VERSION
        self.python_version = PYTHON_VERSION

    def _validate_variable_value(self, key: str, value: Any) -> str:
        """Validate and sanitize variable values for YAML substitution.

        Args:
            key: Variable name
            value: Variable value

        Returns:
            Sanitized string value

        Raises:
            ValueError: If value is invalid or potentially dangerous
        """
        if value is None:
            raise ValueError(f"Variable '{key}' cannot be None")

        str_value = str(value)

        # Check for potentially dangerous content
        dangerous_patterns = [
            r"__[a-zA-Z]+__",  # Python double underscore methods
            r"\$\{[^}]*\}",  # Shell-style variable expansion
            r"`[^`]*`",  # Command substitution
            r"\{\{[^}]*\}\}",  # Nested template variables
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, str_value):
                raise ValueError(
                    f"Variable '{key}' contains potentially dangerous content: {str_value}"
                )

        # Length limit to prevent DoS
        if len(str_value) > 10000:
            raise ValueError(f"Variable '{key}' is too long (max 10000 characters)")

        return str_value

    def substitute_yaml_variables(self, content: str, variables: Dict[str, Any]) -> str:
        """Substitute {{variable}} placeholders in YAML content.

        Args:
            content: String content with {{variable}} placeholders
            variables: Dictionary of variable names to values

        Returns:
            Content with variables substituted

        Raises:
            ValueError: If variable validation fails
        """
        result = content
        for key, value in variables.items():
            validated_value = self._validate_variable_value(key, value)
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, validated_value)

        return result

    def load_yaml_prompt(self, prompt_name: str, **variables: Any) -> Dict[str, Any]:
        """Load a YAML prompt file and substitute variables.

        Args:
            prompt_name: Name of the prompt file (without .prompt.yml extension)
            **variables: Variables to substitute in the prompt

        Returns:
            Dictionary containing the prompt structure
        """
        # Try .prompt.yml first, then .prompt.yaml
        yaml_path = self.prompts_dir / f"{prompt_name}.prompt.yml"
        if not yaml_path.exists():
            yaml_path = self.prompts_dir / f"{prompt_name}.prompt.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML prompt {prompt_name} not found at {self.prompts_dir}")

        # Load YAML content
        yaml_content = yaml_path.read_text()

        # Add default variables
        default_variables = {
            "VTK_VERSION": self.vtk_version,
            "PYTHON_VERSION": self.python_version,
        }
        all_variables = {**default_variables, **variables}

        # Substitute variables in the raw YAML string
        substituted_content = self.substitute_yaml_variables(yaml_content, all_variables)

        # Parse the substituted YAML
        try:
            result = yaml.safe_load(substituted_content)

            # Validate YAML structure
            if not isinstance(result, dict):
                raise ValueError(
                    f"YAML prompt {prompt_name} must contain a dictionary at root level"
                )

            return result

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in prompt {prompt_name}: {e}")

    def format_messages_for_client(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages from YAML prompt for LLM client.

        Args:
            messages: List of message dictionaries from YAML prompt

        Returns:
            Formatted messages ready for LLM client
        """
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    def get_yaml_prompt(self, prompt_name: str, **variables: Any) -> List[Dict[str, str]]:
        """Get a YAML prompt and format it for the LLM client.

        Args:
            prompt_name: Name of the prompt file (without .prompt.yml extension)
            **variables: Variables to substitute in the prompt

        Returns:
            Formatted messages ready for LLM client
        """
        yaml_prompt = self.load_yaml_prompt(prompt_name, **variables)
        return self.format_messages_for_client(yaml_prompt["messages"])
