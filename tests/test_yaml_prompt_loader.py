"""
Tests for YAML prompt loader functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
import yaml

from vtk_prompt.prompts.yaml_prompt_loader import YAMLPromptLoader


class TestYAMLPromptLoader:
    """Test cases for YAMLPromptLoader."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = YAMLPromptLoader()

    def test_initialization(self):
        """Test that YAMLPromptLoader initializes correctly."""
        assert self.loader.prompts_dir.name == "prompts"
        assert self.loader.vtk_version is not None
        assert self.loader.python_version == ">=3.10"

    def test_variable_validation_valid_values(self):
        """Test that valid variable values pass validation."""
        assert self.loader._validate_variable_value("test", "hello") == "hello"
        assert self.loader._validate_variable_value("number", 42) == "42"
        assert self.loader._validate_variable_value("float", 3.14) == "3.14"
        assert self.loader._validate_variable_value("bool", True) == "True"

    def test_variable_validation_dangerous_content(self):
        """Test that dangerous variable values are rejected."""
        dangerous_values = [
            "__import__",
            "__builtins__",
            "${HOME}",
            "${USER}",
            "`ls -la`",
            "`rm -rf /`",
            "{{nested}}",
            "{{another_var}}",
        ]

        for dangerous_value in dangerous_values:
            with pytest.raises(ValueError, match="potentially dangerous content"):
                self.loader._validate_variable_value("test", dangerous_value)

    def test_variable_validation_none_value(self):
        """Test that None values are rejected."""
        with pytest.raises(ValueError, match="cannot be None"):
            self.loader._validate_variable_value("test", None)

    def test_variable_validation_too_long(self):
        """Test that overly long values are rejected."""
        long_value = "x" * 10001  # Exceeds 10000 character limit
        with pytest.raises(ValueError, match="too long"):
            self.loader._validate_variable_value("test", long_value)

    def test_variable_validation_boundary_length(self):
        """Test that exactly 10000 character values are accepted."""
        boundary_value = "x" * 10000  # Exactly at limit
        result = self.loader._validate_variable_value("test", boundary_value)
        assert result == boundary_value

    def test_substitute_yaml_variables_basic(self):
        """Test basic YAML variable substitution."""
        content = "Hello {{name}}, you are {{age}} years old."
        variables = {"name": "Alice", "age": 30}

        result = self.loader.substitute_yaml_variables(content, variables)
        assert result == "Hello Alice, you are 30 years old."

    def test_substitute_yaml_variables_complex(self):
        """Test complex YAML variable substitution with multiple occurrences."""
        content = """
        name: {{project_name}}
        description: {{project_name}} is a {{type}} project
        version: {{version}}
        """
        variables = {"project_name": "VTK-Prompt", "type": "visualization", "version": "1.0.0"}

        result = self.loader.substitute_yaml_variables(content, variables)
        assert "VTK-Prompt" in result
        assert "visualization" in result
        assert "1.0.0" in result
        assert "{{" not in result  # No unresolved placeholders

    def test_substitute_yaml_variables_no_variables(self):
        """Test substitution with no variables provided."""
        content = "Simple content with no placeholders"
        result = self.loader.substitute_yaml_variables(content, {})
        assert result == content

    def test_substitute_yaml_variables_empty_content(self):
        """Test substitution with empty content."""
        result = self.loader.substitute_yaml_variables("", {"key": "value"})
        assert result == ""

    def test_load_yaml_prompt_file_not_found(self):
        """Test that missing files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            self.loader.load_yaml_prompt("nonexistent_prompt")

    def test_load_yaml_prompt_invalid_yaml(self):
        """Test that invalid YAML raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid YAML file
            yaml_file = Path(temp_dir) / "test.prompt.yml"
            yaml_file.write_text("invalid: yaml: content: [")

            # Mock prompts_dir to point to temp directory
            with patch.object(self.loader, "prompts_dir", Path(temp_dir)):
                with pytest.raises(ValueError, match="Invalid YAML"):
                    self.loader.load_yaml_prompt("test")

    def test_load_yaml_prompt_not_dict(self):
        """Test that YAML not containing a dictionary raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create YAML that's a list, not a dict
            yaml_file = Path(temp_dir) / "test.prompt.yml"
            yaml_file.write_text(yaml.dump(["item1", "item2"]))

            with patch.object(self.loader, "prompts_dir", Path(temp_dir)):
                with pytest.raises(ValueError, match="must contain a dictionary at root level"):
                    self.loader.load_yaml_prompt("test")

    def test_load_yaml_prompt_valid_structure(self):
        """Test loading a valid YAML prompt."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid YAML file
            valid_data = {
                "name": "Test Prompt",
                "description": "A test prompt for {{purpose}}",
                "messages": [
                    {"role": "system", "content": "You are using {{VTK_VERSION}}"},
                    {"role": "user", "content": "{{request}}"},
                ],
            }
            yaml_file = Path(temp_dir) / "test.prompt.yml"
            yaml_file.write_text(yaml.dump(valid_data))

            with patch.object(self.loader, "prompts_dir", Path(temp_dir)):
                result = self.loader.load_yaml_prompt("test", purpose="testing", request="help me")

                assert result["name"] == "Test Prompt"
                assert "testing" in result["description"]
                assert self.loader.vtk_version in result["messages"][0]["content"]
                assert result["messages"][1]["content"] == "help me"

    def test_load_yaml_prompt_default_variables(self):
        """Test that default VTK and Python version variables are included."""
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_data = {
                "messages": [
                    {
                        "role": "system",
                        "content": "VTK: {{VTK_VERSION}}, Python: {{PYTHON_VERSION}}",
                    }
                ]
            }
            yaml_file = Path(temp_dir) / "test.prompt.yml"
            yaml_file.write_text(yaml.dump(valid_data))

            with patch.object(self.loader, "prompts_dir", Path(temp_dir)):
                result = self.loader.load_yaml_prompt("test")

                content = result["messages"][0]["content"]
                assert self.loader.vtk_version in content
                assert self.loader.python_version in content

    def test_load_yaml_prompt_variable_override(self):
        """Test that provided variables override defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_data = {"messages": [{"role": "system", "content": "Version: {{VTK_VERSION}}"}]}
            yaml_file = Path(temp_dir) / "test.prompt.yml"
            yaml_file.write_text(yaml.dump(valid_data))

            with patch.object(self.loader, "prompts_dir", Path(temp_dir)):
                result = self.loader.load_yaml_prompt("test", VTK_VERSION="custom-version")

                content = result["messages"][0]["content"]
                assert "custom-version" in content
                assert self.loader.vtk_version not in content

    def test_get_yaml_prompt_integration(self):
        """Test the full get_yaml_prompt workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_data = {
                "name": "Integration Test",
                "messages": [
                    {"role": "system", "content": "You are {{role}}", "id": "sys1"},
                    {"role": "user", "content": "{{request}}", "timestamp": "2024-01-01"},
                ],
            }
            yaml_file = Path(temp_dir) / "test.prompt.yml"
            yaml_file.write_text(yaml.dump(valid_data))

            with patch.object(self.loader, "prompts_dir", Path(temp_dir)):
                result = self.loader.get_yaml_prompt("test", role="assistant", request="help")

                expected = [
                    {"role": "system", "content": "You are assistant"},
                    {"role": "user", "content": "help"},
                ]

                assert result == expected

    def test_dangerous_variable_patterns_comprehensive(self):
        """Test comprehensive dangerous pattern detection."""
        dangerous_patterns = [
            # Python dunder methods
            "__import__",
            "__builtins__",
            "__globals__",
            "__locals__",
            "__dir__",
            # Shell variable expansion
            "${HOME}",
            "${USER}",
            "${PATH}",
            "${}",
            # Command substitution
            "`whoami`",
            "`cat /etc/passwd`",
            "`rm -rf *`",
            "``",
            # Nested template variables
            "{{another_var}}",
            "{{ spaced_var }}",
            "{{nested_{{inner}}}}",
        ]

        for pattern in dangerous_patterns:
            with pytest.raises(ValueError, match="potentially dangerous content"):
                self.loader._validate_variable_value("test", pattern)

    def test_safe_variable_patterns(self):
        """Test that safe patterns are accepted."""
        safe_patterns = [
            "normal_text",
            "under_scores_ok",
            "numbers123",
            "UPPERCASE_TEXT",
            "mixed_Case_123",
            "special!@#$%^&*()chars",
            "newlines\nare\nok",
            "tabs\tare\tOK",
            "unicode_文字_characters",
            "single_underscore_",
            "_leading_underscore",
            "{single_braces}",
            "{{malformed_template",
            "malformed_template}}",
        ]

        for pattern in safe_patterns:
            result = self.loader._validate_variable_value("test", pattern)
            assert result == str(pattern)

    def test_error_message_details(self):
        """Test that error messages provide helpful details."""
        # Test None value error
        with pytest.raises(ValueError, match="Variable 'test_var' cannot be None"):
            self.loader._validate_variable_value("test_var", None)

        # Test dangerous content error
        with pytest.raises(
            ValueError, match="Variable 'danger' contains potentially dangerous content: __import__"
        ):
            self.loader._validate_variable_value("danger", "__import__")

        # Test length error
        long_value = "x" * 10001
        with pytest.raises(ValueError, match="Variable 'long' is too long"):
            self.loader._validate_variable_value("long", long_value)
