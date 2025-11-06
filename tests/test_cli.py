"""
CLI tests for VTK Prompt (excluding RAG functionality).
"""

import pytest
from unittest.mock import Mock, patch
from click.testing import CliRunner

from vtk_prompt.cli import main
from vtk_prompt.client import VTKPromptClient


class TestCLI:
    """Test CLI functionality."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_missing_required_args(self):
        """Test missing required arguments."""
        result = self.runner.invoke(main, [])
        assert result.exit_code == 2
        assert "Missing argument" in result.output

        result = self.runner.invoke(main, ["create sphere"])
        assert result.exit_code == 2
        assert "Missing option" in result.output

    def test_basic_execution(self):
        """Test basic CLI execution."""
        with patch.object(VTKPromptClient, "__new__") as mock_new:
            mock_client = Mock()
            mock_new.return_value = mock_client
            mock_client.query.return_value = ("explanation", "code", None)
            mock_client.run_code.return_value = None

            result = self.runner.invoke(main, ["create sphere", "--token", "test-token"])

            assert result.exit_code == 0
            mock_client.query.assert_called_once()

    @pytest.mark.parametrize(
        "provider,expected_url",
        [
            ("openai", None),
            ("anthropic", "https://api.anthropic.com/v1"),
            ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            ("nim", "https://integrate.api.nvidia.com/v1"),
        ],
    )
    def test_provider_base_urls(self, provider, expected_url):
        """Test provider base URL defaults."""
        with patch.object(VTKPromptClient, "__new__") as mock_new:
            mock_client = Mock()
            mock_new.return_value = mock_client
            mock_client.query.return_value = ("explanation", "code", None)
            mock_client.run_code.return_value = None

            result = self.runner.invoke(
                main, ["create sphere", "--token", "test-token", "--provider", provider]
            )

            assert result.exit_code == 0
            kwargs = mock_client.query.call_args[1]
            assert kwargs["base_url"] == expected_url

    @patch("vtk_prompt.cli.supports_temperature")
    def test_temperature_override(self, mock_supports_temp):
        """Test temperature override for unsupported models."""
        mock_supports_temp.return_value = False

        with patch.object(VTKPromptClient, "__new__") as mock_new:
            mock_client = Mock()
            mock_new.return_value = mock_client
            mock_client.query.return_value = ("explanation", "code", None)
            mock_client.run_code.return_value = None

            result = self.runner.invoke(
                main, ["create sphere", "--token", "test-token", "--temperature", "0.5"]
            )

            assert result.exit_code == 0
            kwargs = mock_client.query.call_args[1]
            assert kwargs["temperature"] == 1.0

    def test_max_tokens_error(self):
        """Test max_tokens error handling."""
        with patch.object(VTKPromptClient, "__new__") as mock_new:
            mock_client = Mock()
            mock_new.return_value = mock_client
            mock_client.query.side_effect = ValueError("max_tokens exceeded")

            result = self.runner.invoke(main, ["create sphere", "--token", "test-token"])

            assert result.exit_code == 3

    @pytest.mark.parametrize(
        "provider,expected_model",
        [
            ("openai", "gpt-5"),
            ("anthropic", "claude-opus-4-1-20250805"),
            ("gemini", "gemini-2.5-pro"),
            ("nim", "meta/llama3-70b-instruct"),
        ],
    )
    def test_provider_model_defaults(self, provider, expected_model):
        """Test provider model defaults when using gpt-5."""
        with patch.object(VTKPromptClient, "__new__") as mock_new:
            mock_client = Mock()
            mock_new.return_value = mock_client
            mock_client.query.return_value = ("explanation", "code", None)
            mock_client.run_code.return_value = None

            result = self.runner.invoke(
                main,
                [
                    "create sphere",
                    "--token",
                    "test-token",
                    "--provider",
                    provider,
                    "--model",
                    "gpt-5",  # Should map to provider default
                ],
            )

            assert result.exit_code == 0
            kwargs = mock_client.query.call_args[1]
            assert kwargs["model"] == expected_model

    def test_numeric_validation(self):
        """Test numeric argument validation."""
        # Invalid max_tokens
        result = self.runner.invoke(
            main, ["create sphere", "--token", "test-token", "--max-tokens", "not-a-number"]
        )
        assert result.exit_code == 2
        assert "Invalid value" in result.output

        # Invalid temperature
        result = self.runner.invoke(
            main, ["create sphere", "--token", "test-token", "--temperature", "not-a-float"]
        )
        assert result.exit_code == 2
        assert "Invalid value" in result.output

    def test_general_error_handling(self):
        """Test general error handling."""
        with patch.object(VTKPromptClient, "__new__") as mock_new:
            mock_client = Mock()
            mock_new.return_value = mock_client
            mock_client.query.side_effect = ValueError("Some general error")

            result = self.runner.invoke(main, ["create sphere", "--token", "test-token"])

            assert result.exit_code == 4
