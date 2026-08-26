"""
CLI tests for VTK Prompt (excluding RAG functionality).
"""

from unittest.mock import Mock, patch

import pytest
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
            ("openai", "gpt-4.1"),
            ("anthropic", "claude-sonnet-5"),
            ("gemini", "gemini-2.5-pro"),
            ("nim", "meta/llama-3.3-70b-instruct"),
        ],
    )
    def test_provider_model_defaults(self, provider, expected_model):
        """Test provider model defaults."""
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
                    expected_model,
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

    def test_embed_mcp_and_mcp_url_mutually_exclusive(self):
        """--embed-mcp and --mcp-url cannot be used together."""
        result = self.runner.invoke(
            main,
            [
                "create sphere",
                "--token",
                "test-token",
                "--embed-mcp",
                "--mcp-url",
                "http://localhost:8000",
            ],
        )
        assert result.exit_code == 2
        assert "mutually exclusive" in result.output

    def test_embed_mcp_launches_embedded_server(self):
        """--embed-mcp launches the embedded server and forwards its URL."""
        with (
            patch("vtk_prompt.cli.embedded_mcp_server") as mock_embed,
            patch.object(VTKPromptClient, "__new__") as mock_new,
        ):
            mock_embed.return_value.__enter__ = Mock(return_value="http://127.0.0.1:12345")
            mock_embed.return_value.__exit__ = Mock(return_value=False)

            mock_client = Mock()
            mock_new.return_value = mock_client
            mock_client.query.return_value = ("explanation", "code", None)
            mock_client.run_code.return_value = None

            result = self.runner.invoke(
                main, ["create sphere", "--token", "test-token", "--embed-mcp"]
            )

            assert result.exit_code == 0
            mock_embed.assert_called_once()
            assert mock_new.call_args[1]["mcp_url"] == "http://127.0.0.1:12345"

    def test_embed_mcp_startup_failure(self):
        """A RuntimeError from the embedded server surfaces as exit code 4."""
        with patch("vtk_prompt.cli.embedded_mcp_server") as mock_embed:
            mock_embed.return_value.__enter__ = Mock(
                side_effect=RuntimeError("vtk-mcp is not installed")
            )

            result = self.runner.invoke(
                main, ["create sphere", "--token", "test-token", "--embed-mcp"]
            )

            assert result.exit_code == 4
