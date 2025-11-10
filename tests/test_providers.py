"""
Comprehensive smoke tests for all LLM providers and models.

These tests make real API calls to verify that:
1. Authentication works with environment API keys
2. Models are available and responding
3. Requests complete without errors
4. Response structure is valid

Tests will skip gracefully if API keys are not available.
Each provider/model combination is tested individually to catch deprecations early.
"""

import pytest

from vtk_prompt.client import VTKPromptClient
from vtk_prompt.provider_utils import (
    OPENAI_MODELS,
    ANTHROPIC_MODELS,
    GEMINI_MODELS,
    NIM_MODELS,
    get_model_temperature,
)

from .conftest import require_api_key


class TestProviders:
    """Tests for all LLM providers using real API calls."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Reset VTKPromptClient singleton before and after each test for test isolation."""
        # Reset singleton. Ensures each test starts with a fresh, clean instance.
        VTKPromptClient._instance = None
        VTKPromptClient._initialized = False

        yield  # Test runs

        # Clean up after test completes
        VTKPromptClient._instance = None
        VTKPromptClient._initialized = False

    def _run_model_test(self, model: str, api_key: str, base_url=None):
        """Generalized test logic for any model."""
        client = VTKPromptClient()

        # Use appropriate temperature for model
        temperature = get_model_temperature(model, requested_temperature=1)
        result = client.query(
            message="Create a simple sphere",
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_tokens=5000,
            temperature=temperature,
        )

        # Verify response structure and content
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3 elements, got {len(result)}"

        explanation, code, usage = result

        assert isinstance(explanation, str), f"Expected str explanation, got {type(explanation)}"
        assert len(explanation.strip()) > 0, "Explanation should not be empty"

        assert isinstance(code, str), f"Expected str code, got {type(code)}"
        assert len(code.strip()) > 0, "Code should not be empty"

        assert "import vtk" in code, "Generated code should contain VTK import"

    # OpenAI Provider Tests
    @pytest.mark.parametrize("model", OPENAI_MODELS)
    def test_openai_model(self, model):
        """Test OpenAI models individually to catch deprecations early."""
        api_key = require_api_key("OpenAI", "OPENAI_API_KEY")
        self._run_model_test(model, api_key)

    def test_openai_api_key_missing(self, monkeypatch):
        """Test proper error handling when OpenAI API key is missing."""
        client = VTKPromptClient()
        # Ensure no fallback is available via environment for duration of test
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key provided"):
            client.query(message="Create a sphere", api_key=None, model="gpt-5")

    # Anthropic Provider Tests
    @pytest.mark.parametrize("model", ANTHROPIC_MODELS)
    def test_anthropic_model(self, model):
        """Test Anthropic models individually to catch deprecations early."""
        api_key = require_api_key("Anthropic", "ANTHROPIC_API_KEY")
        self._run_model_test(
            model,
            api_key,
            base_url="https://api.anthropic.com/v1",
        )

    def test_anthropic_api_key_missing(self, monkeypatch):
        """Test proper error handling when Anthropic API key is missing."""
        client = VTKPromptClient()
        # Ensure no fallback is available via environment for duration of test
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key provided"):
            client.query(message="Create a sphere", api_key=None, model="claude-opus-4-1-20250805")

    # Gemini Provider Tests
    @pytest.mark.parametrize("model", GEMINI_MODELS)
    def test_gemini_model(self, model):
        """Test Gemini models individually to catch deprecations early."""
        api_key = require_api_key("Google", "GOOGLE_API_KEY")
        self._run_model_test(
            model,
            api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def test_gemini_api_key_missing(self, monkeypatch):
        """Test proper error handling when Google API key is missing."""
        client = VTKPromptClient()
        # Ensure no fallback is available via environment for duration of test
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key provided"):
            client.query(message="Create a sphere", api_key=None, model="gemini-2.5-pro")

    # NVIDIA NIM Provider Tests
    @pytest.mark.parametrize("model", NIM_MODELS)
    def test_nim_model(self, model):
        """Test NVIDIA NIM models individually to catch deprecations early."""
        api_key = require_api_key("NVIDIA", "NVIDIA_API_KEY")
        # Use NVIDIA OpenAI-compatible endpoint
        self._run_model_test(
            model,
            api_key,
            base_url="https://integrate.api.nvidia.com/v1",
        )

    def test_nim_api_key_missing(self, monkeypatch):
        """Test proper error handling when NVIDIA API key is missing."""
        client = VTKPromptClient()
        # Ensure no fallback is available via environment for duration of test
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key provided"):
            client.query(message="Create a sphere", api_key=None, model="meta/llama3-70b-instruct")
