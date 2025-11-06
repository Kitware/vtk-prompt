"""
Pytest configuration and fixtures for VTK Prompt tests.

Provides common test fixtures, mock setups, and utilities for testing
the VTK Prompt client across all provider/model combinations.
"""

import json
import os
from unittest.mock import MagicMock, Mock

import pytest

from vtk_prompt.provider_utils import get_available_models, get_supported_providers


def require_api_key(provider_name: str, env_var_name: str) -> str:
    """
    Skip test if API key not available, with helpful message.

    Args:
        provider_name: Human-readable provider name (e.g., "OpenAI", "Anthropic")
        env_var_name: Environment variable name (e.g., "OPENAI_API_KEY")

    Returns:
        The API key value

    Raises:
        pytest.skip: If the API key is not set in environment
    """
    api_key = os.getenv(env_var_name)
    if api_key:
        return api_key
    # In CI, missing keys should fail to catch misconfiguration early.
    is_ci = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"
    if is_ci:
        pytest.fail(
            f"{env_var_name} is required in CI for {provider_name} smoke tests. "
            f"Configure this environment variable in your CI environment.",
            pytrace=False,
        )
    # Local/dev: skip gracefully with a helpful message.
    pytest.skip(
        f"{env_var_name} environment variable not set. "
        f"Set your {provider_name} API key to run this smoke test: "
        f"export {env_var_name}='your-key-here'"
    )


@pytest.fixture
def mock_openai_response():
    """Mock successful OpenAI API response."""
    mock_response = Mock()
    mock_response.choices = [
        Mock(
            message=Mock(
                content="<explanation>This creates a simple VTK sphere.</explanation>"
                "<code>import vtk\n\nsphere = vtk.vtkSphereSource()\n"
                "mapper = vtk.vtkPolyDataMapper()\n"
                "mapper.SetInputConnection(sphere.GetOutputPort())\n"
                "actor = vtk.vtkActor()\nactor.SetMapper(mapper)\n"
                "renderer = vtk.vtkRenderer()\n"
                "renderer.AddActor(actor)\n</code>"
            ),
            finish_reason="stop",
        )
    ]
    mock_response.usage = Mock(prompt_tokens=100, completion_tokens=150)
    return mock_response


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI client with successful response."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    return mock_client


@pytest.fixture
def mock_truncated_response():
    """Mock OpenAI response that was truncated (finish_reason='length')."""
    mock_response = Mock()
    mock_response.choices = [
        Mock(
            message=Mock(content="<explanation>Partial response...</explanation><code>import vtk"),
            finish_reason="length",
        )
    ]
    mock_response.usage = Mock(prompt_tokens=100, completion_tokens=1000)
    return mock_response


@pytest.fixture
def mock_invalid_syntax_response():
    """Mock OpenAI response with invalid Python syntax."""
    mock_response = Mock()
    mock_response.choices = [
        Mock(
            message=Mock(
                content="<explanation>Creates a sphere with syntax error.</explanation>"
                "<code>import vtk\n\nif True\n  sphere = vtk.vtkSphereSource()\n</code>"
            ),
            finish_reason="stop",
        )
    ]
    mock_response.usage = Mock(prompt_tokens=100, completion_tokens=80)
    return mock_response


@pytest.fixture
def test_api_key():
    """Test API key for mocking."""
    return "test-api-key-12345"


@pytest.fixture
def all_provider_model_combinations():
    """Generate all provider/model combinations for comprehensive testing."""
    combinations = []
    available_models = get_available_models()

    for provider in get_supported_providers():
        models = available_models.get(provider, [])
        for model in models:
            combinations.append(
                {"provider": provider, "model": model, "base_url": _get_provider_base_url(provider)}
            )

    return combinations


def _get_provider_base_url(provider: str) -> str | None:
    """Get the appropriate base URL for each provider."""
    base_urls = {
        "anthropic": "https://api.anthropic.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "nim": "https://integrate.api.nvidia.com/v1",
        "openai": None,  # Uses default OpenAI base URL
    }
    return base_urls.get(provider)


@pytest.fixture
def sample_vtk_request():
    """Sample VTK generation request for testing."""
    return "Create a red sphere in the center of the scene"


@pytest.fixture
def mock_environment_variables(monkeypatch):
    """Mock environment variables to avoid requiring real API keys."""
    test_keys = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GOOGLE_API_KEY": "test-google-key",
        "NVIDIA_API_KEY": "test-nvidia-key",
    }

    for key, value in test_keys.items():
        monkeypatch.setenv(key, value)

    return test_keys


@pytest.fixture
def clean_client_state():
    """Ensure VTKPromptClient singleton is reset between tests."""
    # Reset the singleton instance
    from vtk_prompt.client import VTKPromptClient

    VTKPromptClient._instance = None
    VTKPromptClient._initialized = False
    yield
    # Clean up after test
    VTKPromptClient._instance = None
    VTKPromptClient._initialized = False


@pytest.fixture
def temporary_conversation_file(tmp_path):
    """Create a temporary conversation file for testing."""
    conv_file = tmp_path / "test_conversation.json"
    conversation_data = [
        {"role": "system", "content": "You are a VTK code generator."},
        {"role": "user", "content": "Create a cube"},
        {"role": "assistant", "content": "Here's a VTK cube..."},
    ]

    with open(conv_file, "w") as f:
        json.dump(conversation_data, f)

    return str(conv_file)
