"""
Configuration State Helpers Module.

This module provides helper functions for managing configuration state,
including API keys, base URLs, model selection, and configuration summaries.
"""

from typing import Any

from ..provider_utils import DEFAULT_MODEL

# Conventional environment variable per cloud provider, used when the active
# conversation carries no token of its own. GEMINI_API_KEY and GOOGLE_API_KEY
# are both in circulation for Gemini, so the first is tried and the second is
# handled as a fallback below.
_PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "nim": "NVIDIA_API_KEY",
}
_PROVIDER_ENV_FALLBACKS = {
    "gemini": "GOOGLE_API_KEY",
    "nim": "NIM_API_KEY",
}


def get_api_key(app: Any) -> str | None:
    """Get API key from state.

    Cloud providers need a real key. Local models hit an OpenAI-compatible
    endpoint (Ollama, LM Studio, ...) that ignores auth, but the OpenAI client
    still refuses to construct with an empty key, so in local mode we fall back
    to a placeholder. A real key typed into the field overrides it.

    The token is per-conversation, so a conversation that has none falls back to
    the provider's environment variable. Startup only seeds ``OPENAI_API_KEY``
    into state, which left the other providers with no path at all short of
    typing the key in by hand.
    """
    api_token = getattr(app.state, "api_token", "")
    if api_token and api_token.strip():
        return api_token.strip()
    if not app.state.use_cloud_models:
        return "sk-no-key-required"

    import os

    provider = getattr(app.state, "provider", "")
    for var in (
        _PROVIDER_ENV_VARS.get(provider),
        _PROVIDER_ENV_FALLBACKS.get(provider),
    ):
        if not var:
            continue
        from_env = os.environ.get(var, "").strip()
        if from_env:
            return from_env
    return None


def get_base_url(app: Any) -> str | None:
    """Get base URL based on configuration mode."""
    if app.state.use_cloud_models:
        # Use predefined base URLs for cloud providers (OpenAI uses default None)
        base_urls = {
            "anthropic": "https://api.anthropic.com/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "nim": "https://integrate.api.nvidia.com/v1",
        }
        return base_urls.get(app.state.provider)
    else:
        # Use local base URL for local models
        local_url = getattr(app.state, "local_base_url", "")
        return local_url.strip() if local_url and local_url.strip() else None


def get_model(app: Any) -> str:
    """Get model name based on configuration mode."""
    if app.state.use_cloud_models:
        return getattr(app.state, "model", DEFAULT_MODEL)
    else:
        local_model = getattr(app.state, "local_model", "")
        return local_model.strip() if local_model and local_model.strip() else "llama3.2:latest"


def get_current_config_summary(app: Any) -> str:
    """Get a summary of current configuration for display."""
    if app.state.use_cloud_models:
        return f"☁️ {app.state.provider}/{app.state.model}"
    else:
        base_display = (
            app.state.local_base_url.replace("http://", "").replace("https://", "")
            if app.state.local_base_url
            else "localhost"
        )
        model_display = app.state.local_model if app.state.local_model else "default"
        return f"🏠 {base_display}/{model_display}"
