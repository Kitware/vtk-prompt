"""
Configuration State Helpers Module.

This module provides helper functions for managing configuration state,
including API keys, base URLs, model selection, and configuration summaries.
"""

from typing import Any

from ..provider_utils import DEFAULT_MODEL


def get_api_key(app: Any) -> str | None:
    """Get API key from state (requires manual input in UI)."""
    api_token = getattr(app.state, "api_token", "")
    return api_token.strip() if api_token and api_token.strip() else None


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
