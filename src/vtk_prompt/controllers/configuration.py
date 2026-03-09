"""
Configuration Controllers Module.

This module provides controller functions for handling configuration changes,
provider/model selection, and settings management in the VTK Prompt UI.
"""

from typing import Any

import yaml

from ..provider_utils import DEFAULT_PROVIDER, get_default_model, supports_temperature


def on_tab_change(app: Any, tab_index: int, **_: Any) -> None:
    """Handle tab change to sync use_cloud_models state."""
    app.state.use_cloud_models = tab_index == 0


def on_model_change(app: Any, **_: Any) -> None:
    """Handle model change to update temperature support."""
    current_model = app._get_model()
    app.state.temperature_supported = supports_temperature(current_model)
    if not app.state.temperature_supported:
        app.state.temperature = 1


def on_provider_change(app: Any, provider: str, **kwargs: Any) -> None:
    """Handle provider selection change."""
    # Set default model for the provider if current model not available
    if provider in app.state.available_models:
        models = app.state.available_models[provider]
        if models and app.state.model not in models:
            app.state.model = get_default_model(provider)


def save_config(app: Any) -> str:
    """Save current configuration as YAML string for download."""
    use_cloud = bool(getattr(app.state, "use_cloud_models", True))
    provider = getattr(app.state, "provider", DEFAULT_PROVIDER)
    model = app._get_model()
    provider_model = f"{provider}/{model}" if use_cloud else f"local/{model}"
    temperature = float(getattr(app.state, "temperature", 0.0))
    max_tokens = int(getattr(app.state, "max_tokens", 1000))
    retries = int(getattr(app.state, "retry_attempts", 1))
    rag_enabled = bool(getattr(app.state, "use_rag", False))
    top_k = int(getattr(app.state, "top_k", 5))

    content = {
        "name": "Custom VTK Prompt config file",
        "description": f"Exported from UI - {'Cloud' if use_cloud else 'Local'} configuration",
        "model": provider_model,
        "rag": rag_enabled,
        "top_k": top_k,
        "retries": retries,
        "modelParameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }
    return yaml.safe_dump(content, sort_keys=False)
