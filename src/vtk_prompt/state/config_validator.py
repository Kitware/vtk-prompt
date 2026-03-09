"""
Configuration Validator Module.

This module provides functions for validating VTK Prompt UI configuration
and checking that all required settings are properly configured.
"""

from typing import Any


def validate_configuration(app: Any) -> str | None:
    """Validate current configuration and return error message if invalid."""
    if app.state.use_cloud_models:
        # Validate cloud configuration
        if not hasattr(app.state, "provider") or not app.state.provider:
            return "Provider is required for cloud models"
        if app.state.provider not in app.state.available_providers:
            return f"Invalid provider: {app.state.provider}"
        if not hasattr(app.state, "model") or not app.state.model:
            return "Model is required for cloud models"
        if app.state.provider in app.state.available_models:
            if app.state.model not in app.state.available_models[app.state.provider]:
                return f"Invalid model {app.state.model} for provider {app.state.provider}"
    else:
        # Validate local configuration
        if not hasattr(app.state, "local_base_url") or not app.state.local_base_url.strip():
            return "Base URL is required for local models"
        if not hasattr(app.state, "local_model") or not app.state.local_model.strip():
            return "Model name is required for local models"

        # Basic URL validation
        base_url = app.state.local_base_url.strip()
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            return "Base URL must start with http:// or https://"

    return None  # No validation errors
