"""
Configuration Controllers Module.

This module provides controller functions for handling configuration changes,
provider/model selection, and settings management in the VTK Prompt UI.
"""

from typing import Any

import yaml

from .. import get_logger
from ..provider_utils import DEFAULT_PROVIDER, get_default_model, supports_temperature

logger = get_logger(__name__)

# Settings that belong to the install rather than to a conversation, and so are
# written back to the config file. Everything else in that file is left alone:
# `model` and `base_url` follow the active conversation now, and `retries` /
# `modelParameters` are per-conversation too, so writing them here would let
# whichever conversation happened to be open redefine the startup defaults.
_GLOBAL_SETTING_KEYS = (
    "mcp_url",
    "top_k",
    "log_tool_calls",
    "agentic_retrieval",
    "data_root",
)


def _global_settings_path():
    """Return the config file to write global settings to.

    Prefers whichever file was discovered at startup so edits land back in the
    file they came from, and falls back to the per-user location on a fresh
    install where no config exists yet.
    """
    from pathlib import Path

    from ..utils.env_config import _config_home, discover_config_file

    found = discover_config_file()
    if found:
        return Path(found)
    return _config_home() / "config.yml"


def persist_global_settings(app: Any) -> None:
    """Write the install-wide settings back to the config file.

    Read-modify-write rather than a full dump: the file is hand-maintained and
    holds keys this app never sets (``name``, ``description``), so only the keys
    in ``_GLOBAL_SETTING_KEYS`` are touched and existing order is preserved.
    """
    path = _global_settings_path()
    try:
        existing = {}
        if path.is_file():
            loaded = yaml.safe_load(path.read_text())
            if isinstance(loaded, dict):
                existing = loaded

        updated = dict(existing)
        for key in _GLOBAL_SETTING_KEYS:
            if not hasattr(app.state, key):
                continue
            value = getattr(app.state, key)
            if key == "top_k":
                value = int(value)
            elif key in ("log_tool_calls", "agentic_retrieval"):
                value = bool(value)
            else:
                value = (value or "").strip()
                if not value and key not in existing:
                    continue  # don't add empty keys the file never had
            updated[key] = value

        if updated == existing:
            return  # nothing changed; leave the file's mtime alone

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(updated, sort_keys=False))
        logger.info("Saved global settings to %s", path)
    except (OSError, yaml.YAMLError, TypeError, ValueError) as e:
        # Never let a settings write break closing the dialog.
        logger.warning("Could not save global settings to %s: %s", path, e)


def on_tab_change(app: Any, tab_index: int, **_: Any) -> None:
    """Handle tab change to sync use_cloud_models state."""
    app.state.use_cloud_models = tab_index == 0


def on_model_change(app: Any, **_: Any) -> None:
    """Sync temperature support for the selected model.

    Models that ignore temperature need it pinned to 1, but that used to
    overwrite the user's value with no way back: switching to o3 and back left
    the conversation at 1 forever. The pre-clamp value is stashed instead, and
    restored on the way out.
    """
    current_model = app._get_model()
    supported = supports_temperature(current_model)
    app.state.temperature_supported = supported

    if not supported:
        # Stash only on the way in, or a second model change while clamped
        # would overwrite the stash with the clamped value itself.
        if not str(getattr(app.state, "temperature_pref", "") or ""):
            app.state.temperature_pref = str(app.state.temperature)
        app.state.temperature = 1
        return

    stashed = str(getattr(app.state, "temperature_pref", "") or "")
    if stashed:
        app.state.temperature = stashed
        app.state.temperature_pref = ""


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
    mcp_url = getattr(app.state, "mcp_url", "").strip()
    data_root = getattr(app.state, "data_root", "").strip()
    top_k = int(getattr(app.state, "top_k", 5))
    log_tool_calls = bool(getattr(app.state, "log_tool_calls", False))
    agentic_retrieval = bool(getattr(app.state, "agentic_retrieval", False))
    base_url = getattr(app.state, "local_base_url", "").strip() if not use_cloud else ""

    content = {
        "name": "Custom VTK Prompt config file",
        "description": f"Exported from UI - {'Cloud' if use_cloud else 'Local'} configuration",
        "model": provider_model,
        "base_url": base_url,
        "mcp_url": mcp_url,
        "data_root": data_root,
        "top_k": top_k,
        "log_tool_calls": log_tool_calls,
        "agentic_retrieval": agentic_retrieval,
        "retries": retries,
        "modelParameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }
    return yaml.safe_dump(content, sort_keys=False)
