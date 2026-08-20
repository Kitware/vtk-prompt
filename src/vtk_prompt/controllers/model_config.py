"""Per-conversation model configuration.

Each conversation remembers which model it uses (cloud/local, provider, model
name, endpoint, key). The active selection is snapshotted onto the session when
switching away and applied when switching in. A new conversation inherits the
last-used selection simply by copying whatever is currently active, so "new
conversations default to the last used model" needs no extra bookkeeping.
"""

from typing import Any

# Which model a conversation talks to: the endpoint and credentials.
MODEL_IDENTITY_FIELDS = (
    "use_cloud_models",
    "provider",
    "model",
    "local_base_url",
    "local_model",
    "api_token",
    "temperature_supported",
)

# How that model is driven: the "Generation" section of the settings dialog.
# These belong to the conversation for the same reason the model does - a
# conversation tuned for terse deterministic output shouldn't have its
# temperature moved by work done in another tab.
GENERATION_PARAM_FIELDS = (
    "temperature",
    "max_tokens",
    "retry_attempts",
    # Travels with the conversation so a clamped temperature can be restored in
    # the conversation it was clamped in, not whichever one is open later.
    "temperature_pref",
)

# The full per-conversation cluster, in apply order. Identity must come first:
# writing `model` fires the @change("model") hook, which clamps temperature for
# models that don't support it, so the params have to settle afterwards.
#
# Deliberately global (not per-conversation): mcp_url, top_k, log_tool_calls,
# agentic_retrieval, data_root, uploaded_files.
MODEL_CONFIG_FIELDS = MODEL_IDENTITY_FIELDS + GENERATION_PARAM_FIELDS


def snapshot_model_config(app: Any) -> dict:
    """Read the active model-config cluster off live state into a plain dict."""
    return {f: getattr(app.state, f, None) for f in MODEL_CONFIG_FIELDS}


def apply_model_config(app: Any, cfg: dict) -> None:
    """Write a saved model-config cluster back onto live state.

    Missing keys are left untouched so an older session file (saved before this
    feature) keeps whatever is currently active rather than blanking the model.
    An empty ``api_token`` is treated the same way: a conversation saved without
    a token should fall back to the environment-seeded one rather than clearing
    it for everything opened afterwards.
    """
    if not cfg:
        return
    for f in MODEL_CONFIG_FIELDS:
        if f not in cfg or cfg[f] is None:
            continue
        if f == "api_token" and not str(cfg[f]).strip():
            continue
        setattr(app.state, f, cfg[f])


def build_model_options(app: Any) -> list:
    """Flat, display-ready list of selectable models for the toolbar picker.

    Each entry is a plain dict with precomputed fields so the Vue template binds
    simple values (no nested v-for or inline expressions, which have historically
    produced malformed markup):
        {"key", "label", "provider", "model", "cloud"}
    Cloud models come from the curated per-provider lists; a single "local"
    entry represents the configured local endpoint.
    """
    opts: list = []
    available = getattr(app.state, "available_models", {}) or {}
    for provider in sorted(available.keys()):
        for model in available[provider]:
            opts.append(
                {
                    "key": f"cloud:{provider}:{model}",
                    "label": f"{provider} / {model}",
                    "provider": provider,
                    "model": model,
                    "cloud": True,
                }
            )
    local_model = getattr(app.state, "local_model", "") or "default"
    opts.append(
        {
            "key": "local",
            "label": f"Local: {local_model}",
            "provider": "local",
            "model": local_model,
            "cloud": False,
        }
    )
    return opts


def select_model_option(app: Any, key: str) -> None:
    """Apply a picker choice (by its precomputed key) to the active conversation."""
    if key == "local":
        app.state.use_cloud_models = False
        # Keep the settings dialog's Cloud/Local tab in step with the picker.
        app.state.tab_index = 1
    else:
        # key form: "cloud:<provider>:<model>"
        parts = key.split(":", 2)
        if len(parts) != 3:
            return
        _, provider, model = parts
        app.state.use_cloud_models = True
        app.state.provider = provider
        app.state.model = model
        app.state.tab_index = 0

    # Recompute temperature support through the same path the settings dialog
    # uses, so the picker and the dialog can't disagree. This also covers the
    # local branch, where no @change("model") hook fires.
    from . import configuration

    configuration.on_model_change(app)

    # Persist immediately so the choice sticks to this conversation.
    from . import sessions as sessions_mod

    sessions_mod.capture_current_session(app)
