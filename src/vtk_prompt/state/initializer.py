"""
State Initializer Module.

This module provides functions for initializing application state variables
and setting up the VTK Prompt UI application state.
"""

import os
from typing import Any

from .. import get_logger
from ..client import VTKPromptClient
from ..provider_utils import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    get_available_models,
    get_supported_providers,
)

logger = get_logger(__name__)


def initialize_state(app: Any) -> None:
    """Initialize application state variables."""
    # App state variables
    app.state.query_text = ""
    app.state.generated_code = ""
    app.state.generated_explanation = ""
    # Version history for the editable code panel (undo/redo across generations,
    # runs, and manual edits). code_history_pos indexes the active snapshot.
    app.state.code_history = []
    app.state.code_history_pos = -1
    app.state.is_loading = False
    app.state.mcp_url = ""
    app.state.error_message = ""
    app.state.input_tokens = 0
    app.state.output_tokens = 0
    app.state.advanced_settings_open = False
    app.state.active_settings_tab = "files"

    # File upload state variables
    app.state.uploaded_files = None

    # Conversation state variables
    app._conversation_loading = False
    app.state.conversation_object = None
    app.state.conversation_file = None
    app.state.conversation = None
    app.state.conversation_index = 0
    app.state.conversation_navigation = []
    app.state.can_navigate_left = False
    app.state.can_navigate_right = False
    app.state.is_viewing_history = False
    app.state.history_sort_order = "newest"  # "newest" or "oldest"
    app.state.favorited_conversations = []  # indices into conversation_navigation
    app.state.history_filter_mode = "all"  # "all" or "favorites"

    # Prompt file state variables
    app.state.prompt_object = None
    app.state.prompt_file = None

    # Toast notification state
    app.state.toast_message = ""
    app.state.toast_visible = False
    app.state.toast_color = "warning"

    # API configuration state
    app.state.use_cloud_models = True  # Toggle between cloud and local
    app.state.tab_index = 0  # Tab navigation state

    # Cloud model configuration
    app.state.provider = DEFAULT_PROVIDER
    app.state.model = DEFAULT_MODEL
    app.state.temperature_supported = True
    # Initialize with supported providers and fallback models
    app.state.available_providers = get_supported_providers()
    app.state.available_models = get_available_models()

    # Load component defaults and sync UI state
    _load_component_defaults(app)

    # Seed the API token from the environment (e.g. OPENAI_API_KEY from a .env)
    # so the token field is populated without manual entry. Without this the
    # "Generate Code" button stays hidden (it is gated on api_token), which is a
    # dead end when the key is supplied via environment rather than typed.
    app.state.api_token = os.environ.get("OPENAI_API_KEY", "")

    # Build UI
    app._build_ui()

    # Initialize the VTK prompt client
    init_prompt_client(app)


def _load_component_defaults(app: Any) -> None:
    """Load component defaults and sync UI state."""
    try:
        from ..prompts import assemble_vtk_prompt

        prompt_data = assemble_vtk_prompt("placeholder")  # Just to get defaults
        model_params = prompt_data.get("modelParameters", {})

        # Update state with component model configuration
        if "temperature" in model_params:
            app.state.temperature = str(model_params["temperature"])
        if "max_tokens" in model_params:
            app.state.max_tokens = str(model_params["max_tokens"])

        # Parse default model from component data
        default_model = prompt_data.get("model", f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL}")
        if "/" in default_model:
            provider, model = default_model.split("/", 1)
            app.state.provider = provider
        logger.debug(
            "Loaded component defaults: provider=%s, model=%s, temp=%s, max_tokens=%s",
            app.state.provider,
            app.state.model,
            app.state.temperature,
            app.state.max_tokens,
        )
    except Exception as e:
        logger.warning("Could not load component defaults: %s", e)
        # Fall back to default values
        app.state.temperature = "0.5"
        app.state.max_tokens = "10000"


def init_prompt_client(app: Any) -> None:
    """Initialize the prompt client based on current settings."""
    try:
        # Validate configuration
        from .config_validator import validate_configuration

        validation_error = validate_configuration(app)
        if validation_error:
            app.state.error_message = validation_error
            return

        mcp_url = getattr(app.state, "mcp_url", "").strip() or None
        app.prompt_client = VTKPromptClient(
            verbose=False,
            conversation=app.state.conversation,
            mcp_url=mcp_url,
        )
    except ValueError as e:
        app.state.error_message = str(e)
