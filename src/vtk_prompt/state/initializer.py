"""
State Initializer Module.

This module provides functions for initializing application state variables
and setting up the VTK Prompt UI application state.
"""

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
    app.state.is_loading = False
    app.state.use_rag = False
    app.state.error_message = ""
    app.state.input_tokens = 0
    app.state.output_tokens = 0

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

    app.state.api_token = ""

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

        app.prompt_client = VTKPromptClient(
            collection_name="vtk-examples",
            database_path="./db/codesage-codesage-large-v2",
            verbose=False,
            conversation=app.state.conversation,
        )
    except ValueError as e:
        app.state.error_message = str(e)
