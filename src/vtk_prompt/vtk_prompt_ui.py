"""
VTK Prompt Interactive User Interface.

This module provides a web-based interactive user interface for VTK code generation using Trame.
It combines VTK visualization with AI-powered code generation capabilities in a single application.

The interface includes:
- Real-time VTK code generation and execution
- Interactive 3D visualization with VTK render window
- Conversation management with history navigation
- File upload/download for conversation persistence
- Live code editing and execution with error handling
- RAG integration for context-aware code generation

Example:
    >>> vtk-prompt-ui --port 9090
"""

import sys
from pathlib import Path
from typing import Any, Optional

import vtk
import yaml
from trame.app import TrameApp
from trame.decorators import change, controller, trigger
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch  # noqa

from . import get_logger
from .client import VTKPromptClient
from .controllers import configuration, conversation, generation
from .provider_utils import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    get_available_models,
    get_supported_providers,
    supports_temperature,
)
from .rendering import (
    add_default_scene,
    setup_vtk_renderer,
)
from .ui.layout import build_content, build_drawer, build_toolbar

logger = get_logger(__name__)

EXPLAIN_RENDERER = (
    "# renderer is a vtkRenderer injected by this webapp"
    + "\n"
    + "# Use your own vtkRenderer in your application"
)
EXPLANATION_PATTERN = r"<explanation>(.*?)</explanation>"
CODE_PATTERN = r"<code>(.*?)</code>"
EXTRA_INSTRUCTIONS_TAG = "</extra_instructions>"


def load_js(server: Any) -> None:
    """Load JavaScript utilities for VTK Prompt UI."""
    js_file = Path(__file__).with_name("utils.js")
    server.enable_module(
        {
            "serve": {"vtk_prompt": str(js_file.parent)},
            "scripts": [f"vtk_prompt/{js_file.name}"],
        }
    )


class VTKPromptApp(TrameApp):
    """VTK Prompt interactive application with 3D visualization and AI chat interface."""

    def __init__(
        self, server: Optional[Any] = None, custom_prompt_file: Optional[str] = None
    ) -> None:
        """Initialize VTK Prompt application.

        Args:
            server: Trame server instance
            custom_prompt_file: Path to custom YAML prompt file
        """
        super().__init__(server=server, client_type="vue3")
        self.state.trame__title = "VTK Prompt"

        # Store custom prompt file path and data
        self.custom_prompt_file = custom_prompt_file
        self.custom_prompt_data = None

        # Add CLI argument for custom prompt file
        self.server.cli.add_argument(
            "--prompt-file",
            help="Path to custom YAML prompt file (overrides built-in prompts and defaults)",
            dest="prompt_file",
        )

        # Make sure JS is loaded
        load_js(self.server)

        # Suppress VTK warnings to reduce console noise
        vtk.vtkObject.GlobalWarningDisplayOff()

        # Initialize VTK components for trame
        self.renderer, self.render_window, self.render_window_interactor = setup_vtk_renderer()
        self._conversation_loading = False
        add_default_scene(self.renderer)

        # Initialize application state
        self._initialize_state()

        # Load custom prompt file after VTK initialization
        if custom_prompt_file:
            self._load_custom_prompt_file()

    def _load_custom_prompt_file(self) -> None:
        """Load custom YAML prompt file and extract model parameters."""
        if not self.custom_prompt_file:
            return

        try:
            custom_file_path = Path(self.custom_prompt_file)
            if not custom_file_path.exists():
                logger.error("Custom prompt file not found: %s", self.custom_prompt_file)
                return

            with open(custom_file_path, "r") as f:
                self.custom_prompt_data = yaml.safe_load(f)

            logger.info("Loaded custom prompt file: %s", custom_file_path.name)

            # Override UI defaults with custom prompt parameters
            if self.custom_prompt_data and isinstance(self.custom_prompt_data, dict):
                model_value = self.custom_prompt_data.get("model", DEFAULT_MODEL)
                if isinstance(model_value, str) and model_value:
                    if "/" in model_value:
                        provider_part, model_part = model_value.split("/", 1)
                        # Validate provider
                        supported = set(get_supported_providers() + ["local"])
                        if provider_part not in supported or not model_part.strip():
                            msg = (
                                "Invalid 'model' in prompt file. Expected '<provider>/<model>' "
                                "with provider in {openai, anthropic, gemini, nim, local}."
                            )
                            self.state.error_message = msg
                            raise ValueError(msg)
                        if provider_part == "local":
                            # Switch to local mode
                            self.state.use_cloud_models = False
                            self.state.tab_index = 1
                            self.state.local_model = model_part
                        else:
                            # Cloud provider/model
                            self.state.use_cloud_models = True
                            self.state.tab_index = 0
                            self.state.provider = provider_part
                            self.state.model = model_part
                    else:
                        # Enforce explicit provider/model format
                        msg = (
                            "Invalid 'model' format in prompt file. Expected '<provider>/<model>' "
                            "(e.g., 'openai/gpt-5' or 'local/llama3')."
                        )
                        self.state.error_message = msg
                        raise ValueError(msg)

                # RAG and generation controls
                if "rag" in self.custom_prompt_data:
                    self.state.use_rag = bool(self.custom_prompt_data.get("rag"))
                if "top_k" in self.custom_prompt_data:
                    _top_k = self.custom_prompt_data.get("top_k")
                    if isinstance(_top_k, int):
                        self.state.top_k = _top_k
                    elif isinstance(_top_k, str) and _top_k.isdigit():
                        self.state.top_k = int(_top_k)
                    else:
                        logger.warning("Invalid top_k in prompt file: %r; keeping existing", _top_k)
                if "retries" in self.custom_prompt_data:
                    _retries = self.custom_prompt_data.get("retries")
                    if isinstance(_retries, int):
                        self.state.retry_attempts = _retries
                    elif isinstance(_retries, str) and _retries.isdigit():
                        self.state.retry_attempts = int(_retries)
                    else:
                        logger.warning(
                            "Invalid retries in prompt file: %r; keeping existing", _retries
                        )

                self.state.temperature_supported = supports_temperature(model_part)
                # Set model parameters from prompt file
                model_params = self.custom_prompt_data.get("modelParameters", {})
                if isinstance(model_params, dict):
                    if "temperature" in model_params:
                        if not self.state.temperature_supported:
                            self.state.temperature = 1.0  # enforce
                            logger.warning(
                                "Temperature not supported for model %s; forcing 1.0", model_part
                            )
                        else:
                            self.state.temperature = model_params["temperature"]
                    if "max_tokens" in model_params:
                        self.state.max_tokens = model_params["max_tokens"]
        except (yaml.YAMLError, ValueError) as e:
            # Log error and surface to UI as well
            logger.error("Failed to load custom prompt file %s: %s", self.custom_prompt_file, e)
            self.state.error_message = str(e)
            self.custom_prompt_data = None

    def _initialize_state(self) -> None:
        """Initialize application state variables."""
        # App state variables
        self.state.query_text = ""
        self.state.generated_code = ""
        self.state.generated_explanation = ""
        self.state.is_loading = False
        self.state.use_rag = False
        self.state.error_message = ""
        self.state.input_tokens = 0
        self.state.output_tokens = 0

        # Conversation state variables
        self._conversation_loading = False
        self.state.conversation_object = None
        self.state.conversation_file = None
        self.state.conversation = None
        self.state.conversation_index = 0
        self.state.conversation_navigation = []
        self.state.can_navigate_left = False
        self.state.can_navigate_right = False
        self.state.is_viewing_history = False

        # Toast notification state
        self.state.toast_message = ""
        self.state.toast_visible = False
        self.state.toast_color = "warning"

        # API configuration state
        self.state.use_cloud_models = True  # Toggle between cloud and local
        self.state.tab_index = 0  # Tab navigation state

        # Cloud model configuration
        self.state.provider = DEFAULT_PROVIDER
        self.state.model = DEFAULT_MODEL
        self.state.temperature_supported = True
        # Initialize with supported providers and fallback models
        self.state.available_providers = get_supported_providers()
        self.state.available_models = get_available_models()

        # Load component defaults and sync UI state
        try:
            from .prompts import assemble_vtk_prompt

            prompt_data = assemble_vtk_prompt("placeholder")  # Just to get defaults
            model_params = prompt_data.get("modelParameters", {})

            # Update state with component model configuration
            if "temperature" in model_params:
                self.state.temperature = str(model_params["temperature"])
            if "max_tokens" in model_params:
                self.state.max_tokens = str(model_params["max_tokens"])

            # Parse default model from component data
            default_model = prompt_data.get("model", f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL}")
            if "/" in default_model:
                provider, model = default_model.split("/", 1)
                self.state.provider = provider
            logger.debug(
                "Loaded component defaults: provider=%s, model=%s, temp=%s, max_tokens=%s",
                self.state.provider,
                self.state.model,
                self.state.temperature,
                self.state.max_tokens,
            )
        except Exception as e:
            logger.warning("Could not load component defaults: %s", e)
            # Fall back to default values
            self.state.temperature = "0.5"
            self.state.max_tokens = "10000"

        self.state.api_token = ""

        # Build UI
        self._build_ui()

        # Initialize the VTK prompt client
        self._init_prompt_client()

    def _init_prompt_client(self) -> None:
        """Initialize the prompt client based on current settings."""
        try:
            # Validate configuration
            validation_error = self._validate_configuration()
            if validation_error:
                self.state.error_message = validation_error
                return

            self.prompt_client = VTKPromptClient(
                collection_name="vtk-examples",
                database_path="./db/codesage-codesage-large-v2",
                verbose=False,
                conversation=self.state.conversation,
            )
        except ValueError as e:
            self.state.error_message = str(e)

    def _get_api_key(self) -> Optional[str]:
        """Get API key from state (requires manual input in UI)."""
        api_token = getattr(self.state, "api_token", "")
        return api_token.strip() if api_token and api_token.strip() else None

    def _get_base_url(self) -> Optional[str]:
        """Get base URL based on configuration mode."""
        if self.state.use_cloud_models:
            # Use predefined base URLs for cloud providers (OpenAI uses default None)
            base_urls = {
                "anthropic": "https://api.anthropic.com/v1",
                "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "nim": "https://integrate.api.nvidia.com/v1",
            }
            return base_urls.get(self.state.provider)
        else:
            # Use local base URL for local models
            local_url = getattr(self.state, "local_base_url", "")
            return local_url.strip() if local_url and local_url.strip() else None

    def _get_model(self) -> str:
        """Get model name based on configuration mode."""
        if self.state.use_cloud_models:
            return getattr(self.state, "model", DEFAULT_MODEL)
        else:
            local_model = getattr(self.state, "local_model", "")
            return local_model.strip() if local_model and local_model.strip() else "llama3.2:latest"

    def _get_current_config_summary(self) -> str:
        """Get a summary of current configuration for display."""
        if self.state.use_cloud_models:
            return f"☁️ {self.state.provider}/{self.state.model}"
        else:
            base_display = (
                self.state.local_base_url.replace("http://", "").replace("https://", "")
                if self.state.local_base_url
                else "localhost"
            )
            model_display = self.state.local_model if self.state.local_model else "default"
            return f"🏠 {base_display}/{model_display}"

    def _validate_configuration(self) -> Optional[str]:
        """Validate current configuration and return error message if invalid."""
        if self.state.use_cloud_models:
            # Validate cloud configuration
            if not hasattr(self.state, "provider") or not self.state.provider:
                return "Provider is required for cloud models"
            if self.state.provider not in self.state.available_providers:
                return f"Invalid provider: {self.state.provider}"
            if not hasattr(self.state, "model") or not self.state.model:
                return "Model is required for cloud models"
            if self.state.provider in self.state.available_models:
                if self.state.model not in self.state.available_models[self.state.provider]:
                    return f"Invalid model {self.state.model} for provider {self.state.provider}"
        else:
            # Validate local configuration
            if not hasattr(self.state, "local_base_url") or not self.state.local_base_url.strip():
                return "Base URL is required for local models"
            if not hasattr(self.state, "local_model") or not self.state.local_model.strip():
                return "Model name is required for local models"

            # Basic URL validation
            base_url = self.state.local_base_url.strip()
            if not (base_url.startswith("http://") or base_url.startswith("https://")):
                return "Base URL must start with http:// or https://"

        return None  # No validation errors

    @change("tab_index")
    def on_tab_change(self, tab_index: int, **_: Any) -> None:
        """Handle tab change to sync use_cloud_models state."""
        configuration.on_tab_change(self, tab_index, **_)

    @change("model", "local_model")
    def _on_model_change(self, **_: Any) -> None:
        """Handle model change to update temperature support."""
        configuration.on_model_change(self, **_)

    @controller.set("generate_code")
    def generate_code(self) -> None:
        """Generate VTK code from user query."""
        generation.generate_code(self)

    @controller.set("clear_scene")
    def clear_scene(self) -> None:
        """Clear the VTK scene and restore default axes."""
        generation.clear_scene(self)

    @controller.set("reset_camera")
    def reset_camera(self) -> None:
        """Reset camera view."""
        generation.reset_camera(self)

    @controller.set("trigger_warning_toast")
    def trigger_warning_toast(self, message: str) -> None:
        """Display a warning toast notification.

        Args:
            message: Warning message to display
        """
        generation.trigger_warning_toast(self, message)

    def _generate_and_execute_code(self) -> None:
        """Generate VTK code using AI API and execute it."""
        generation.generate_and_execute_code(self)

    def _execute_with_renderer(self, code_string: str) -> None:
        """Execute VTK code with our renderer."""
        generation.execute_with_renderer(self, code_string)

    @change("conversation_object")
    def on_conversation_file_data_change(
        self, conversation_object: Optional[dict[str, Any]], **_: Any
    ) -> None:
        """Handle conversation file data changes and load conversation history."""
        conversation.on_conversation_file_data_change(self, conversation_object, **_)

    def _build_conversation_navigation(self) -> None:
        """Build list of conversation pairs (user message + assistant response) for navigation."""
        conversation.build_conversation_navigation(self)

    def _sync_with_prompt_client(self) -> None:
        """Sync conversation navigation with prompt client conversation."""
        conversation.sync_with_prompt_client(self)

    def _process_conversation_pair(self, pair_index: Optional[int] = None) -> None:
        """Process a specific conversation pair by index."""
        from .controllers.conversation import _process_conversation_pair

        _process_conversation_pair(self, pair_index)

    def _process_loaded_conversation(self) -> None:
        """Process loaded conversation file."""
        from .controllers.conversation import _process_loaded_conversation

        _process_loaded_conversation(self)

    @controller.set("navigate_conversation_left")
    def navigate_conversation_left(self) -> None:
        """Navigate to previous conversation pair."""
        conversation.navigate_conversation_left(self)

    @controller.set("navigate_conversation_right")
    def navigate_conversation_right(self) -> None:
        """Navigate to next conversation pair."""
        conversation.navigate_conversation_right(self)

    @trigger("save_conversation")
    def save_conversation(self) -> str:
        """Save current conversation history as JSON string."""
        return conversation.save_conversation(self)

    @trigger("save_config")
    def save_config(self) -> str:
        """Save current configuration as YAML string for download."""
        return configuration.save_config(self)

    @change("provider")
    def _on_provider_change(self, provider, **kwargs) -> None:
        """Handle provider selection change."""
        configuration.on_provider_change(self, provider, **kwargs)

    def _build_ui(self) -> None:
        """Build a simplified Vuetify UI."""
        # Initialize drawer state as collapsed
        self.state.main_drawer = False

        with SinglePageWithDrawerLayout(
            self.server, theme=("theme_mode", "light"), style="max-height: 100vh;"
        ) as layout:
            layout.title.set_text("VTK Prompt UI")

            # Build UI sections using layout modules
            build_toolbar(layout)
            build_drawer(layout)
            build_content(layout, self)

    def start(self) -> None:
        """Start the trame server."""
        self.server.start()


def main() -> None:
    """Start the trame app."""
    print("VTK Prompt UI - Enter your API token in the application settings.")
    print("Supported providers: OpenAI, Anthropic, Google Gemini, NVIDIA NIM")
    print("For local Ollama, use custom base URL and model configuration.")

    # Check for custom prompt file in CLI arguments
    custom_prompt_file = None

    # Extract --prompt-file before Trame processes args
    for i, arg in enumerate(sys.argv):
        if arg == "--prompt-file" and i + 1 < len(sys.argv):
            custom_prompt_file = sys.argv[i + 1]
            break

    # Create and start the app
    app = VTKPromptApp(custom_prompt_file=custom_prompt_file)
    app.start()


if __name__ == "__main__":
    main()
