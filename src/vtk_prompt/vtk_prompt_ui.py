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
from typing import Any, Optional

import vtk
from trame.app import TrameApp
from trame.decorators import change, controller, trigger
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch  # noqa

from . import get_logger
from .controllers import configuration, conversation, generation
from .rendering import (
    add_default_scene,
    setup_vtk_renderer,
)
from .state import config_state, config_validator, initializer
from .ui.layout import build_content, build_drawer, build_toolbar
from .utils import file_handlers, prompt_loader

logger = get_logger(__name__)


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
        file_handlers.load_js(self.server)

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
        prompt_loader.load_custom_prompt_file(self)

    def _initialize_state(self) -> None:
        """Initialize application state variables."""
        initializer.initialize_state(self)

    def _init_prompt_client(self) -> None:
        """Initialize the prompt client based on current settings."""
        initializer.init_prompt_client(self)

    def _get_api_key(self) -> Optional[str]:
        """Get API key from state (requires manual input in UI)."""
        return config_state.get_api_key(self)

    def _get_base_url(self) -> Optional[str]:
        """Get base URL based on configuration mode."""
        return config_state.get_base_url(self)

    def _get_model(self) -> str:
        """Get model name based on configuration mode."""
        return config_state.get_model(self)

    def _get_current_config_summary(self) -> str:
        """Get a summary of current configuration for display."""
        return config_state.get_current_config_summary(self)

    def _validate_configuration(self) -> Optional[str]:
        """Validate current configuration and return error message if invalid."""
        return config_validator.validate_configuration(self)

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
