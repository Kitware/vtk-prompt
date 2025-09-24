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

import json
import re
from pathlib import Path
from typing import Any, Optional

import vtk
from trame.app import TrameApp
from trame.decorators import change, controller, trigger
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify
from trame_vtk.widgets import vtk as vtk_widgets
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch  # noqa

from . import get_logger
from .client import VTKPromptClient
from .prompts import get_ui_post_prompt
from .provider_utils import (
    get_available_models,
    get_default_model,
    get_supported_providers,
    supports_temperature,
)

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

    def __init__(self, server: Optional[Any] = None) -> None:
        """Initialize VTK Prompt application."""
        super().__init__(server=server, client_type="vue3")
        self.state.trame__title = "VTK Prompt"

        # Make sure JS is loaded
        load_js(self.server)

        # Suppress VTK warnings to reduce console noise
        vtk.vtkObject.GlobalWarningDisplayOff()

        # Initialize VTK components for trame
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.OffScreenRenderingOn()  # Prevent external window
        self.render_window.SetSize(800, 600)

        # Initialize render window interactor properly
        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetRenderWindow(self.render_window)
        self.render_window_interactor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        # Set a default background and add a simple default scene to prevent segfault
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self._conversation_loading = False
        self._add_default_scene()

        # Initial render
        self.render_window.Render()

    def _add_default_scene(self) -> None:
        """Add default coordinate axes to prevent empty scene segfaults."""
        try:
            # Create simple axes
            axes = vtk.vtkAxesActor()
            axes.SetTotalLength(1, 1, 1)
            axes.SetShaftType(0)  # Line shaft
            axes.SetCylinderRadius(0.02)

            # Add to renderer
            self.renderer.AddActor(axes)

            # Reset camera to show axes
            self.renderer.ResetCamera()
        except Exception as e:
            logger.warning("Could not add default scene: %s", e)

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

        # API configuration state
        self.state.use_cloud_models = True  # Toggle between cloud and local
        self.state.tab_index = 0  # Tab navigation state

        # Cloud model configuration
        self.state.provider = "openai"
        self.state.model = "gpt-5"
        self.state.temperature_supported = True

        # Initialize with supported providers and fallback models
        self.state.available_providers = get_supported_providers()
        self.state.available_models = get_available_models()

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
            return getattr(self.state, "model", "gpt-5")
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
        self.state.use_cloud_models = tab_index == 0

    @change("model", "local_model")
    def _on_model_change(self, **_: Any) -> None:
        """Handle model change to update temperature support."""
        current_model = self._get_model()
        self.state.temperature_supported = supports_temperature(current_model)
        if not self.state.temperature_supported:
            self.state.temperature = 1

    @controller.set("generate_code")
    def generate_code(self) -> None:
        """Generate VTK code from user query."""
        self._generate_and_execute_code()

    @controller.set("clear_scene")
    def clear_scene(self) -> None:
        """Clear the VTK scene and restore default axes."""
        try:
            self.renderer.RemoveAllViewProps()
            self._add_default_scene()
            self.renderer.ResetCamera()
            self.render_window.Render()
            self.ctrl.view_update()
        except Exception as e:
            logger.error("Error clearing scene: %s", e)

    @controller.set("reset_camera")
    def reset_camera(self) -> None:
        """Reset camera view."""
        try:
            self.renderer.ResetCamera()
            self.render_window.Render()
            self.ctrl.view_update()
        except Exception as e:
            logger.error("Error resetting camera: %s", e)

    def _generate_and_execute_code(self) -> None:
        """Generate VTK code using Anthropic API and execute it."""
        self.state.is_loading = True
        self.state.error_message = ""

        try:
            if not self._conversation_loading:
                # Generate code using prompt functionality - reuse existing methods
                enhanced_query = self.state.query_text
                if self.state.query_text:
                    post_prompt = get_ui_post_prompt()
                    enhanced_query = post_prompt + self.state.query_text

                # Reinitialize client with current settings
                self._init_prompt_client()
                if hasattr(self.state, "error_message") and self.state.error_message:
                    return

                result = self.prompt_client.query(
                    enhanced_query,
                    api_key=self._get_api_key(),
                    model=self._get_model(),
                    base_url=self._get_base_url(),
                    max_tokens=int(self.state.max_tokens),
                    temperature=float(self.state.temperature),
                    top_k=int(self.state.top_k),
                    rag=self.state.use_rag,
                    retry_attempts=int(self.state.retry_attempts),
                )
                # Keep UI in sync with conversation
                self.state.conversation = self.prompt_client.conversation

                # Handle both code and usage information
                if isinstance(result, tuple) and len(result) == 3:
                    generated_explanation, generated_code, usage = result
                    if usage:
                        self.state.input_tokens = usage.prompt_tokens
                        self.state.output_tokens = usage.completion_tokens
                else:
                    # Handle string result
                    generated_explanation = str(result)
                    generated_code = ""
                    # Reset token counts if no usage info
                    self.state.input_tokens = 0
                    self.state.output_tokens = 0

                self.state.generated_explanation = generated_explanation
                self.state.generated_code = EXPLAIN_RENDERER + "\n" + generated_code

                # Update navigation after new conversation entry
                self._build_conversation_navigation()

            self._conversation_loading = False
            # Execute the generated code using the existing run_code method
            # But we need to modify it to work with our renderer
            self._execute_with_renderer(self.state.generated_code)
        except ValueError as e:
            if "max_tokens" in str(e):
                self.state.error_message = (
                    f"{str(e)} Current: {self.state.max_tokens}. Try increasing max tokens."
                )
            else:
                self.state.error_message = f"Error generating code: {str(e)}"
        except Exception as e:
            self.state.error_message = f"Error generating code: {str(e)}"
        finally:
            self.state.is_loading = False

    def _execute_with_renderer(self, code_string: str) -> None:
        """Execute VTK code with our renderer using prompt.py's run_code logic."""
        try:
            # Clear previous actors
            self.renderer.RemoveAllViewProps()

            # Use the same code cleaning logic from prompt.py
            pos = code_string.find("import vtk")
            if pos != -1:
                code_string = code_string[pos:]

            # Ensure vtk is imported
            code_segment = code_string
            if "import vtk" not in code_segment:
                code_segment = "import vtk\n" + code_segment

            # Create execution globals with our renderer available
            exec_globals = {
                "vtk": vtk,
                "renderer": self.renderer,
            }

            # Use the pre-initialized interactor
            # No need to create a new one

            exec(code_segment, exec_globals, {})

            # Reset camera and update view safely
            try:
                self.renderer.ResetCamera()
                self.render_window.Render()
                self.ctrl.view_update()
            except Exception as render_error:
                logger.warning("Render error: %s", render_error)
                # Still update the view even if render fails
                self.ctrl.view_update()
        except Exception as e:
            self.state.error_message = f"Error executing code: {str(e)}"

    @change("conversation_object")
    def on_conversation_file_data_change(
        self, conversation_object: Optional[dict[str, Any]], **_: Any
    ) -> None:
        """Handle conversation file data changes and load conversation history."""
        invalid = (
            conversation_object is None
            or conversation_object.get("type") != "application/json"
            or Path(conversation_object.get("name", "")).suffix != ".json"
            or not conversation_object.get("content")
        )

        if not invalid and conversation_object is not None:
            loaded_conversation = json.loads(conversation_object["content"])

            # Append loaded conversation to existing conversation instead of overwriting
            if self.state.conversation is None:
                self.state.conversation = []

            # Extend the existing conversation with the loaded one
            self.state.conversation.extend(loaded_conversation)
            self.state.conversation_file = conversation_object["name"]
            self.prompt_client.update_conversation(
                loaded_conversation, self.state.conversation_file
            )

            self._process_loaded_conversation()
        else:
            self.state.conversation_file = None

        if not invalid and self.state.auto_run_conversation_file:
            self._conversation_loading = True
            self.generate_code()

    def _parse_assistant_content(self, content: str) -> tuple[Optional[str], Optional[str]]:
        """Parse assistant message content for explanation and code."""
        try:
            explanation_match = re.findall(r"<explanation>(.*?)</explanation>", content, re.DOTALL)
            code_match = re.findall(r"<code>(.*?)</code>", content, re.DOTALL)

            if explanation_match and code_match:
                return explanation_match[0].strip(), code_match[0].strip()
            return None, None
        except Exception:
            return None, None

    def _build_conversation_navigation(self) -> None:
        """Build list of conversation pairs (user message + assistant response) for navigation."""
        if not self.state.conversation:
            self.state.conversation_navigation = []
            self.state.conversation_index = 0
            self._update_navigation_state()
            return

        pairs = []
        current_user = None

        for message in self.state.conversation:
            if message.get("role") == "user":
                current_user = message
            elif message.get("role") == "assistant" and current_user:
                pairs.append({"user": current_user, "assistant": message})
                current_user = None

        self.state.conversation_navigation = pairs
        # Reset index to last pair if we have pairs
        if pairs:
            self.state.conversation_index = len(pairs) - 1
        else:
            self.state.conversation_index = 0

        self._update_navigation_state()

    def _update_navigation_state(self) -> None:
        """Update navigation button states based on current position."""
        nav_length = len(self.state.conversation_navigation)

        # Update navigation buttons
        self.state.can_navigate_left = nav_length > 0 and self.state.conversation_index > 0
        self.state.can_navigate_right = (
            nav_length > 0 and self.state.conversation_index < nav_length
        )

        # Update viewing mode - we're viewing history if not at the "new entry" position
        self.state.is_viewing_history = (
            nav_length > 0 and self.state.conversation_index < nav_length
        )

    def _sync_with_prompt_client(self) -> None:
        """Sync conversation navigation with prompt client conversation."""
        if self.prompt_client and self.prompt_client.conversation:
            self.state.conversation = self.prompt_client.conversation
            self._build_conversation_navigation()

    def _process_conversation_pair(self, pair_index: Optional[int] = None) -> None:
        """Process a specific conversation pair by index."""
        if not self.state.conversation_navigation:
            return

        if pair_index is None:
            pair_index = self.state.conversation_index

        if pair_index < 0 or pair_index >= len(self.state.conversation_navigation):
            return

        pair = self.state.conversation_navigation[pair_index]

        # Process assistant message for explanation and code
        assistant_content = pair["assistant"].get("content", "")
        explanation, code = self._parse_assistant_content(assistant_content)

        if explanation and code:
            # Set explanation and code in UI state
            self.state.generated_explanation = explanation
            self.state.generated_code = EXPLAIN_RENDERER + "\n" + code

            # Execute the code to display visualization
            self._execute_with_renderer(code)

        # Process user message for query text
        user_content = pair["user"].get("content", "").strip()
        if "</extra_instructions>" in user_content:
            parts = user_content.split("</extra_instructions>", 1)
            query_text = parts[1].strip() if len(parts) > 1 else user_content
        else:
            query_text = user_content

        self.state.query_text = query_text

    def _process_loaded_conversation(self) -> None:
        """Process loaded conversation file."""
        if not self.state.conversation:
            return

        # Build navigation pairs and process the latest one
        self._build_conversation_navigation()
        self._process_conversation_pair()

    @controller.set("navigate_conversation_left")
    def navigate_conversation_left(self) -> None:
        """Navigate to previous conversation pair."""
        if not self.state.conversation_navigation:
            return

        if self.state.conversation_index >= 0:
            self.state.conversation_index -= 1
            if self.state.conversation_index >= 0:
                self._process_conversation_pair()
            self._update_navigation_state()

    @controller.set("navigate_conversation_right")
    def navigate_conversation_right(self) -> None:
        """Navigate to next conversation pair."""
        if not self.state.conversation_navigation:
            return

        nav_length = len(self.state.conversation_navigation)
        if self.state.conversation_index < nav_length:
            self.state.conversation_index += 1
            if self.state.conversation_index < nav_length:
                # Still viewing history
                self._process_conversation_pair()
            else:
                # Moved to "new entry" mode - clear only query text for new input
                self.state.query_text = ""
            self._update_navigation_state()

    @trigger("save_conversation")
    def save_conversation(self) -> str:
        """Save current conversation history as JSON string."""
        if hasattr(self, "prompt_client") and self.prompt_client is not None:
            return json.dumps(self.prompt_client.conversation, indent=2)
        return ""

    @change("provider")
    def _on_provider_change(self, provider, **kwargs) -> None:
        """Handle provider selection change."""
        # Set default model for the provider if current model not available
        if provider in self.state.available_models:
            models = self.state.available_models[provider]
            if models and self.state.model not in models:
                self.state.model = get_default_model(provider)

    def _build_ui(self) -> None:
        """Build a simplified Vuetify UI."""
        # Initialize drawer state as collapsed
        self.state.main_drawer = True

        with SinglePageWithDrawerLayout(
            self.server, theme=("theme_mode", "light"), style="max-height: 100vh;"
        ) as layout:
            layout.title.set_text("VTK Prompt UI")
            with layout.toolbar:
                vuetify.VSpacer()
                # Token usage display
                with vuetify.VChip(
                    small=True,
                    color="primary",
                    text_color="white",
                    v_show="input_tokens > 0 || output_tokens > 0",
                    classes="mr-2",
                ):
                    html.Span("Tokens: In: {{ input_tokens }} | Out: {{ output_tokens }}")

                # VTK control buttons
                with vuetify.VBtn(
                    click=self.ctrl.clear_scene,
                    icon=True,
                    v_tooltip_bottom="Clear Scene",
                ):
                    vuetify.VIcon("mdi-reload")
                with vuetify.VBtn(
                    click=self.ctrl.reset_camera,
                    icon=True,
                    v_tooltip_bottom="Reset Camera",
                ):
                    vuetify.VIcon("mdi-camera-retake-outline")

                vuetify.VSwitch(
                    v_model=("theme_mode", "light"),
                    hide_details=True,
                    density="compact",
                    label="Dark Mode",
                    classes="mr-2",
                    true_value="dark",
                    false_value="light",
                )

            with layout.drawer as drawer:
                drawer.width = 350

                with vuetify.VContainer():
                    # Tab Navigation - Centered
                    with vuetify.VRow(justify="center"):
                        with vuetify.VCol(cols="auto"):
                            with vuetify.VTabs(
                                v_model=("tab_index", 0),
                                color="primary",
                                slider_color="primary",
                                centered=True,
                                grow=False,
                            ):
                                vuetify.VTab("☁️ Cloud")
                                vuetify.VTab("🏠Local")

                    # Tab Content
                    with vuetify.VTabsWindow(v_model="tab_index"):
                        # Cloud Providers Tab Content
                        with vuetify.VTabsWindowItem():
                            with vuetify.VCard(flat=True, style="mt-2"):
                                with vuetify.VCardText():
                                    # Provider selection
                                    vuetify.VSelect(
                                        label="Provider",
                                        v_model=("provider", "openai"),
                                        items=("available_providers", []),
                                        density="compact",
                                        variant="outlined",
                                        prepend_icon="mdi-cloud",
                                    )

                                    # Model selection
                                    vuetify.VSelect(
                                        label="Model",
                                        v_model=("model", "gpt-5"),
                                        items=("available_models[provider] || []",),
                                        density="compact",
                                        variant="outlined",
                                        prepend_icon="mdi-brain",
                                    )

                                    # API Token
                                    vuetify.VTextField(
                                        label="API Token",
                                        v_model=("api_token", ""),
                                        placeholder="Enter your API token",
                                        type="password",
                                        density="compact",
                                        variant="outlined",
                                        prepend_icon="mdi-key",
                                        hint="Required for cloud providers",
                                        persistent_hint=True,
                                        error=("!api_token", False),
                                    )

                        # Local Models Tab Content
                        with vuetify.VTabsWindowItem():
                            with vuetify.VCard(flat=True, style="mt-2"):
                                with vuetify.VCardText():
                                    vuetify.VTextField(
                                        label="Base URL",
                                        v_model=(
                                            "local_base_url",
                                            "http://localhost:11434/v1",
                                        ),
                                        placeholder="http://localhost:11434/v1",
                                        density="compact",
                                        variant="outlined",
                                        prepend_icon="mdi-server",
                                        hint="Ollama, LM Studio, etc.",
                                        persistent_hint=True,
                                    )

                                    vuetify.VTextField(
                                        label="Model Name",
                                        v_model=("local_model", "devstral"),
                                        placeholder="devstral",
                                        density="compact",
                                        variant="outlined",
                                        prepend_icon="mdi-brain",
                                        hint="Model identifier",
                                        persistent_hint=True,
                                    )

                                    # Optional API Token for local
                                    vuetify.VTextField(
                                        label="API Token (Optional)",
                                        v_model=("api_token", "ollama"),
                                        placeholder="ollama",
                                        type="password",
                                        density="compact",
                                        variant="outlined",
                                        prepend_icon="mdi-key",
                                        hint="Optional for local servers",
                                        persistent_hint=True,
                                    )

                    with vuetify.VCard(classes="mt-2"):
                        vuetify.VCardTitle("⚙️  RAG settings", classes="pb-0")
                        with vuetify.VCardText():
                            vuetify.VCheckbox(
                                v_model=("use_rag", False),
                                label="RAG",
                                prepend_icon="mdi-bookshelf",
                                density="compact",
                            )
                            vuetify.VTextField(
                                label="Top K",
                                v_model=("top_k", 5),
                                type="number",
                                min=1,
                                max=15,
                                density="compact",
                                disabled=("!use_rag",),
                                variant="outlined",
                                prepend_icon="mdi-chart-scatter-plot",
                            )

                    with vuetify.VCard(classes="mt-2"):
                        vuetify.VCardTitle("⚙️ Generation Settings", classes="pb-0")
                        with vuetify.VCardText():
                            vuetify.VSlider(
                                label="Temperature",
                                v_model=("temperature", 0.1),
                                min=0.0,
                                max=1.0,
                                step=0.1,
                                thumb_label="always",
                                color="orange",
                                prepend_icon="mdi-thermometer",
                                classes="mt-2",
                                disabled=("!temperature_supported",),
                            )
                            vuetify.VTextField(
                                label="Max Tokens",
                                v_model=("max_tokens", 1000),
                                type="number",
                                density="compact",
                                variant="outlined",
                                prepend_icon="mdi-format-text",
                            )
                            vuetify.VTextField(
                                label="Retry Attempts",
                                v_model=("retry_attempts", 1),
                                type="number",
                                min=1,
                                max=5,
                                density="compact",
                                variant="outlined",
                                prepend_icon="mdi-repeat",
                            )

                    with vuetify.VCard(classes="mt-2"):
                        vuetify.VCardTitle("⚙️ Files", hide_details=True, density="compact")
                        with vuetify.VCardText():
                            vuetify.VCheckbox(
                                label="Run new conversation files",
                                v_model=("auto_run_conversation_file", True),
                                prepend_icon="mdi-file-refresh-outline",
                                density="compact",
                                color="primary",
                                hide_details=True,
                            )
                            with html.Div(classes="d-flex align-center justify-space-between"):
                                with vuetify.VTooltip(
                                    text=("conversation_file", "No file loaded"),
                                    location="top",
                                    disabled=("!conversation_object",),
                                ):
                                    with vuetify.Template(v_slot_activator="{ props }"):
                                        vuetify.VFileInput(
                                            label="Conversation File",
                                            v_model=("conversation_object", None),
                                            accept=".json",
                                            density="compact",
                                            variant="solo",
                                            prepend_icon="mdi-forum-outline",
                                            hide_details="auto",
                                            classes="py-1 pr-1 mr-1 text-truncate",
                                            open_on_focus=False,
                                            clearable=False,
                                            v_bind="props",
                                            rules=["[utils.vtk_prompt.rules.json_file]"],
                                        )
                                with vuetify.VTooltip(
                                    text="Download conversation file",
                                    location="right",
                                ):
                                    with vuetify.Template(v_slot_activator="{ props }"):
                                        with vuetify.VBtn(
                                            icon=True,
                                            density="comfortable",
                                            color="secondary",
                                            rounded="lg",
                                            v_bind="props",
                                            disabled=("!conversation",),
                                            click="utils.download("
                                            + "`${model}_${new Date().toISOString()}.json`,"
                                            + "trigger('save_conversation'),"
                                            + "'application/json'"
                                            + ")",
                                        ):
                                            vuetify.VIcon("mdi-file-download-outline")

            with layout.content:
                with vuetify.VContainer(classes="fluid fill-height pt-0", style="min-width: 100%;"):
                    with vuetify.VRow(rows=12, classes="fill-height"):
                        # Left column - Generated code view
                        with vuetify.VCol(cols=7, classes="fill-height"):
                            with vuetify.VExpansionPanels(
                                v_model=("explanation_expanded", [0, 1]),
                                classes="fill-height",
                                multiple=True,
                            ):
                                with vuetify.VExpansionPanel(
                                    classes="mt-1 flex-grow-1 flex-shrink-0 d-flex flex-column",
                                    style="max-height: 25%;",
                                ):
                                    vuetify.VExpansionPanelTitle("Explanation", classes="text-h6")
                                    with vuetify.VExpansionPanelText(
                                        classes="fill-height flex-shrink-1",
                                        style="overflow: hidden;",
                                    ):
                                        vuetify.VTextarea(
                                            v_model=("generated_explanation", ""),
                                            readonly=True,
                                            solo=True,
                                            hide_details=True,
                                            no_resize=True,
                                            classes="overflow-y-auto fill-height",
                                            placeholder="Explanation will appear here...",
                                            auto_grow=True,
                                            density="compact",
                                            style="overflow-y: auto;",
                                        )
                                with vuetify.VExpansionPanel(
                                    classes=(
                                        "mt-1 fill-height flex-grow-2 flex-shrink-0"
                                        + "d-flex flex-column"
                                    ),
                                    readonly=True,
                                    style=(
                                        "explanation_expanded.length > 1 ? "
                                        + "'max-height: 75%;' : 'max-height: 95%;'",
                                    ),
                                ):
                                    vuetify.VExpansionPanelTitle(
                                        "Generated Code",
                                        collapse_icon=False,
                                        classes="text-h6",
                                    )
                                    with vuetify.VExpansionPanelText(
                                        style="overflow: hidden; height: 90%;",
                                        classes="flex-grow-1",
                                    ):
                                        vuetify.VTextarea(
                                            v_model=("generated_code", ""),
                                            readonly=True,
                                            solo=True,
                                            hide_details=True,
                                            no_resize=True,
                                            classes="overflow-y-auto fill-height",
                                            style="font-family: monospace;",
                                            placeholder="Generated VTK code will appear here...",
                                        )

                        # Right column - VTK viewer and prompt
                        with vuetify.VCol(cols=5, classes="fill-height"):
                            with vuetify.VRow(no_gutters=True, classes="fill-height"):
                                # Top: VTK render view
                                with vuetify.VCol(
                                    cols=12,
                                    classes="mb-2 flex-grow-1 flex-shrink-0",
                                    style="min-height: calc(100% - 256px);",
                                ):
                                    with vuetify.VCard(classes="fill-height"):
                                        vuetify.VCardTitle("VTK Visualization")
                                        with vuetify.VCardText(style="height: 90%;"):
                                            # VTK render window
                                            view = vtk_widgets.VtkRemoteView(
                                                self.render_window,
                                                ref="view",
                                                classes="w-100 h-100",
                                                interactor_settings=[
                                                    (
                                                        "SetInteractorStyle",
                                                        ["vtkInteractorStyleTrackballCamera"],
                                                    ),
                                                ],
                                            )
                                            self.ctrl.view_update = view.update
                                            self.ctrl.view_reset_camera = view.reset_camera

                                            # Register custom controller methods
                                            self.ctrl.on_tab_change = self.on_tab_change

                                            # Ensure initial render
                                            view.update()

                                # Bottom: Prompt input
                                with vuetify.VCol(
                                    cols=12,
                                    classes="flex-grow-0 flex-shrink-0",
                                    style="height: 256px;",
                                ):
                                    with vuetify.VCard(classes="fill-height"):
                                        with vuetify.VCardText(
                                            classes="d-flex flex-column",
                                            style="height: 100%;",
                                        ):
                                            with html.Div(classes="d-flex"):
                                                # Cloud models chip
                                                vuetify.VChip(
                                                    "☁️ {{ provider }}/{{ model }}",
                                                    small=True,
                                                    color="blue",
                                                    text_color="white",
                                                    label=True,
                                                    classes="mb-2",
                                                    v_show="use_cloud_models",
                                                )
                                                # Local models chip
                                                vuetify.VChip(
                                                    (
                                                        "🏠 "
                                                        "{{ local_base_url.replace('http://', '')"
                                                        ".replace('https://', '') }}/"
                                                        "{{ local_model }}"
                                                    ),
                                                    small=True,
                                                    color="green",
                                                    text_color="white",
                                                    label=True,
                                                    classes="mb-2",
                                                    v_show="!use_cloud_models",
                                                )
                                                vuetify.VSpacer()
                                                # API token warning chip
                                                vuetify.VChip(
                                                    "API token is required for cloud models.",
                                                    small=True,
                                                    color="error",
                                                    text_color="white",
                                                    label=True,
                                                    classes="mb-2",
                                                    v_show="use_cloud_models && !api_token.trim()",
                                                    prepend_icon="mdi-alert",
                                                )

                                            with html.Div(
                                                classes="d-flex mb-2",
                                                style="height: 100%;",
                                            ):
                                                with vuetify.VBtn(
                                                    variant="tonal",
                                                    icon=True,
                                                    rounded="0",
                                                    disabled=("!can_navigate_left",),
                                                    classes="h-auto mr-1",
                                                    click=self.ctrl.navigate_conversation_left,
                                                ):
                                                    vuetify.VIcon("mdi-arrow-left-circle")
                                                # Query input
                                                vuetify.VTextarea(
                                                    label="Describe VTK visualization",
                                                    v_model=("query_text", ""),
                                                    rows=4,
                                                    variant="outlined",
                                                    placeholder=(
                                                        "e.g., Create a red sphere with lighting"
                                                    ),
                                                    hide_details=True,
                                                    no_resize=True,
                                                    disabled=(
                                                        "is_viewing_history",
                                                        False,
                                                    ),
                                                )
                                                with vuetify.VBtn(
                                                    color=(
                                                        "conversation_index ==="
                                                        + " conversation_navigation.length - 1"
                                                        + " ? 'success' : 'default'",
                                                        "default",
                                                    ),
                                                    variant="tonal",
                                                    icon=True,
                                                    rounded="0",
                                                    disabled=("!can_navigate_right",),
                                                    classes="h-auto ml-1",
                                                    click=self.ctrl.navigate_conversation_right,
                                                ):
                                                    vuetify.VIcon(
                                                        "mdi-arrow-right-circle",
                                                        v_show="conversation_index <"
                                                        + " conversation_navigation.length - 1",
                                                    )
                                                    vuetify.VIcon(
                                                        "mdi-message-plus",
                                                        v_show="conversation_index ==="
                                                        + " conversation_navigation.length - 1",
                                                    )

                                            # Generate button
                                            vuetify.VBtn(
                                                "Generate Code",
                                                color="primary",
                                                block=True,
                                                loading=("trame__busy", False),
                                                click=self.ctrl.generate_code,
                                                classes="mb-2",
                                                disabled=(
                                                    "is_viewing_history ||"
                                                    + " !query_text.trim() ||"
                                                    + " (use_cloud_models && !api_token.trim())",
                                                ),
                                            )

            vuetify.VAlert(
                closable=True,
                v_show=("error_message", ""),
                density="compact",
                type="error",
                text=("error_message",),
                classes="h-auto position-absolute bottom-0 align-self-center mb-1",
                style="width: 30%; z-index: 1000;",
                icon="mdi-alert-outline",
            )

    def start(self) -> None:
        """Start the trame server."""
        self.server.start()


def main() -> None:
    """Start the trame app."""
    print("VTK Prompt UI - Enter your API token in the application settings.")
    print("Supported providers: OpenAI, Anthropic, Google Gemini, NVIDIA NIM")
    print("For local Ollama, use custom base URL and model configuration.")

    # Create and start the app
    app = VTKPromptApp()
    app.start()


if __name__ == "__main__":
    main()
