#!/usr/bin/env python3

import json
from pathlib import Path

# Add VTK and Trame imports
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch  # noqa
from trame.app import TrameApp
from trame.decorators import change, trigger
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify
from trame_vtk.widgets import vtk as vtk_widgets
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
import vtk

# Import our prompt functionality
from .prompt import VTKPromptClient

# Legacy prompts removed - using YAML system exclusively
from .yaml_prompt_loader import GitHubModelYAMLLoader

EXPLAIN_RENDERER = (
    "# renderer is a vtkRenderer injected by this webapp"
    + "\n"
    + "# Use your own vtkRenderer in your application"
)


def load_js(server):
    js_file = Path(__file__).with_name("utils.js")
    server.enable_module(
        dict(
            serve={"vtk_prompt": str(js_file.parent)},
            scripts=[f"vtk_prompt/{js_file.name}"],
        )
    )


class VTKPromptApp(TrameApp):
    def __init__(self, server=None):
        super().__init__(server=server, client_type="vue3")

        # Make sure JS is loaded
        load_js(self.server)

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

        # Add a simple coordinate axes as default content
        self._add_default_scene()

        # Initial render
        self.render_window.Render()

    def _add_default_scene(self):
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
            print(f"Warning: Could not add default scene: {e}")

        # App state variables
        self.state.query_text = ""
        self.state.generated_code = ""
        self.state.generated_explanation = ""
        self.state.is_loading = False
        self.state.use_rag = False
        self.state.error_message = ""
        self.state.conversation_object = None
        self.state.conversation_file = None
        self.state.conversation = None

        # YAML prompt configuration - UI always uses ui_context prompt
        from pathlib import Path

        prompts_dir = Path(__file__).parent / "prompts"
        yaml_loader = GitHubModelYAMLLoader(prompts_dir)

        # Get default parameters from YAML ui_context prompt
        self.default_params = yaml_loader.get_model_parameters("ui_context")

        # Token usage tracking
        self.state.input_tokens = 0
        self.state.output_tokens = 0

        # API configuration state
        self.state.use_cloud_models = True  # Toggle between cloud and local
        self.state.tab_index = 0  # Tab navigation state

        # Cloud model configuration
        self.state.provider = "openai"
        self.state.model = yaml_loader.get_model_name("ui_context")
        self.state.available_providers = [
            "openai",
            "anthropic",
            "gemini",
            "nim",
        ]
        self.state.available_models = {
            "openai": ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"],
            "anthropic": [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
            ],
            "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
            "nim": [
                "meta/llama3-70b-instruct",
                "meta/llama3-8b-instruct",
                "microsoft/phi-3-medium-4k-instruct",
                "nvidia/llama3-chatqa-1.5-70b",
            ],
        }

        self.state.api_token = "ollama"

        # Build UI
        self._build_ui()

        # Initialize the VTK prompt client
        self._init_prompt_client()

    def _init_prompt_client(self):
        """Initialize the prompt client based on current settings."""
        try:
            # Validate configuration
            validation_error = self._validate_configuration()
            if validation_error:
                self.state.error_message = validation_error
                return

            _ = self._get_base_url()
            api_key = self._get_api_key()

            # For cloud models, API key is usually required
            if self.state.use_cloud_models and not api_key:
                self.state.error_message = (
                    f"API token is required for {self.state.provider}. "
                    "Please enter your API token."
                )
                return

            self.prompt_client = VTKPromptClient(
                collection_name="vtk-examples",
                database_path="./db/codesage-codesage-large-v2",
                verbose=False,
                conversation=self.state.conversation,
            )
        except ValueError as e:
            self.state.error_message = str(e)

    def _get_api_key(self):
        """Get API key from state (requires manual input in UI)."""
        api_token = getattr(self.state, "api_token", "")
        return api_token.strip() if api_token and api_token.strip() else None

    def _get_base_url(self):
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

    def _get_model(self):
        """Get model name based on configuration mode."""
        if self.state.use_cloud_models:
            return getattr(self.state, "model", "gpt-4o")
        else:
            local_model = getattr(self.state, "local_model", "")
            return (
                local_model.strip()
                if local_model and local_model.strip()
                else "llama3.2:latest"
            )

    def _get_current_config_summary(self):
        """Get a summary of current configuration for display."""
        if self.state.use_cloud_models:
            return f"☁️ {self.state.provider}/{self.state.model}"
        else:
            base_display = (
                self.state.local_base_url.replace("http://", "").replace("https://", "")
                if self.state.local_base_url
                else "localhost"
            )
            model_display = (
                self.state.local_model if self.state.local_model else "default"
            )
            return f"🏠 {base_display}/{model_display}"

    def _validate_configuration(self):
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
                if (
                    self.state.model
                    not in self.state.available_models[self.state.provider]
                ):
                    return f"Invalid model {self.state.model} for provider {self.state.provider}"
        else:
            # Validate local configuration
            if (
                not hasattr(self.state, "local_base_url")
                or not self.state.local_base_url.strip()
            ):
                return "Base URL is required for local models"
            if (
                not hasattr(self.state, "local_model")
                or not self.state.local_model.strip()
            ):
                return "Model name is required for local models"

            # Basic URL validation
            base_url = self.state.local_base_url.strip()
            if not (base_url.startswith("http://") or base_url.startswith("https://")):
                return "Base URL must start with http:// or https://"

        return None  # No validation errors

    def on_tab_change(self, tab_index):
        """Handle tab change to sync use_cloud_models state."""
        self.state.tab_index = tab_index
        self.state.use_cloud_models = tab_index == 0

    def generate_code(self):
        """Generate VTK code from user query."""
        self._generate_and_execute_code()

    def clear_scene(self):
        """Clear the VTK scene and restore default axes."""
        try:
            self.renderer.RemoveAllViewProps()
            self._add_default_scene()
            self.renderer.ResetCamera()
            self.render_window.Render()
            self.ctrl.view_update()
        except Exception as e:
            print(f"Error clearing scene: {e}")

    def reset_camera(self):
        """Reset camera view."""
        try:
            self.renderer.ResetCamera()
            self.render_window.Render()
            self.ctrl.view_update()
        except Exception as e:
            print(f"Error resetting camera: {e}")

    def _generate_and_execute_code(self):
        """Generate VTK code using Anthropic API and execute it."""
        self.state.is_loading = True
        self.state.error_message = ""

        try:
            # Reinitialize client with current settings
            self._init_prompt_client()
            if hasattr(self.state, "error_message") and self.state.error_message:
                return

            # Use YAML system exclusively - UI uses ui_context prompt
            result = self.prompt_client.query_yaml(
                self.state.query_text,
                api_key=self._get_api_key(),
                prompt_name="ui_context",
                base_url=self._get_base_url(),
                rag=self.state.use_rag,
                top_k=int(self.state.top_k),
                retry_attempts=int(self.state.retry_attempts),
                # Override parameters from UI settings when different from defaults
                override_temperature=(
                    float(self.state.temperature)
                    if float(self.state.temperature)
                    != self.default_params.get("temperature", 0.1)
                    else None
                ),
                override_max_tokens=(
                    int(self.state.max_tokens)
                    if int(self.state.max_tokens)
                    != self.default_params.get("max_tokens", 1000)
                    else None
                ),
            )
            # Keep UI in sync with conversation
            self.state.conversation = self.prompt_client.conversation

            # Handle generated code
            generated_code, generated_explanation = result
            # Reset token counts for YAML system (no usage info yet)
            self.state.input_tokens = 0
            self.state.output_tokens = 0

            self.state.generated_explanation = generated_explanation.strip()
            self.state.generated_code = EXPLAIN_RENDERER + "\n" + generated_code.strip()

            # Execute the generated code using the existing run_code method
            # But we need to modify it to work with our renderer
            self._execute_with_renderer(generated_code)

        except ValueError as e:
            if "max_tokens" in str(e):
                self.state.error_message = f"{str(e)} Current: {self.state.max_tokens}. Try increasing max tokens."
            else:
                self.state.error_message = f"Error generating code: {str(e)}"
        except Exception as e:
            self.state.error_message = f"Error generating code: {str(e)}"
        finally:
            self.state.is_loading = False

    def _execute_with_renderer(self, code_string):
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
                print(f"Render error: {render_error}")
                # Still update the view even if render fails
                self.ctrl.view_update()

        except Exception as e:
            self.state.error_message = f"Error executing code: {str(e)}"

    @change("conversation_object")
    def on_conversation_file_data_change(self, conversation_object, **_):
        invalid = (
            conversation_object is None
            or conversation_object["type"] != "application/json"
            or Path(conversation_object["name"]).suffix != ".json"
            or not conversation_object["content"]
        )
        self.state.conversation = (
            None if invalid else json.loads(conversation_object["content"])
        )
        self.state.conversation_file = None if invalid else conversation_object["name"]
        if not invalid and self.state.auto_run_conversation_file:
            self.generate_code()

    @trigger("save_conversation")
    def save_conversation(self):
        if self.prompt_client is None:
            return ""
        return json.dumps(self.prompt_client.conversation, indent=2)

    def _build_ui(self):
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
                    html.Span(
                        "Tokens: In: {{ input_tokens }} | Out: {{ output_tokens }}"
                    )

                # VTK control buttons
                with vuetify.VBtn(
                    click=self.clear_scene,
                    icon=True,
                    v_tooltip_bottom="Clear Scene",
                ):
                    vuetify.VIcon("mdi-reload")
                with vuetify.VBtn(
                    click=self.reset_camera,
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
                                change=(self.on_tab_change, "[$event]"),
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
                                        v_model=("model", "gpt-4o"),
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
                                v_model=(
                                    "temperature",
                                    self.default_params.get("temperature", 0.1),
                                ),
                                min=0.0,
                                max=1.0,
                                step=0.1,
                                thumb_label="always",
                                color="orange",
                                prepend_icon="mdi-thermometer",
                                classes="mt-2",
                            )
                            vuetify.VTextField(
                                label="Max Tokens",
                                v_model=(
                                    "max_tokens",
                                    self.default_params.get("max_tokens", 1000),
                                ),
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
                        vuetify.VCardTitle(
                            "⚙️ Files", hide_details=True, density="compact"
                        )
                        with vuetify.VCardText():
                            vuetify.VCheckbox(
                                label="Run new conversation files",
                                v_model=("auto_run_conversation_file", True),
                                prepend_icon="mdi-file-refresh-outline",
                                density="compact",
                                color="primary",
                                hide_details=True,
                            )
                            with html.Div(
                                classes="d-flex align-center justify-space-between"
                            ):
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
                                            rules=[
                                                "[utils.vtk_prompt.rules.json_file]"
                                            ],
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
                with vuetify.VContainer(
                    classes="fluid fill-height", style="min-width: 100%;"
                ):
                    with vuetify.VRow(rows=12, classes="fill-height"):
                        # Left column - Generated code view
                        with vuetify.VCol(cols=6, classes="fill-height"):
                            with vuetify.VExpansionPanels(
                                v_model=("explanation_expanded", [0, 1]),
                                classes="fill-height",
                                multiple=True,
                            ):
                                with vuetify.VExpansionPanel(
                                    classes="mt-1",
                                    style="height: fit-content; max-height: 30%;",
                                ):
                                    vuetify.VExpansionPanelTitle(
                                        "Explanation", classes="text-h6"
                                    )
                                    with vuetify.VExpansionPanelText(
                                        style="overflow: hidden;"
                                    ):
                                        vuetify.VTextarea(
                                            v_model=("generated_explanation", ""),
                                            readonly=True,
                                            solo=True,
                                            hide_details=True,
                                            no_resize=True,
                                            classes="overflow-y-auto fill-height",
                                            placeholder="Explanation will appear here...",
                                        )
                                with vuetify.VExpansionPanel(
                                    classes="mt-1 fill-height",
                                    readonly=True,
                                    style=(
                                        "explanation_expanded.length > 1 ? 'max-height: 75%;' : 'max-height: 95%;'",
                                    ),
                                ):
                                    vuetify.VExpansionPanelTitle(
                                        "Generated Code",
                                        collapse_icon=False,
                                        classes="text-h6",
                                    )
                                    with vuetify.VExpansionPanelText(
                                        style="overflow: hidden; height: 90%;"
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
                        with vuetify.VCol(cols=6, classes="fill-height"):
                            with vuetify.VRow(no_gutters=True, classes="fill-height"):
                                # Top: VTK render view
                                with vuetify.VCol(
                                    cols=12, classes="mb-2", style="height: 70%;"
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
                                                        [
                                                            "vtkInteractorStyleTrackballCamera"
                                                        ],
                                                    ),
                                                ],
                                            )
                                            self.ctrl.view_update = view.update
                                            self.ctrl.view_reset_camera = (
                                                view.reset_camera
                                            )

                                            # Register custom controller methods
                                            self.ctrl.on_tab_change = self.on_tab_change

                                            # Ensure initial render
                                            view.update()

                                # Bottom: Prompt input
                                with vuetify.VCol(cols=12, style="height: 30%;"):
                                    with vuetify.VCard(classes="fill-height"):
                                        with vuetify.VCardText():
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
                                                    "🏠 {{ local_base_url.replace('http://', '')"
                                                    ".replace('https://', '') }}/{{ local_model }}"
                                                ),
                                                small=True,
                                                color="green",
                                                text_color="white",
                                                label=True,
                                                classes="mb-2",
                                                v_show="!use_cloud_models",
                                            )

                                            # Query input
                                            vuetify.VTextarea(
                                                label="Describe VTK visualization",
                                                v_model=("query_text", ""),
                                                rows=4,
                                                variant="outlined",
                                                placeholder="e.g., Create a red sphere with lighting",
                                            )

                                            # Generate button
                                            vuetify.VBtn(
                                                "Generate Code",
                                                color="primary",
                                                block=True,
                                                loading=("trame__busy", False),
                                                click=self.generate_code,
                                                classes="mb-2",
                                                disabled=("!query_text.trim()",),
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

    def start(self):
        """Start the trame server."""
        self.server.start()


def main():
    """Main entry point for the trame app."""

    print("VTK Prompt UI - Enter your API token in the application settings.")
    print("Supported providers: OpenAI, Anthropic, Google Gemini, NVIDIA NIM")
    print("For local Ollama, use custom base URL and model configuration.")

    # Create and start the app
    app = VTKPromptApp()
    app.start()


if __name__ == "__main__":
    main()
