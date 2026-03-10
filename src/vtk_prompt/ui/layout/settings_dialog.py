"""
Settings Dialog Layout Module.

This module provides the advanced settings dialog layout for the VTK Prompt UI.
The dialog contains model configuration, RAG settings, and file controls.
"""

from typing import Any

from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

from ...provider_utils import DEFAULT_MODEL, DEFAULT_PROVIDER


def build_settings_dialog(layout: Any, app: Any) -> None:
    """Build the advanced settings dialog with configuration options."""
    with layout.content:
        with vuetify.VDialog(v_model=("advanced_settings_open", True), classes="w-25"):
            with vuetify.VCard():
                with vuetify.VTabs(v_model=("active_settings_tab", "files"), color="primary"):
                    vuetify.VTab("Files", value="files")
                    vuetify.VTab("Model", value="model")
                    vuetify.VTab("Advanced", value="advanced")
                with vuetify.VTabsWindow(v_model=("active_settings_tab", "files")):
                    # Files Tab
                    with vuetify.VTabsWindowItem(value="files"):
                        with vuetify.VRow(classes="ma-2 justify-center"):
                            html.Span(
                                "Drag and drop files or click to open file browser",
                                classes="font-italic",
                            )
                        with vuetify.VRow(classes="ma-2"):
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
                                        classes="py-1 pr-1 mr-1 text-truncate w-100",
                                        style="height: 100px;",
                                        open_on_focus=False,
                                        clearable=False,
                                        v_bind="props",
                                        rules=["[utils.vtk_prompt.rules.json_file]"],
                                    )
                            with vuetify.VTooltip(
                                text=("prompt_file", "No file loaded"),
                                location="top",
                                disabled=("!prompt_object",),
                            ):
                                with vuetify.Template(v_slot_activator="{ props }"):
                                    vuetify.VFileInput(
                                        label="Prompt File",
                                        v_model=("prompt_object", None),
                                        accept=".yaml,.yml",
                                        density="compact",
                                        variant="solo",
                                        prepend_icon="mdi-forum-outline",
                                        hide_details="auto",
                                        classes="py-1 pr-1 mr-1 text-truncate w-100",
                                        style="height: 100px;",
                                        open_on_focus=False,
                                        clearable=False,
                                        v_bind="props",
                                        rules=["[utils.vtk_prompt.rules.yaml_file]"],
                                    )
                        with vuetify.VRow(classes="ma-2 justify-end"):
                            vuetify.VCheckbox(
                                label="Automatically run new conversation files",
                                v_model=("auto_run_conversation_file", True),
                                density="compact",
                                color="primary",
                                hide_details=True,
                            )
                    # Model Tab
                    with vuetify.VTabsWindowItem(value="model"):
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
                                            v_model=("provider", DEFAULT_PROVIDER),
                                            items=("available_providers", []),
                                            density="compact",
                                            variant="outlined",
                                            prepend_icon="mdi-cloud",
                                        )
                                        # Model selection
                                        vuetify.VSelect(
                                            label="Model",
                                            v_model=("model", DEFAULT_MODEL),
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

                    # Advanced Settings Tab
                    with vuetify.VTabsWindowItem(value="advanced"):
                        # RAG Settings Card
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

                        # Generation Settings Card
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
