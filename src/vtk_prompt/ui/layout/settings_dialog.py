"""
Settings Dialog Layout Module.

This module provides the advanced settings dialog layout for the VTK Prompt UI.
The dialog contains model configuration, vtk-mcp settings, and file controls.
"""

from typing import Any

from trame.widgets import vuetify3 as vuetify

from ...provider_utils import DEFAULT_MODEL, DEFAULT_PROVIDER

vuetify.enable_lab()


def build_settings_dialog(layout: Any, app: Any) -> None:
    """Build the advanced settings dialog with configuration options."""
    with layout.content:
        with vuetify.VDialog(v_model=("advanced_settings_open", False), classes="w-33"):
            with vuetify.VCard():
                with vuetify.VTabs(
                    v_model=("active_settings_tab", "files"),
                    color="primary",
                    classes="pa-1",
                ):
                    vuetify.VTab("Config", value="files")
                    vuetify.VTab("Model", value="model")
                    vuetify.VTab("Advanced", value="advanced")
                with vuetify.VTabsWindow(v_model=("active_settings_tab", "files")):
                    # Files Tab
                    with vuetify.VTabsWindowItem(value="files"):
                        with vuetify.VCard():
                            with vuetify.VCardTitle("Import config"):
                                vuetify.VCardSubtitle(
                                    "Load a prompt/config file (.yaml/.yml). To import "
                                    "a conversation, use Import in Recents."
                                )
                            with vuetify.VCardText():
                                vuetify.VFileUpload(
                                    label="Choose a config file (.yaml, .yml)",
                                    v_model=("uploaded_files", None),
                                    accept=".yaml,.yml",
                                    multiple=True,
                                    hide_details="auto",
                                    classes="py-3 pr-1 mr-1 w-100",
                                    color="teal-lighten-5",
                                )
                        with vuetify.VCard():
                            with vuetify.VCardTitle("Settings"):
                                vuetify.VCardSubtitle("Configure default behavior")
                            with vuetify.VCardText():
                                vuetify.VCheckbox(
                                    label="Automatically run new conversation files",
                                    v_model=("auto_run_conversation_file", True),
                                    density="compact",
                                    color="primary",
                                    hide_details=True,
                                )
                        with vuetify.VCard():
                            with vuetify.VCardTitle("Export config"):
                                vuetify.VCardSubtitle(
                                    "Download the current settings as a config file"
                                )
                            with vuetify.VCardText():
                                vuetify.VBtn(
                                    "Download Config File",
                                    color="secondary",
                                    classes="mt-2 w-100",
                                    click="window.trame.utils.vtk_prompt.exportConfig()",
                                    append_icon="mdi-download",
                                )
                        with vuetify.VCard():
                            with vuetify.VCardTitle("Sample data"):
                                vuetify.VCardSubtitle(
                                    "Local VTK data tree used to resolve example "
                                    "datasets by name (e.g. cow.g)"
                                )
                            with vuetify.VCardText():
                                vuetify.VTextField(
                                    label="Sample data location",
                                    v_model=("data_root", ""),
                                    placeholder="/path/to/VTK/Testing/Data",
                                    hint="Folder of .sha512 data pointers; blank uses "
                                    "the VTK_PROMPT_DATA_ROOT environment variable",
                                    persistent_hint=True,
                                    density="compact",
                                    hide_details="auto",
                                    clearable=True,
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
                        # vtk-mcp Settings Card
                        with vuetify.VCard(classes="mt-2"):
                            vuetify.VCardTitle("🔌 vtk-mcp", classes="pb-0")
                            with vuetify.VCardText():
                                vuetify.VTextField(
                                    label="vtk-mcp Server URL",
                                    v_model=("mcp_url", ""),
                                    placeholder="http://localhost:8000",
                                    clearable=True,
                                    density="compact",
                                    variant="outlined",
                                    prepend_icon="mdi-bookshelf",
                                    hint="Enables context retrieval and VTK API "
                                    "validation; leave blank for baseline generation",
                                    persistent_hint=True,
                                )
                                vuetify.VTextField(
                                    label="Top K",
                                    v_model=("top_k", 5),
                                    type="number",
                                    min=1,
                                    max=15,
                                    density="compact",
                                    disabled=("!mcp_url",),
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
                                    v_model=("retry_attempts", 3),
                                    type="number",
                                    min=1,
                                    max=5,
                                    density="compact",
                                    variant="outlined",
                                    prepend_icon="mdi-repeat",
                                )
