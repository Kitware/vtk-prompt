"""
Settings Dialog Layout Module.

This module provides the advanced settings dialog layout for the VTK Prompt UI.
The dialog contains model configuration, vtk-mcp settings, and file controls.
"""

from typing import Any

from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

from ...provider_utils import DEFAULT_MODEL, DEFAULT_PROVIDER

vuetify.enable_lab()

_LABEL = "text-overline text-medium-emphasis d-block mb-1"
_DESC = "text-caption text-medium-emphasis d-block mb-3"


def _section(title: str) -> None:
    html.Div(title, classes=_LABEL)


def build_settings_dialog(layout: Any, app: Any) -> None:
    """Build the advanced settings dialog with configuration options."""
    with layout.content:
        with vuetify.VDialog(v_model=("advanced_settings_open", False), max_width="560"):
            with vuetify.VCard():
                with vuetify.VTabs(
                    v_model=("active_settings_tab", "files"),
                    color="primary",
                    classes="px-2",
                ):
                    vuetify.VTab("Config", value="files")
                    vuetify.VTab("Data", value="data")
                    vuetify.VTab("Model", value="model")
                    vuetify.VTab("Advanced", value="advanced")
                vuetify.VDivider()
                with vuetify.VTabsWindow(v_model=("active_settings_tab", "files")):
                    _config_tab()
                    _data_tab()
                    _model_tab()
                    _advanced_tab()


def _config_tab() -> None:
    with vuetify.VTabsWindowItem(value="files"):
        with vuetify.VCardText(classes="pa-4"):
            _section("Configuration")
            html.Div(
                "Import or export the model/prompt config (.yaml). To import a "
                "conversation, use Import in Recents.",
                classes=_DESC,
            )
            vuetify.VFileInput(
                label="Import config file (.yaml, .yml)",
                v_model=("uploaded_files", None),
                accept=".yaml,.yml",
                multiple=True,
                prepend_icon="",
                prepend_inner_icon="mdi-upload",
                density="compact",
                variant="outlined",
                hide_details=True,
                classes="mb-3",
            )
            vuetify.VBtn(
                "Download config",
                variant="outlined",
                color="primary",
                size="small",
                prepend_icon="mdi-download",
                click="window.trame.utils.vtk_prompt.exportConfig()",
            )

            vuetify.VDivider(classes="my-5")
            _section("Behavior")
            vuetify.VCheckbox(
                label="Automatically run imported conversations",
                v_model=("auto_run_conversation_file", True),
                density="compact",
                color="primary",
                hide_details=True,
            )


def _model_tab() -> None:
    with vuetify.VTabsWindowItem(value="model"):
        with vuetify.VCardText(classes="pa-4"):
            with vuetify.VTabs(
                v_model=("tab_index", 0),
                color="primary",
                density="compact",
                classes="mb-4",
            ):
                vuetify.VTab("Cloud", prepend_icon="mdi-cloud-outline")
                vuetify.VTab("Local", prepend_icon="mdi-laptop")
            with vuetify.VTabsWindow(v_model="tab_index", classes="pt-3"):
                with vuetify.VTabsWindowItem():
                    vuetify.VSelect(
                        label="Provider",
                        v_model=("provider", DEFAULT_PROVIDER),
                        items=("available_providers", []),
                        density="compact",
                        variant="outlined",
                        classes="mb-3",
                    )
                    vuetify.VSelect(
                        label="Model",
                        v_model=("model", DEFAULT_MODEL),
                        items=("available_models[provider] || []",),
                        density="compact",
                        variant="outlined",
                        classes="mb-3",
                    )
                    vuetify.VTextField(
                        label="API token",
                        v_model=("api_token", ""),
                        placeholder="Enter your API token",
                        type="password",
                        density="compact",
                        variant="outlined",
                        hint="Required for cloud providers",
                        persistent_hint=True,
                        error=("!api_token", False),
                    )
                with vuetify.VTabsWindowItem():
                    vuetify.VTextField(
                        label="Base URL",
                        v_model=("local_base_url", "http://localhost:11434/v1"),
                        placeholder="http://localhost:11434/v1",
                        density="compact",
                        variant="outlined",
                        hint="Ollama, LM Studio, and other OpenAI-compatible servers",
                        persistent_hint=True,
                        classes="mb-3",
                    )
                    vuetify.VTextField(
                        label="Model name",
                        v_model=("local_model", "devstral"),
                        placeholder="devstral",
                        density="compact",
                        variant="outlined",
                        hint="Model identifier as served by your endpoint",
                        persistent_hint=True,
                        classes="mb-3",
                    )
                    vuetify.VTextField(
                        label="API token",
                        v_model=("api_token", "ollama"),
                        placeholder="ollama",
                        type="password",
                        density="compact",
                        variant="outlined",
                        hint="Optional for local servers",
                        persistent_hint=True,
                    )


def _advanced_tab() -> None:
    with vuetify.VTabsWindowItem(value="advanced"):
        with vuetify.VCardText(classes="pa-4"):
            _section("vtk-mcp")
            html.Div(
                "Connect a vtk-mcp server to ground generation in the real VTK API "
                "and validate code before returning it.",
                classes=_DESC,
            )
            vuetify.VTextField(
                label="Server URL",
                v_model=("mcp_url", ""),
                placeholder="http://localhost:8000",
                clearable=True,
                density="compact",
                variant="outlined",
                append_inner_icon=(
                    "mcp_status === 'ok' ? 'mdi-check-circle' : "
                    "mcp_status === 'error' ? 'mdi-alert-circle' : "
                    "mcp_status === 'checking' ? 'mdi-loading mdi-spin' : ''",
                ),
                color=(
                    "mcp_status === 'ok' ? 'success' : "
                    "mcp_status === 'error' ? 'error' : undefined",
                ),
                hint="Leave blank for baseline generation without tools",
                persistent_hint=True,
                classes="mb-3",
            )
            vuetify.VTextField(
                label="Top K",
                v_model=("top_k", 5),
                type="number",
                min=1,
                max=15,
                density="compact",
                variant="outlined",
                disabled=("!mcp_url",),
                hint="Context snippets retrieved per request",
                persistent_hint=True,
                classes="mb-3",
            )
            vuetify.VCheckbox(
                label="Auto-translate to DSL",
                v_model=("dsl_translation", True),
                density="compact",
                color="primary",
                disabled=("!mcp_url",),
                hide_details="auto",
                hint="Convert natural language prompts to the VTK "
                "pipeline DSL before code generation",
                persistent_hint=True,
            )
            vuetify.VCheckbox(
                label="Log tool calls to the server console",
                v_model=("log_tool_calls", False),
                density="compact",
                color="primary",
                disabled=("!mcp_url",),
                hide_details=True,
                classes="mt-2",
            )
            vuetify.VCheckbox(
                label="Agentic retrieval (use tools instead of pre-injected context)",
                v_model=("agentic_retrieval", False),
                density="compact",
                color="primary",
                disabled=("!mcp_url",),
                hide_details=True,
            )

            vuetify.VDivider(classes="my-5")
            _section("Generation")
            with html.Div(classes="d-flex align-center justify-space-between mt-2 mb-1"):
                html.Span("Temperature", classes="text-body-2")
                html.Span(
                    "{{ temperature }}", classes="text-body-2 text-medium-emphasis"
                )
            vuetify.VSlider(
                v_model=("temperature", 0.1),
                min=0.0,
                max=1.0,
                step=0.1,
                thumb_label=True,
                color="primary",
                hide_details=True,
                disabled=("!temperature_supported",),
                classes="mb-4",
            )
            vuetify.VTextField(
                label="Max tokens",
                v_model=("max_tokens", 1000),
                type="number",
                density="compact",
                variant="outlined",
                classes="mb-3",
            )
            vuetify.VTextField(
                label="Retry attempts",
                v_model=("retry_attempts", 3),
                type="number",
                min=1,
                max=5,
                density="compact",
                variant="outlined",
            )


def _data_tab() -> None:
    with vuetify.VTabsWindowItem(value="data"):
        with vuetify.VCardText(classes="pa-4"):
            _section("Sample data location")
            html.Div(
                "Local VTK data tree used to resolve example datasets by name "
                "(e.g. cow.g).",
                classes=_DESC,
            )
            vuetify.VTextField(
                label="Sample data location",
                v_model=("data_root", ""),
                placeholder="/path/to/VTK/Testing/Data",
                hint="Folder of .sha512 data pointers; blank uses the "
                "VTK_PROMPT_DATA_ROOT environment variable",
                persistent_hint=True,
                density="compact",
                variant="outlined",
                clearable=True,
            )

            vuetify.VDivider(classes="my-5")
            _section("Uploaded files")
            html.Div(
                "Your own data files, referenced by bare name in generated code.",
                classes=_DESC,
            )
            html.Div(
                "No files uploaded yet.",
                classes="text-caption text-disabled mb-2",
                v_show="uploaded_data_files.length === 0",
            )
            with html.Div(
                classes="d-flex flex-wrap",
                v_show="uploaded_data_files.length > 0",
            ):
                vuetify.VChip(
                    "{{ f }}",
                    v_for="f in uploaded_data_files",
                    key="f",
                    size="small",
                    variant="tonal",
                    color="info",
                    closable=True,
                    click_close="window.trame.trigger('remove_uploaded_file', [f])",
                    prepend_icon="mdi-file-outline",
                    classes="mr-1 mb-1",
                )

            vuetify.VDivider(classes="my-5")
            with html.Div(classes="d-flex align-center justify-space-between"):
                html.Div("Fetched sample data", classes=_LABEL)
                vuetify.VBtn(
                    "Clear cache",
                    variant="text",
                    size="small",
                    color="primary",
                    prepend_icon="mdi-delete-outline",
                    click="window.trame.trigger('clear_data_cache')",
                    v_show="cached_data_files.length > 0",
                )
            html.Div(
                "Sample datasets downloaded to the local cache.",
                classes=_DESC,
            )
            html.Div(
                "Nothing fetched yet.",
                classes="text-caption text-disabled mb-2",
                v_show="cached_data_files.length === 0",
            )
            with html.Div(
                classes="d-flex flex-wrap",
                v_show="cached_data_files.length > 0",
            ):
                vuetify.VChip(
                    "{{ f }}",
                    v_for="f in cached_data_files",
                    key="f",
                    size="small",
                    variant="tonal",
                    color="success",
                    prepend_icon="mdi-file-check",
                    classes="mr-1 mb-1",
                )
