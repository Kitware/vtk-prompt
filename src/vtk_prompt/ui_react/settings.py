"""Global settings dialog for the react client."""

from typing import Any

from trame.widgets import html, mui, react

from .theme import MUTED

C = react.Callback
B = react.Bind

LABEL = {
    "fontSize": "0.68rem",
    "letterSpacing": "0.06em",
    "textTransform": "uppercase",
    "color": MUTED,
    "margin": "0 0 6px",
}
DESC = {"fontSize": "0.78rem", "color": MUTED, "marginBottom": "10px"}
FIELD = {"marginBottom": "12px"}

TABS = [("files", "Config"), ("data", "Data"), ("advanced", "Advanced")]


def _check(label: str, var: str, **kwargs) -> None:
    with html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}):
        mui.Checkbox(
            size="small",
            checked=B(var),
            on_change=C(f"{var} = $event.target.checked"),
            style={"padding": "4px"},
            **kwargs,
        )
        html.Span(label, style={"fontSize": "0.85rem"})


def _config_tab(app: Any) -> None:
    with react.If("active_settings_tab === 'files'"):
        with html.Div(style={"padding": "4px 2px"}):
            html.Div("Configuration", style=LABEL)
            html.Div(
                "Import or export the model/prompt config (.yaml). To import a"
                " conversation, use Import in Recents.",
                style=DESC,
            )
            html.Input(
                type="file",
                accept=".yaml,.yml",
                multiple=True,
                on_change=C(
                    "window.trame.utils.vtk_prompt.uploadFiles($event.target.files,"
                    f" '{app.ctrl.trigger_name(app.ctrl.import_uploaded_files)}')"
                ),
                style={"marginBottom": "12px", "display": "block"},
            )
            mui.Button(
                "Download config",
                size="small",
                variant="outlined",
                on_click=C("window.trame.utils.vtk_prompt.exportConfig()"),
            )
            mui.Divider(style={"margin": "18px 0"})
            html.Div("Behavior", style=LABEL)
            _check(
                "Automatically run imported conversations",
                "auto_run_conversation_file",
            )


def _data_tab(app: Any) -> None:
    with react.If("active_settings_tab === 'data'"):
        with html.Div(style={"padding": "4px 2px"}):
            html.Div("Sample data location", style=LABEL)
            html.Div(
                "Local VTK data tree used to resolve example datasets by name" " (e.g. cow.g).",
                style=DESC,
            )
            mui.TextField(
                label="Sample data location",
                placeholder="/path/to/VTK/Testing/Data",
                helper_text=(
                    "Folder of .sha512 data pointers; blank uses the"
                    " VTK_PROMPT_DATA_ROOT environment variable"
                ),
                size="small",
                full_width=True,
                value=B("data_root"),
                on_change=C("data_root = $event.target.value"),
                style=FIELD,
            )
            mui.Divider(style={"margin": "18px 0"})
            html.Div("Uploaded files", style=LABEL)
            html.Div(
                "Your own data files, referenced by bare name in generated code.",
                style=DESC,
            )
            with react.If("uploaded_data_files.length === 0"):
                html.Div(
                    "No files uploaded yet.",
                    style={"fontSize": "0.78rem", "color": MUTED},
                )
            with html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}):
                with react.For(items="uploaded_data_files", name="f"):
                    mui.Chip(label=B("f"), key=B("f"), size="small", variant="outlined")


def _advanced_tab(app: Any) -> None:
    with react.If("active_settings_tab === 'advanced'"):
        with html.Div(style={"padding": "4px 2px"}):
            html.Div("vtk-mcp", style=LABEL)
            html.Div(
                "Connect a vtk-mcp server to ground generation in the real VTK"
                " API and validate code before returning it.",
                style=DESC,
            )
            mui.TextField(
                label="Server URL",
                placeholder="http://localhost:8000",
                helper_text="Leave blank for baseline generation without tools",
                size="small",
                full_width=True,
                value=B("mcp_url"),
                on_change=C("mcp_url = $event.target.value"),
                color=B(
                    "mcp_status === 'ok' ? 'success'"
                    " : mcp_status === 'error' ? 'error' : 'primary'"
                ),
                style=FIELD,
            )
            mui.TextField(
                label="Top K",
                type="number",
                size="small",
                full_width=True,
                helper_text="Context snippets retrieved per request",
                disabled=B("!mcp_url"),
                value=B("top_k"),
                on_change=C("top_k = $event.target.value"),
                style=FIELD,
            )
            _check("Auto-translate to DSL", "dsl_translation", disabled=B("!mcp_url"))
            _check(
                "Log tool calls to the server console",
                "log_tool_calls",
                disabled=B("!mcp_url"),
            )
            _check(
                "Agentic retrieval (use tools instead of pre-injected context)",
                "agentic_retrieval",
                disabled=B("!mcp_url"),
            )


def build_settings_dialog(app: Any) -> None:
    """Config, Data and Advanced tabs in one dialog."""
    with mui.Dialog(
        open=B("advanced_settings_open"),
        on_close=C("advanced_settings_open = false"),
        max_width="sm",
        full_width=True,
    ):
        mui.DialogTitle(["Settings"])
        with mui.DialogContent():
            with mui.Tabs(value=B("active_settings_tab"), variant="fullWidth"):
                for value, label in TABS:
                    mui.Tab(
                        label=label,
                        value=value,
                        key=value,
                        on_click=C(f"active_settings_tab = '{value}'"),
                    )
            with html.Div(style={"paddingTop": "14px"}):
                _config_tab(app)
                _data_tab(app)
                _advanced_tab(app)
        with mui.DialogActions():
            mui.Button("Close", on_click=C("advanced_settings_open = false"))
