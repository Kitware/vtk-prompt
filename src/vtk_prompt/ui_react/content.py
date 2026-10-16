"""Main content for the react client: editor, render view, transcript."""

from typing import Any

from trame.widgets import code, html, mui, react
from trame_rca.widgets import rca  # type: ignore[import-not-found]

from ..ui.langs import PYTHON_TEXTMATE
from .model_picker import build_model_picker
from .theme import MONO, MUTED

C = react.Callback
B = react.Bind

EDITOR_OPTIONS = {
    "automaticLayout": True,
    "minimap": {"enabled": False},
    "fontSize": 13,
    "scrollBeyondLastLine": False,
    "lineNumbers": "on",
    "tabSize": 4,
    # Render hover/suggest widgets on the body so surrounding overflow does
    # not clip them.
    "fixedOverflowWidgets": True,
    "hover": {"enabled": True, "sticky": True},
}

PANEL = {
    "background": "var(--vp-surface)",
    "border": "1px solid var(--vp-line)",
    "borderRadius": "6px",
    "display": "flex",
    "flexDirection": "column",
    "minHeight": 0,
    "overflow": "hidden",
}

BAR = {
    "display": "flex",
    "alignItems": "center",
    "gap": "4px",
    "padding": "6px 8px",
    "borderBottom": "1px solid var(--vp-line)",
    "flex": "0 0 auto",
}


def _icon_button(icon, tip, **kwargs):
    with mui.Tooltip(title=tip):
        with mui.IconButton(size="small", **kwargs):
            mui.Icon([icon], font_size="small")


def _data_chips(app: Any) -> None:
    """Show data files the code references, with fixes for missing ones."""
    with react.For(items="data_artifacts", name="a"):
        with mui.Tooltip(
            key=B("a.name"),
            title=B(
                "`${a.source === 'upload' ? 'Uploaded'"
                " : (a.cached ? 'Fetched' : 'Will download on run')}"
                "\\n${a.path}`"
            ),
        ):
            mui.Chip(
                label=B("a.name"),
                size="small",
                variant="outlined",
                color=B("a.source === 'upload' ? 'info'" " : (a.cached ? 'success' : 'primary')"),
                style={"marginRight": "4px"},
            )

    with mui.Tooltip(title="Upload data file"):
        mui.Chip(
            label="Add data",
            size="small",
            variant="outlined",
            on_click=C("window.document.querySelector('#data-upload-input').click()"),
            style={"marginRight": "4px", "cursor": "pointer"},
        )
    html.Input(
        id="data-upload-input",
        type="file",
        multiple=True,
        style={"display": "none"},
        on_change=C(
            "window.trame.utils.vtk_prompt.uploadFiles($event.target.files,"
            f" '{app.ctrl.trigger_name(app.ctrl.import_uploaded_files)}')"
        ),
    )

    with react.If("data_suggestions.length > 0"):
        mui.Button(
            B("`Fix data file (${data_suggestions.length})`"),
            size="small",
            color="warning",
            variant="outlined",
            on_click=C("data_fix_anchor = $event.currentTarget"),
            style={"marginRight": "4px"},
        )
    with mui.Menu(
        open=B("!!data_fix_anchor"),
        anchor_el=B("data_fix_anchor"),
        on_close=C("data_fix_anchor = null"),
    ):
        with react.For(items="data_suggestions", name="pick"):
            mui.MenuItem(
                [B("`${pick.missing} \u2192 ${pick.suggestion}`")],
                key=B("pick.missing + pick.suggestion"),
                on_click=C(
                    app.ctrl.apply_data_suggestion,
                    "[pick.missing, pick.suggestion]",
                ),
            )


def _editor_panel(app: Any) -> None:
    app.state.data_fix_anchor = None
    with html.Div(style={**PANEL, "flex": "1"}):
        with react.If("is_loading"):
            mui.LinearProgress(style={"height": "2px"})
        with html.Div(style=BAR):
            html.Span("Generated Code", style={"fontWeight": 600})
            with react.If("is_loading"):
                html.Span(
                    "generating\u2026",
                    style={"fontSize": "0.78rem", "color": "var(--vp-muted)", "marginLeft": "8px"},
                )
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "flex": "1",
                    "marginLeft": "8px",
                    "overflow": "hidden",
                }
            )
            _data_chips(app)
            _icon_button(
                "undo",
                "Undo",
                on_click=C(app.ctrl.undo_code),
                disabled=B("code_history_pos <= 0"),
            )
            _icon_button(
                "redo",
                "Redo",
                on_click=C(app.ctrl.redo_code),
                disabled=B("code_history_pos >= code_history.length - 1"),
            )
            mui.Button(
                "Run",
                size="small",
                variant="contained",
                start_icon=None,
                on_click=C(app.ctrl.run_current_code),
                disabled=B("is_loading || !generated_code" " || generated_code === rendered_code"),
            )
        with html.Div(style={"flex": "1", "minHeight": 0}):
            code.Editor(
                # Remount only when the conversation changes: keying on code
                # version too would dispose an open autocomplete popup.
                key=B("current_session_id"),
                value=B("generated_code"),
                on_change=C("generated_code = $event"),
                language="python",
                theme=B("theme_mode === 'dark' ? 'vs-dark' : 'vs'"),
                textmate=PYTHON_TEXTMATE,
                completion=app.jedi_complete,
                hover=app.jedi_hover,
                options=EDITOR_OPTIONS,
                style={"height": "100%", "width": "100%"},
            )


def _prompt_panel(app: Any) -> None:
    with html.Div(style={**PANEL, "flex": "0 0 auto", "marginTop": "8px"}):
        with react.If("is_loading"):
            with html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "10px",
                    "padding": "10px 12px",
                    "background": "#E3F2F4",
                    "borderBottom": "1px solid var(--vp-line)",
                }
            ):
                mui.CircularProgress(size=18)
                html.Span(
                    "Generating code\u2026",
                    style={"fontWeight": 600, "fontSize": "0.85rem"},
                )
            mui.LinearProgress(style={"height": "3px"})
        with html.Div(style={"padding": "8px 8px 0"}):
            build_model_picker(app)
        with html.Div(style={"padding": "8px", "display": "flex", "gap": "8px"}):
            mui.TextField(
                label="Describe VTK visualization",
                placeholder="e.g., Create a red sphere with lighting",
                multiline=True,
                rows=3,
                size="small",
                full_width=True,
                value=B("query_text"),
                on_change=C("query_text = $event.target.value"),
            )
            with html.Div(style={"display": "flex", "alignItems": "flex-end"}):
                with react.If("is_loading"):
                    mui.CircularProgress(size=24, style={"margin": "0 8px 8px"})
                with react.If("!is_loading"):
                    with mui.IconButton(
                        color="primary",
                        on_click=C(app.ctrl.generate_code),
                        disabled=B(
                            "!query_text.trim()" " || (use_cloud_models && !api_token.trim())"
                        ),
                    ):
                        mui.Icon(["arrow_upward"])
        with react.If("use_cloud_models && !api_token.trim()"):
            mui.Alert(
                ["API token is required for cloud models."],
                severity="warning",
                style={"margin": "0 8px 8px"},
            )


def _render_panel(app: Any) -> None:
    with html.Div(style={**PANEL, "flex": "1"}):
        with html.Div(style=BAR):
            html.Span("VTK Visualization", style={"fontWeight": 600, "flex": "1"})
            with react.If("input_tokens > 0 || output_tokens > 0"):
                html.Span(
                    B("`Tokens: In ${input_tokens} | Out ${output_tokens}`"),
                    style={
                        "fontSize": "0.72rem",
                        "color": "var(--vp-muted)",
                        "marginRight": "6px",
                    },
                )
            _icon_button(
                "layers_clear",
                "Clear scene",
                on_click=C(app.ctrl.clear_scene),
            )
            _icon_button(
                "center_focus_strong",
                "Reset camera",
                on_click=C(app.ctrl.view_reset_camera),
            )
        with html.Div(style={"flex": "1", "minHeight": 0, "position": "relative"}):
            view = rca.RemoteControlledArea(display="image", name="vtk-view")
            app.view_handler = view.create_view_handler(app.render_window)
            app.ctrl.view_update = app.view_handler.update
            app.ctrl.view_reset_camera = app.reset_camera


def _transcript_panel(app: Any) -> None:
    with html.Div(style={**PANEL, "flex": "0 0 auto", "height": "260px", "marginTop": "8px"}):
        # Only $event (the first handler argument) is exposed, so the tab
        # value cannot come from Tabs' onChange(event, value); each tab sets it.
        # The badge sits beside the tab bar rather than inside the tab: MUI's
        # Tab renders its label prop and ignores children, and an element
        # cannot be passed as a prop value.
        with html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "borderBottom": "1px solid var(--vp-line)",
                "paddingRight": "8px",
            }
        ):
            with mui.Tabs(
                value=B("info_tab"),
                variant="fullWidth",
                style={"minHeight": "36px", "flex": "1"},
            ):
                mui.Tab(
                    label="Conversation",
                    value="conversation",
                    on_click=C("info_tab = 'conversation'"),
                    style={"minHeight": "36px"},
                )
                mui.Tab(
                    label="Console",
                    value="console",
                    on_click=C("info_tab = 'console'"),
                    style={"minHeight": "36px"},
                )
            with react.If("console_log.length > 0"):
                mui.Chip(
                    label=B("String(console_log.length)"),
                    size="small",
                    color=B(
                        "console_level === 'err' ? 'error'"
                        " : console_level === 'warn' ? 'warning' : 'default'"
                    ),
                    on_click=C("info_tab = 'console'"),
                    style={"cursor": "pointer"},
                )

        with react.If("info_tab === 'conversation'"):
            with html.Div(style={"flex": "1", "overflowY": "auto", "padding": "8px"}):
                with react.If("conversation_navigation.length === 0"):
                    html.Div(
                        "No prompts yet.",
                        style={"color": "var(--vp-muted)", "fontSize": "0.8rem"},
                    )
                with react.For(items="conversation_navigation", name="turn"):
                    with html.Div(
                        key=B("'turn-' + conversation_navigation.indexOf(turn)"),
                        classes=B(
                            "conversation_index ==="
                            " conversation_navigation.indexOf(turn)"
                            " ? 'vp-turn vp-turn-active' : 'vp-turn'"
                        ),
                    ):
                        html.Div(
                            B("turn.prompt"),
                            on_click=C(
                                app.ctrl.navigate_to_conversation,
                                "[conversation_navigation.indexOf(turn)]",
                            ),
                            classes="vp-turn-prompt",
                        )
                        with react.If("turn.explanation"):
                            html.Div(
                                B("turn.explanation"),
                                classes="vp-turn-explanation",
                            )
                        with react.If("turn.trace && turn.trace.length"):
                            with mui.Accordion(
                                disable_gutters=True,
                                elevation=0,
                                square=True,
                                style={"background": "transparent"},
                            ):
                                with mui.AccordionSummary(
                                    style={"minHeight": 0, "padding": "0 8px"}
                                ):
                                    html.Span(
                                        B("`Show work (${turn.trace.length})`"),
                                        style={"fontSize": "0.72rem", "color": MUTED},
                                    )
                                with mui.AccordionDetails(style={"padding": "0 8px 8px 20px"}):
                                    with react.For(items="turn.trace", name="step"):
                                        html.Div(
                                            B(
                                                "step.name + (step.detail"
                                                " ? ': ' + step.detail : '')"
                                            ),
                                            classes="vp-step-name",
                                        )
                                        with react.If("step.result"):
                                            html.Div(
                                                B("step.result"),
                                                classes="vp-step-result",
                                            )
                with react.If("is_loading && current_prompt"):
                    with html.Div(classes="vp-turn"):
                        html.Div(B("current_prompt"), classes="vp-turn-prompt")
                        with html.Div(
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "8px",
                                "padding": "4px 0 0 8px",
                            }
                        ):
                            mui.CircularProgress(size=14)
                            html.Span(
                                "thinking\u2026",
                                style={"fontSize": "0.78rem", "color": MUTED},
                            )

        with react.If("info_tab === 'console'"):
            with html.Div(
                style={
                    "flex": "1",
                    "overflow": "auto",
                    "padding": "8px",
                    "fontFamily": MONO,
                    "fontSize": "12px",
                }
            ):
                with react.If("console_lines.length === 0"):
                    html.Div(
                        "No output yet.",
                        style={"color": "var(--vp-muted)", "fontSize": "0.75rem"},
                    )
                with react.If("console_lines.length > 0"):
                    with html.Div(style={"display": "flex", "justifyContent": "flex-end"}):
                        mui.Button(
                            "Clear",
                            size="small",
                            on_click=C("(console_lines = [], console_log = [])"),
                        )
                with react.For(items="console_lines", name="line"):
                    # A bound class name rather than a bound style object:
                    # binding style to an expression rebuilds the object every
                    # render and React loops on it.
                    html.Pre(
                        B("line.text"),
                        key=B("'line-' + console_lines.indexOf(line)"),
                        classes=B("'vp-console-' + line.kind"),
                    )


def build_content(app: Any) -> None:
    """Two columns: editor plus prompt, and render view plus transcript."""
    with html.Div(
        style={
            "display": "flex",
            "gap": "8px",
            "padding": "8px",
            "flex": "1",
            "minHeight": 0,
        }
    ):
        with html.Div(
            style={
                "flex": "1",
                "minWidth": 0,
                "display": "flex",
                "flexDirection": "column",
                "minHeight": 0,
            }
        ):
            _editor_panel(app)
            _prompt_panel(app)
        with html.Div(
            style={
                "flex": "1",
                "minWidth": 0,
                "display": "flex",
                "flexDirection": "column",
                "minHeight": 0,
            }
        ):
            _render_panel(app)
            _transcript_panel(app)
