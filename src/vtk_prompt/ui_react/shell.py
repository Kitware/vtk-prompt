"""React UI shell: app bar, conversations drawer, content."""

from typing import Any

from trame.ui.mui import MuiLayout
from trame.widgets import html, mui, react

from .content import build_content
from .drawer import build_conversation_history
from .settings import build_settings_dialog
from .theme import (
    CANVAS,
    ERR,
    FONTS,
    ICON_FONT,
    INK,
    LINE,
    MONO,
    MUTED,
    SURFACE,
    THEME,
    WARN,
)
from .toolbar import build_toolbar

DRAWER_WIDTH = 320


def build_ui(app: Any) -> None:
    """Assemble the react layout against the existing controllers."""
    app.state.main_drawer = True
    app.state.theme_mode = "light"

    # mode is a declared prop on ThemeProvider, so binding it to theme_mode
    # makes light/dark switch without rebuilding the layout.
    with MuiLayout(app.server, theme=THEME, mode=react.Bind("theme_mode")) as layout:
        mui.GlobalStyles(
            styles=(
                f"@import url('{FONTS}');"
                f"@import url('{ICON_FONT}');"
                ".vp-light {"
                f" --vp-canvas: {CANVAS}; --vp-surface: {SURFACE};"
                f" --vp-line: {LINE}; --vp-muted: {MUTED}; --vp-ink: {INK};"
                " --vp-selected: #E3F2F4;"
                " --vp-selected-hover: #D2E9ED; }"
                ".vp-dark {"
                " --vp-canvas: #11161F; --vp-surface: #182230;"
                " --vp-line: #2A3647; --vp-muted: #93A0B4; --vp-ink: #E6EAF2;"
                " --vp-selected: #0E3A42;"
                " --vp-selected-hover: #124A54; }"
                ".vp-console-out, .vp-console-err, .vp-console-warn,"
                " .vp-console-run { margin: 0; white-space: pre;"
                f" width: max-content; font-family: {MONO}; font-size: 12px; }}"
                f".vp-console-err {{ color: {ERR}; }}"
                f".vp-console-warn {{ color: {WARN}; }}"
                ".vp-console-run { color: var(--vp-muted); font-weight: 500; }"
                ".vp-turn { border-left: 3px solid transparent;"
                " padding: 2px 0 2px 6px; margin-bottom: 6px; }"
                ".vp-turn-active { border-left-color: var(--vp-accent, #0A7C8C); }"
                ".vp-turn-prompt { font-size: 0.82rem; cursor: pointer;"
                " padding: 2px 8px; }"
                ".vp-turn-explanation { font-size: 0.8rem;"
                " color: var(--vp-muted); white-space: pre-wrap;"
                " padding: 0 8px 0 20px; }"
                ".vp-step-name { font-size: 0.72rem; font-weight: 500; }"
                ".vp-step-result { font-size: 0.72rem; color: var(--vp-muted);"
                " white-space: pre-wrap; margin-bottom: 6px; }"
            )
        )
        with html.Div(
            classes=react.Bind("theme_mode === 'dark' ? 'vp-dark' : 'vp-light'"),
            style={
                "display": "flex",
                "flexDirection": "column",
                "height": "100vh",
                "background": "var(--vp-canvas)",
                "color": "var(--vp-ink)",
            },
        ):
            build_toolbar(app)
            with html.Div(style={"display": "flex", "flex": "1", "minHeight": 0}):
                # A plain panel rather than mui.Drawer: the sidebar is always
                # in flow, and Drawer renders fixed-position in a portal.
                # Mounted conditionally rather than width-bound, since binding
                # style to an expression rebuilds the object every render and
                # React loops on it.
                with react.If("main_drawer"):
                    with html.Aside(
                        style={
                            "width": f"{DRAWER_WIDTH}px",
                            "flex": "0 0 auto",
                            "overflow": "hidden",
                            "borderRight": "1px solid var(--vp-line)",
                            "background": "var(--vp-surface)",
                            "display": "flex",
                            "flexDirection": "column",
                            "minHeight": 0,
                        }
                    ):
                        build_conversation_history(app)
                with html.Div(
                    style={
                        "flex": "1",
                        "minWidth": 0,
                        "display": "flex",
                        "flexDirection": "column",
                    }
                ):
                    build_content(app)
        build_settings_dialog(app)
        with mui.Snackbar(
            open=react.Bind("toast_visible"),
            auto_hide_duration=5000,
            on_close=react.Callback("toast_visible = false"),
            anchor_origin={"vertical": "top", "horizontal": "center"},
        ):
            mui.Alert(
                [react.Bind("toast_message")],
                severity=react.Bind(
                    "toast_color === 'error' ? 'error'"
                    " : toast_color === 'success' ? 'success' : 'warning'"
                ),
                variant="filled",
            )

    app.ui = layout
