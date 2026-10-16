"""React UI shell: app bar, conversations drawer, content."""

from typing import Any

from trame.ui.mui import MuiLayout
from trame.widgets import html, mui, react

from .content import build_content
from .drawer import build_conversation_history
from .settings import build_settings_dialog
from .theme import CANVAS, ERR, FONTS, ICON_FONT, MONO, THEME, WARN
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
                # The one brand deviation from the MUI palette: a tinted canvas
                # in light mode so panels read as raised. Dark mode takes MUI's
                # background.default, which already separates from paper.
                f".vp-light-canvas {{ background: {CANVAS}; }}"
                ".vp-console-out, .vp-console-err, .vp-console-warn,"
                " .vp-console-run { margin: 0; white-space: pre;"
                f" width: max-content; font-family: {MONO}; font-size: 12px; }}"
                f".vp-console-err {{ color: {ERR}; }}"
                f".vp-console-warn {{ color: {WARN}; }}"
                ".vp-console-run { color: var(--mui-palette-text-secondary);"
                " font-weight: 500; }"
                ".vp-turn { border-left: 3px solid transparent;"
                " padding: 2px 0 2px 6px; margin-bottom: 6px; }"
                ".vp-turn-active {"
                " border-left-color: var(--mui-palette-primary-main); }"
                ".vp-turn-prompt { font-size: 0.82rem; cursor: pointer;"
                " padding: 2px 8px; }"
                ".vp-turn-explanation { font-size: 0.8rem;"
                " color: var(--mui-palette-text-secondary);"
                " white-space: pre-wrap; padding: 0 8px 0 20px; }"
                ".vp-step-name { font-size: 0.72rem; font-weight: 500; }"
                ".vp-step-result { font-size: 0.72rem;"
                " color: var(--mui-palette-text-secondary);"
                " white-space: pre-wrap; margin-bottom: 6px; }"
            )
        )
        with html.Div(
            classes=react.Bind("theme_mode === 'light' ? 'vp-light-canvas' : ''"),
            style={
                "display": "flex",
                "flexDirection": "column",
                "height": "100vh",
                "color": "var(--mui-palette-text-primary)",
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
                            "borderRight": "1px solid var(--mui-palette-divider)",
                            "background": "var(--mui-palette-background-paper)",
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
