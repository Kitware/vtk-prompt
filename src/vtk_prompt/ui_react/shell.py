"""React UI shell: app bar, conversations drawer, content."""

from typing import Any

from trame.ui.mui import MuiLayout
from trame.widgets import html, mui, react

from .content import build_content
from .drawer import build_conversation_history
from .settings import build_settings_dialog
from .theme import CANVAS, FONTS, ICON_FONT, LINE, SURFACE, THEME
from .toolbar import build_toolbar

DRAWER_WIDTH = 320


def build_ui(app: Any) -> None:
    """Assemble the react layout against the existing controllers."""
    app.state.main_drawer = True

    with MuiLayout(app.server, theme=THEME) as layout:
        mui.GlobalStyles(
            styles=(
                f"@import url('{FONTS}');"
                f"@import url('{ICON_FONT}');"
                f"body {{ background: {CANVAS}; margin: 0; }}"
            )
        )
        with html.Div(
            style={
                "display": "flex",
                "flexDirection": "column",
                "height": "100vh",
                "background": CANVAS,
            }
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
                            "borderRight": f"1px solid {LINE}",
                            "background": SURFACE,
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

    app.ui = layout
