"""Top bar for the react client."""

from typing import Any

from trame.widgets import html, mui, react

C = react.Callback
B = react.Bind

_SETTINGS = [
    ("files", "folder_open", "Load or download files"),
    ("data", "storage", "Data files"),
    ("advanced", "settings", "Advanced settings"),
]


def build_toolbar(app: Any) -> None:
    """Title, drawer toggle and settings entry points."""
    with html.Header(
        style={
            "background": "var(--mui-palette-background-paper)",
            "borderBottom": "1px solid var(--mui-palette-divider)",
            "display": "flex",
            "alignItems": "center",
            "gap": "10px",
            "height": "52px",
            "padding": "0 12px 0 6px",
            "flex": "0 0 auto",
        }
    ):
        with mui.Tooltip(title="Conversations"):
            with mui.IconButton(
                size="small", on_click=react.Callback("main_drawer = !main_drawer")
            ):
                mui.Icon(["menu"], font_size="small")

        html.Span(
            "VTK Prompt",
            style={
                "fontWeight": 600,
                "fontSize": "1.05rem",
                "color": "var(--mui-palette-text-primary)",
                "letterSpacing": "-0.01em",
            },
        )
        html.Span(
            react.Bind("(sessions_list.find((s) => s.active) || {}).title || ''"),
            style={
                "color": "var(--mui-palette-text-secondary)",
                "fontSize": "0.8rem",
                "maxWidth": "38vw",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
            },
        )

        html.Div(style={"flex": "1"})

        with mui.Tooltip(title="Toggle light/dark"):
            with mui.IconButton(
                size="small",
                color="primary",
                on_click=C("theme_mode = (theme_mode === 'light' ? 'dark' : 'light')"),
            ):
                mui.Icon(
                    [B("theme_mode === 'light' ? 'light_mode' : 'dark_mode'")],
                    font_size="small",
                )

        with react.If("auth_enabled"):
            with mui.Tooltip(title="Sign out"):
                # href needs component="a", which trame-mui does not pass
                # through, so the navigation happens on click instead.
                with mui.IconButton(
                    size="small",
                    color="primary",
                    on_click=C("window.location.href = '/logout'"),
                ):
                    mui.Icon(["logout"], font_size="small")

        for tab, icon, tip in _SETTINGS:
            with mui.Tooltip(title=tip, key=tab):
                with mui.IconButton(
                    size="small",
                    color="primary",
                    on_click=react.Callback(
                        f"(advanced_settings_open = true," f" active_settings_tab = '{tab}')"
                    ),
                ):
                    mui.Icon([icon], font_size="small")
