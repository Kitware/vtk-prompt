"""Per-conversation model picker and its settings menu."""

from typing import Any

from trame.widgets import html, mui, react

C = react.Callback
B = react.Bind

LABEL = {
    "fontSize": "0.68rem",
    "letterSpacing": "0.06em",
    "textTransform": "uppercase",
    "color": "var(--mui-palette-text-secondary)",
    "marginBottom": "6px",
}

FIELD = {"marginBottom": "10px"}


def _chip_label() -> str:
    return (
        "use_cloud_models"
        " ? `\u2601\ufe0f ${provider}/${model}`"
        " : `\U0001f3e0 ${local_base_url"
        ".replace('http://', '').replace('https://', '')}/${local_model}`"
    )


def build_model_picker(app: Any) -> None:
    """Model chip with a list menu, plus a settings menu beside it."""
    app.state.model_menu_anchor = None
    app.state.model_settings_anchor = None

    with html.Div(style={"display": "flex", "alignItems": "center", "gap": "4px"}):
        mui.Chip(
            label=B(_chip_label()),
            size="small",
            color=B("use_cloud_models ? 'primary' : 'success'"),
            variant="outlined",
            on_click=C("model_menu_anchor = $event.currentTarget"),
            style={"cursor": "pointer"},
        )
        with mui.Tooltip(title="Model settings"):
            with mui.IconButton(
                size="small",
                on_click=C("model_settings_anchor = $event.currentTarget"),
            ):
                mui.Icon(["tune"], font_size="small")

    with mui.Menu(
        open=B("!!model_menu_anchor"),
        anchor_el=B("model_menu_anchor"),
        on_close=C("model_menu_anchor = null"),
    ):
        with html.Div(style={"maxHeight": "40vh", "overflowY": "auto"}):
            with react.For(items="model_options", name="opt"):
                mui.MenuItem(
                    [B("opt.label")],
                    key=B("opt.key"),
                    on_click=C(
                        "(window.trame.trigger"
                        f"('{app.ctrl.trigger_name(app.ctrl.select_model)}',"
                        " [opt.key]), model_menu_anchor = null)"
                    ),
                )

    with mui.Menu(
        open=B("!!model_settings_anchor"),
        anchor_el=B("model_settings_anchor"),
        on_close=C("model_settings_anchor = null"),
    ):
        with html.Div(style={"width": "360px", "padding": "12px 16px"}):
            html.Div("Connection", style=LABEL)
            with react.If("use_cloud_models"):
                mui.TextField(
                    label="API token",
                    type="password",
                    placeholder="Enter your API token",
                    size="small",
                    full_width=True,
                    error=B("!api_token"),
                    value=B("api_token"),
                    on_change=C("api_token = $event.target.value"),
                    style=FIELD,
                )
            with react.If("!use_cloud_models"):
                mui.TextField(
                    label="Base URL",
                    size="small",
                    full_width=True,
                    value=B("local_base_url"),
                    on_change=C("local_base_url = $event.target.value"),
                    style=FIELD,
                )
                mui.TextField(
                    label="Model name",
                    size="small",
                    full_width=True,
                    value=B("local_model"),
                    on_change=C("local_model = $event.target.value"),
                    style=FIELD,
                )
                mui.TextField(
                    label="API token",
                    type="password",
                    size="small",
                    full_width=True,
                    value=B("api_token"),
                    on_change=C("api_token = $event.target.value"),
                    style=FIELD,
                )

            mui.Divider(style={"margin": "4px 0 12px"})
            html.Div("Generation", style=LABEL)
            with html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "fontSize": "0.8rem",
                }
            ):
                html.Span("Temperature")
                html.Span(
                    B("String(temperature)"), style={"color": "var(--mui-palette-text-secondary)"}
                )
            mui.Slider(
                min=0,
                max=1,
                step=0.1,
                size="small",
                disabled=B("!temperature_supported"),
                value=B("Number(temperature)"),
                on_change=C("temperature = $event.target.value"),
            )
            mui.TextField(
                label="Max tokens",
                type="number",
                size="small",
                full_width=True,
                value=B("max_tokens"),
                on_change=C("max_tokens = $event.target.value"),
                style={**FIELD, "marginTop": "8px"},
            )
            mui.TextField(
                label="Retry attempts",
                type="number",
                size="small",
                full_width=True,
                value=B("retry_attempts"),
                on_change=C("retry_attempts = $event.target.value"),
            )
