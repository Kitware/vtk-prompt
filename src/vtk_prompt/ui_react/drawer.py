"""Recents drawer for the react client."""

from typing import Any

from trame.widgets import html, mui, react

C = react.Callback
B = react.Bind

EXPORT_ONE = "window.trame.utils.vtk_prompt.exportSession(s.id, s.title)"
EXPORT_MANY = "window.trame.utils.vtk_prompt.exportSessions(selected_session_ids, sessions_list)"


def _header(app: Any) -> None:
    with html.Div(
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "2px",
            "padding": "10px 8px 6px 16px",
        }
    ):
        html.Span("Recents", style={"fontWeight": 600, "flex": "1"})
        with mui.Tooltip(title="New conversation"):
            with mui.IconButton(
                size="small",
                color="primary",
                on_click=C(app.ctrl.start_new_conversation),
                disabled=B("conversation_navigation.length === 0"),
            ):
                mui.Icon(["add"], font_size="small")
        with mui.Tooltip(title="Import conversation"):
            with mui.IconButton(size="small", color="primary", on_click=C("import_dialog = true")):
                mui.Icon(["file_upload"], font_size="small")
        with mui.Tooltip(title="Toggle sort order"):
            with mui.IconButton(
                size="small",
                color="primary",
                on_click=C(
                    "history_sort_order = "
                    "(history_sort_order === 'newest') ? 'oldest' : 'newest'"
                ),
                disabled=B("sessions_list.length === 0"),
            ):
                mui.Icon(
                    [B("history_sort_order === 'newest'" " ? 'arrow_downward' : 'arrow_upward'")],
                    font_size="small",
                )


def _selection_bar(app: Any) -> None:
    """Bulk actions, shown once anything is checked.

    Pin and Unpin stay separate: a mixed selection has nothing coherent to
    toggle to, and each is enabled only when it would change something.
    """
    with react.If("selected_session_ids.length > 0"):
        with html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "flexWrap": "wrap",
                "gap": "2px",
                "padding": "0 8px 8px 16px",
            }
        ):
            html.Span(
                B("`${selected_session_ids.length} selected`"),
                style={"fontSize": "0.75rem", "color": "var(--vp-muted)", "marginRight": "4px"},
            )
            mui.Button(
                "All",
                size="small",
                on_click=C(app.ctrl.select_all_sessions),
                disabled=B("selected_session_ids.length === sessions_list.length"),
            )
            mui.Button(
                "None",
                size="small",
                on_click=C("selected_session_ids = []"),
            )
            html.Div(style={"flex": "1"})
            with mui.Tooltip(title="Pin selected"):
                with mui.IconButton(
                    size="small",
                    on_click=C(app.ctrl.set_selection_pinned, "[true]"),
                    disabled=B(
                        "!sessions_list.some("
                        "(s) => selected_session_ids.includes(s.id) && !s.pinned)"
                    ),
                ):
                    mui.Icon(["push_pin"], font_size="small")
            with mui.Tooltip(title="Unpin selected"):
                with mui.IconButton(
                    size="small",
                    on_click=C(app.ctrl.set_selection_pinned, "[false]"),
                    disabled=B(
                        "!sessions_list.some("
                        "(s) => selected_session_ids.includes(s.id) && s.pinned)"
                    ),
                ):
                    mui.Icon(["remove_circle_outline"], font_size="small")
            with mui.Tooltip(title="Export selected"):
                with mui.IconButton(size="small", on_click=C(EXPORT_MANY)):
                    mui.Icon(["file_download"], font_size="small")
            with mui.Tooltip(title="Delete selected"):
                with mui.IconButton(
                    size="small", color="error", on_click=C("bulk_delete_dialog = true")
                ):
                    mui.Icon(["delete"], font_size="small")


def _row_menu(app: Any) -> None:
    with mui.IconButton(
        size="small",
        on_click=C("(row_menu_id = s.id, row_menu_anchor = $event.currentTarget)"),
    ):
        mui.Icon(["more_vert"], font_size="small")


def _row_menu_popup(app: Any) -> None:
    """One menu reused by every row, anchored to whichever opened it."""
    close = "row_menu_id = '', row_menu_anchor = null"
    row = "(sessions_list.find((x) => x.id === row_menu_id) || {})"
    with mui.Menu(
        open=B("!!row_menu_id"),
        anchor_el=B("row_menu_anchor"),
        on_close=C(close),
    ):
        mui.MenuItem(
            [B(f"{row}.pinned ? 'Unpin' : 'Pin'")],
            on_click=C(app.ctrl.toggle_pin_session, "[row_menu_id]"),
        )
        mui.MenuItem(
            ["Rename"],
            on_click=C(
                f"(rename_target_id = row_menu_id, rename_text = {row}.title,"
                f" rename_dialog = true, {close})"
            ),
        )
        mui.MenuItem(
            ["Export"],
            on_click=C(
                "(window.trame.utils.vtk_prompt.exportSession(row_menu_id,"
                f" {row}.title), {close})"
            ),
        )
        mui.MenuItem(
            ["Delete"],
            on_click=C(
                f"(delete_target_id = row_menu_id, delete_target_title = {row}.title,"
                f" delete_dialog = true, {close})"
            ),
        )


def _list(app: Any) -> None:
    with html.Div(style={"flex": "1", "overflowY": "auto", "padding": "0 8px 8px"}):
        with react.If("sessions_list.length === 0"):
            mui.Alert(
                ["No conversations yet. Start by generating some VTK code!"],
                severity="info",
                variant="outlined",
                style={"margin": "8px"},
            )
        with mui.List(dense=True), react.For(items="sessions_list", name="s"):
            with html.Div(
                key=B("s.id"),
                style={"display": "flex", "alignItems": "center", "gap": "2px"},
            ):
                mui.Checkbox(
                    size="small",
                    checked=B("selected_session_ids.includes(s.id)"),
                    on_change=C(
                        "selected_session_ids = $event.target.checked"
                        " ? selected_session_ids.concat([s.id])"
                        " : selected_session_ids.filter((x) => x !== s.id)"
                    ),
                    style={"padding": "4px"},
                )
                with mui.ListItemButton(
                    selected=B("s.active"),
                    on_click=C(app.ctrl.switch_session, "[s.id]"),
                    style={"paddingLeft": "4px", "minWidth": 0, "flex": "1"},
                ):
                    with react.If("s.pinned"):
                        mui.Icon(
                            ["push_pin"],
                            color="primary",
                            style={"fontSize": "14px", "marginRight": "4px"},
                        )
                    mui.ListItemText(
                        primary=B("s.title"),
                        primary_typography_props={
                            "noWrap": True,
                            "style": {"fontSize": "0.85rem"},
                        },
                    )
                _row_menu(app)


def _dialogs(app: Any) -> None:
    with mui.Dialog(
        open=B("rename_dialog"),
        on_close=C("rename_dialog = false"),
        max_width="xs",
        full_width=True,
    ):
        mui.DialogTitle(["Rename conversation"])
        with mui.DialogContent():
            mui.TextField(
                label="Title",
                size="small",
                full_width=True,
                auto_focus=True,
                margin="dense",
                value=B("rename_text"),
                on_change=C("rename_text = $event.target.value"),
            )
        with mui.DialogActions():
            mui.Button("Cancel", on_click=C("rename_dialog = false"))
            mui.Button(
                "Save",
                variant="contained",
                on_click=C(app.ctrl.confirm_rename_session),
            )

    with mui.Dialog(
        open=B("import_dialog"),
        on_close=C("import_dialog = false"),
        max_width="sm",
        full_width=True,
    ):
        mui.DialogTitle(["Import conversation"])
        with mui.DialogContent():
            html.Input(
                type="file",
                accept=".json",
                multiple=True,
                on_change=C(
                    "window.trame.utils.vtk_prompt.uploadFiles($event.target.files,"
                    f" '{app.ctrl.trigger_name(app.ctrl.import_uploaded_files)}')"
                ),
                style={"padding": "8px 0"},
            )
        with mui.DialogActions():
            mui.Button("Close", on_click=C("import_dialog = false"))

    with mui.Dialog(
        open=B("delete_dialog"),
        on_close=C("delete_dialog = false"),
        max_width="xs",
        full_width=True,
    ):
        mui.DialogTitle(["Delete conversation"])
        with mui.DialogContent():
            mui.DialogContentText(
                [B("`Delete \u201c${delete_target_title}\u201d? This cannot be undone.`")]
            )
        with mui.DialogActions():
            mui.Button("Cancel", on_click=C("delete_dialog = false"))
            mui.Button(
                "Delete",
                color="error",
                variant="contained",
                on_click=C(app.ctrl.confirm_delete_session),
            )

    with mui.Dialog(
        open=B("bulk_delete_dialog"),
        on_close=C("bulk_delete_dialog = false"),
        max_width="xs",
        full_width=True,
    ):
        mui.DialogTitle(["Delete conversations"])
        with mui.DialogContent():
            mui.DialogContentText(
                [
                    B(
                        "`Delete ${selected_session_ids.length} conversation"
                        "${selected_session_ids.length === 1 ? '' : 's'}?"
                        " This cannot be undone.`"
                    )
                ]
            )
        with mui.DialogActions():
            mui.Button("Cancel", on_click=C("bulk_delete_dialog = false"))
            mui.Button(
                "Delete",
                color="error",
                variant="contained",
                on_click=C(app.ctrl.confirm_delete_selection),
            )


def build_conversation_history(app: Any) -> None:
    """Recents: header, bulk bar, conversation list and dialogs."""
    app.state.row_menu_id = ""
    app.state.row_menu_anchor = None

    _header(app)
    _selection_bar(app)
    html.Hr(style={"border": "none", "borderTop": "1px solid var(--vp-line)", "margin": 0})
    _list(app)
    _row_menu_popup(app)
    _dialogs(app)
