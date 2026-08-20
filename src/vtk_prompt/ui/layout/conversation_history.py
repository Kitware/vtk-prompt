"""Recents drawer: the list of conversations to switch between."""

from typing import Any

from trame.widgets import html
from trame.widgets import vuetify3 as vuetify


def _header(app: Any) -> None:
    with vuetify.VCardTitle("Recents", classes="d-flex align-center"):
        vuetify.VSpacer()
        with vuetify.VTooltip(text="New conversation", location="bottom"):
            with vuetify.Template(v_slot_activator="{ props }"):
                vuetify.VBtn(
                    icon="mdi-plus",
                    click=app.ctrl.start_new_conversation,
                    variant="text",
                    density="compact",
                    color="primary",
                    disabled=("conversation_navigation.length === 0", True),
                    v_bind="props",
                )
        with vuetify.VTooltip(text="Import conversation", location="bottom"):
            with vuetify.Template(v_slot_activator="{ props }"):
                vuetify.VBtn(
                    icon="mdi-tray-arrow-up",
                    click="import_dialog = true",
                    variant="text",
                    density="compact",
                    color="primary",
                    v_bind="props",
                )
        with vuetify.VTooltip(text="Toggle sort order", location="bottom"):
            with vuetify.Template(v_slot_activator="{ props }"):
                vuetify.VBtn(
                    icon=(
                        "history_sort_order === 'newest'"
                        + " ? 'mdi-sort-descending' : 'mdi-sort-ascending'",
                        "mdi-sort-descending",
                    ),
                    click=(
                        "history_sort_order = "
                        + "(history_sort_order === 'newest') ? 'oldest' : 'newest'"
                    ),
                    variant="text",
                    density="compact",
                    color="primary",
                    disabled=("sessions_list.length === 0", False),
                    v_bind="props",
                )


def _selection_bar(app: Any) -> None:
    """Bulk actions for the current selection, shown once anything is checked.

    Pin and Unpin are separate rather than one toggle: a mixed selection has no
    sensible thing to toggle to. Rename is absent by nature - there is no useful
    way to give many conversations one name.
    """
    with html.Div(
        v_show="selected_session_ids.length > 0",
        classes="d-flex align-center flex-wrap px-4 pb-2",
    ):
        html.Span(
            "{{ selected_session_ids.length }} selected",
            classes="text-caption text-medium-emphasis mr-2",
        )
        vuetify.VBtn(
            "All",
            click=app.ctrl.select_all_sessions,
            variant="text",
            size="x-small",
            disabled=(
                "selected_session_ids.length === sessions_list.length",
                False,
            ),
        )
        vuetify.VBtn(
            "None",
            click="selected_session_ids = []",
            variant="text",
            size="x-small",
            disabled=("selected_session_ids.length === 0", True),
        )
        vuetify.VSpacer()
        # Each is enabled only when it would actually change something: Pin when
        # some selected row is unpinned, Unpin when some selected row is pinned.
        # `some` over an empty selection is false, so these also cover the
        # nothing-selected case without a separate length check.
        with vuetify.VTooltip(text="Pin selected", location="bottom"):
            with vuetify.Template(v_slot_activator="{ props }"):
                vuetify.VBtn(
                    icon="mdi-pin",
                    click=(app.ctrl.set_selection_pinned, "[true]"),
                    variant="text",
                    density="compact",
                    size="small",
                    disabled=(
                        "!sessions_list.some("
                        "s => selected_session_ids.includes(s.id) && !s.pinned)",
                        True,
                    ),
                    v_bind="props",
                )
        with vuetify.VTooltip(text="Unpin selected", location="bottom"):
            with vuetify.Template(v_slot_activator="{ props }"):
                vuetify.VBtn(
                    icon="mdi-pin-off",
                    click=(app.ctrl.set_selection_pinned, "[false]"),
                    variant="text",
                    density="compact",
                    size="small",
                    disabled=(
                        "!sessions_list.some("
                        "s => selected_session_ids.includes(s.id) && s.pinned)",
                        True,
                    ),
                    v_bind="props",
                )
        with vuetify.VTooltip(text="Export selected", location="bottom"):
            with vuetify.Template(v_slot_activator="{ props }"):
                vuetify.VBtn(
                    icon="mdi-tray-arrow-down",
                    click=(
                        "window.trame.utils.vtk_prompt.exportSessions("
                        "selected_session_ids, sessions_list)"
                    ),
                    variant="text",
                    density="compact",
                    size="small",
                    disabled=("selected_session_ids.length === 0", True),
                    v_bind="props",
                )
        with vuetify.VTooltip(text="Delete selected", location="bottom"):
            with vuetify.Template(v_slot_activator="{ props }"):
                vuetify.VBtn(
                    icon="mdi-delete",
                    click="bulk_delete_dialog = true",
                    variant="text",
                    density="compact",
                    size="small",
                    color="error",
                    disabled=("selected_session_ids.length === 0", True),
                    v_bind="props",
                )


def _row_menu(app: Any) -> None:
    with vuetify.VMenu(location="bottom end"):
        with vuetify.Template(v_slot_activator="{ props }"):
            vuetify.VBtn(
                icon="mdi-dots-vertical",
                size="x-small",
                variant="text",
                color="grey",
                v_bind="props",
            )
        with vuetify.VList(density="compact"):
            with vuetify.VListItem(click=(app.ctrl.toggle_pin_session, "[s.id]")):
                vuetify.VListItemTitle("{{ s.pinned ? 'Unpin' : 'Pin' }}")
            with vuetify.VListItem(
                click="rename_target_id = s.id; rename_text = s.title; rename_dialog = true"
            ):
                vuetify.VListItemTitle("Rename")
            with vuetify.VListItem(
                click="window.trame.utils.vtk_prompt.exportSession(s.id, s.title)"
            ):
                vuetify.VListItemTitle("Export")
            with vuetify.VListItem(
                click=(
                    "delete_target_id = s.id; delete_target_title = s.title;"
                    + " delete_dialog = true"
                )
            ):
                vuetify.VListItemTitle("Delete")


def _dialogs(app: Any) -> None:
    # Rename
    with vuetify.VDialog(v_model=("rename_dialog", False), max_width="420"):
        with vuetify.VCard():
            vuetify.VCardTitle("Rename conversation")
            with vuetify.VCardText():
                vuetify.VTextField(
                    v_model=("rename_text", ""),
                    label="Title",
                    autofocus=True,
                    hide_details=True,
                    keydown_enter=app.ctrl.confirm_rename_session,
                )
            with vuetify.VCardActions():
                vuetify.VSpacer()
                vuetify.VBtn("Cancel", click="rename_dialog = false", variant="text")
                vuetify.VBtn(
                    "Save",
                    click=app.ctrl.confirm_rename_session,
                    color="primary",
                    variant="text",
                )
    # Import
    with vuetify.VDialog(v_model=("import_dialog", False), max_width="480"):
        with vuetify.VCard():
            vuetify.VCardTitle("Import conversation")
            with vuetify.VCardText():
                vuetify.VFileUpload(
                    label="Choose a .json conversation file",
                    v_model=("uploaded_files", None),
                    accept=".json",
                    multiple=True,
                    hide_details="auto",
                    density="compact",
                    color="teal-lighten-5",
                )
            with vuetify.VCardActions():
                vuetify.VSpacer()
                vuetify.VBtn("Close", click="import_dialog = false", variant="text")
    # Delete
    with vuetify.VDialog(v_model=("delete_dialog", False), max_width="420"):
        with vuetify.VCard():
            vuetify.VCardTitle("Delete conversation")
            vuetify.VCardText(
                "Delete \u201c{{ delete_target_title }}\u201d? This cannot be undone."
            )
            with vuetify.VCardActions():
                vuetify.VSpacer()
                vuetify.VBtn("Cancel", click="delete_dialog = false", variant="text")
                vuetify.VBtn(
                    "Delete",
                    click=app.ctrl.confirm_delete_session,
                    color="error",
                    variant="text",
                )


def _bulk_delete_dialog(app: Any) -> None:
    with vuetify.VDialog(v_model=("bulk_delete_dialog", False), max_width="420"):
        with vuetify.VCard():
            vuetify.VCardTitle("Delete conversations")
            vuetify.VCardText(
                "Delete {{ selected_session_ids.length }} conversation"
                "{{ selected_session_ids.length === 1 ? '' : 's' }}?"
                " This cannot be undone."
            )
            with vuetify.VCardActions():
                vuetify.VSpacer()
                vuetify.VBtn(
                    "Cancel", click="bulk_delete_dialog = false", variant="text"
                )
                vuetify.VBtn(
                    "Delete",
                    click=app.ctrl.confirm_delete_selection,
                    color="error",
                    variant="text",
                )


def build_conversation_history(app: Any) -> None:
    """Build the Recents drawer: conversations, plus the active one's prompts."""
    with vuetify.VCard(classes="w-100", flat=True):
        _header(app)
        _selection_bar(app)
        with vuetify.VCardText(style="overflow-y: auto;"):
            vuetify.VAlert(
                text="No conversations yet. Start by generating some VTK code!",
                type="info",
                variant="tonal",
                v_show="sessions_list.length === 0",
            )
            with vuetify.VList(density="compact", nav=True):
                with vuetify.VListItem(
                    v_for="s in sessions_list",
                    key="s.id",
                    active=("s.active", False),
                    color="primary",
                ):
                    with html.Div(classes="d-flex align-center w-100"):
                        # Always visible rather than revealed on hover: hiding it
                        # cost discoverability, and `visibility: hidden` drops the
                        # control out of the tab order entirely, making bulk
                        # selection keyboard-inaccessible. Sizing is left to
                        # Vuetify - constraining the width collapses the control;
                        # the title's min-width:0 is what prevents overlap.
                        vuetify.VCheckbox(
                            v_model=("selected_session_ids", []),
                            value=("s.id",),
                            density="compact",
                            hide_details=True,
                            color="primary",
                            # Round rather than square: purely the icon, so this
                            # stays a real checkbox input with role="checkbox" -
                            # screen readers and keyboard behaviour are
                            # unchanged, despite round conventionally meaning
                            # "radio, pick one".
                            false_icon="mdi-circle-outline",
                            true_icon="mdi-check-circle",
                            # font-size sizes the glyph; --v-selection-control-size
                            # sizes the box (and so the row height). Independent:
                            # the title is a sibling, so neither touches it.
                            # Vuetify's default control size is 40px - 32 keeps
                            # rows close to their original height while the
                            # smaller glyph does the visual work.
                            style=(
                                "font-size: 9px;"
                                " --v-selection-control-size: 32px;"
                            ),
                            classes="mr-1 flex-shrink-0",
                        )
                        vuetify.VIcon(
                            "mdi-pin",
                            size="x-small",
                            color="primary",
                            classes="mr-1",
                            v_show="s.pinned",
                        )
                        html.Span(
                            "{{ s.title }}",
                            click=(app.ctrl.switch_session, "[s.id]"),
                            classes="flex-grow-1 text-truncate",
                            # min-width:0 is what makes text-truncate actually
                            # clip inside a flex row; without it the span keeps
                            # its full intrinsic width and pushes into whatever
                            # sits beside it.
                            style="cursor: pointer; min-width: 0;",
                        )
                        # This conversation is generating, wherever you are.
                        vuetify.VProgressCircular(
                            indeterminate=True,
                            size="14",
                            width="2",
                            color="primary",
                            classes="ml-1",
                            v_show="s.generating",
                        )
                        # A conversation that finished while you were elsewhere.
                        with vuetify.VTooltip(text="New result", location="left"):
                            with vuetify.Template(v_slot_activator="{ props }"):
                                vuetify.VIcon(
                                    "mdi-circle-medium",
                                    v_bind="props",
                                    size="small",
                                    color="primary",
                                    v_show="s.unseen && !s.generating",
                                )
                        _row_menu(app)
        _dialogs(app)
        _bulk_delete_dialog(app)
