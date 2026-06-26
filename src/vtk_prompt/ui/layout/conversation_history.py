"""Recents drawer: conversations to switch between, with the active one's turns."""

from typing import Any

from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

# Strip the extra-instructions prefix and a leading "Request:" for a clean preview.
_PROMPT_PREVIEW = (
    "(pair.user.content.includes('</extra_instructions>')"
    " ? pair.user.content.split('</extra_instructions>')[1]"
    " : pair.user.content).trim().replace(/^Request:\\s*/i, '')"
)


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


def build_conversation_history(app: Any) -> None:
    """Build the Recents drawer: conversations, plus the active one's prompts."""
    with vuetify.VCard(classes="w-100", flat=True):
        _header(app)
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
                            style="cursor: pointer;",
                        )
                        _row_menu(app)

                    # The active conversation's follow-up prompts (its turns).
                    with html.Div(
                        v_if="s.active && conversation_navigation.length > 0",
                        classes="ml-4 mt-1 mb-1",
                    ):
                        with html.Div(
                            v_for="(pair, idx) in conversation_navigation",
                            key="'turn-' + idx",
                            click=(app.ctrl.navigate_to_conversation, "[idx]"),
                            classes=(
                                "(conversation_index === idx"
                                + " ? 'text-primary font-weight-medium' : 'text-medium-emphasis')"
                                + " + ' text-caption text-truncate py-1 px-2'",
                                "text-caption",
                            ),
                            style=(
                                "'cursor: pointer; border-left: 2px solid '"
                                + " + (conversation_index === idx"
                                + " ? 'rgb(var(--v-theme-primary))' : 'transparent')",
                                "cursor: pointer;",
                            ),
                        ):
                            html.Span("{{ idx + 1 }}. {{ " + _PROMPT_PREVIEW + " }}")
        _dialogs(app)
