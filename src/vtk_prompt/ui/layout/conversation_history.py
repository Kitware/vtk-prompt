"""Recents drawer: the list of conversations (sessions) the user can switch between."""

from typing import Any

from trame.widgets import html
from trame.widgets import vuetify3 as vuetify


def build_conversation_history(app: Any) -> None:
    """Build the Recents drawer: one row per conversation, pinned/newest first."""
    with vuetify.VCard(classes="w-100", flat=True):
        with vuetify.VCardTitle("Recents", classes="d-flex align-center"):
            vuetify.VSpacer()

            # New conversation: archive the current one and start a fresh thread.
            with vuetify.VTooltip(text="New conversation", location="bottom"):
                with vuetify.Template(v_slot_activator="{ props }"):
                    vuetify.VBtn(
                        icon="mdi-plus",
                        click=app.ctrl.start_new_conversation,
                        variant="text",
                        density="compact",
                        color="primary",
                        # Already on a fresh, empty conversation -> nothing to add.
                        disabled=("conversation_navigation.length === 0", True),
                        v_bind="props",
                    )

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
                        # Pin / unpin this conversation (its own click; no switch).
                        vuetify.VBtn(
                            icon=(
                                "s.pinned ? 'mdi-pin' : 'mdi-pin-outline'",
                                "mdi-pin-outline",
                            ),
                            click=(app.ctrl.toggle_pin_session, "[s.id]"),
                            size="x-small",
                            variant="text",
                            color=("s.pinned ? 'primary' : 'grey'", "grey"),
                            classes="mr-1",
                        )
                        # Title: click to switch to this conversation.
                        html.Span(
                            "{{ s.title }}",
                            click=(app.ctrl.switch_session, "[s.id]"),
                            classes="flex-grow-1 text-truncate",
                            style="cursor: pointer;",
                        )
