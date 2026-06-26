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


def build_conversation_history(app: Any) -> None:
    """Build the Recents drawer: conversations, plus the active one's prompts."""
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

            # Sort order (newest / oldest).
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

            # Favorites filter (show only hearted conversations).
            with vuetify.VTooltip(text="Toggle favorites filter", location="bottom"):
                with vuetify.Template(v_slot_activator="{ props }"):
                    vuetify.VBtn(
                        icon=(
                            "history_filter_mode === 'favorites'"
                            + " ? 'mdi-heart' : 'mdi-heart-off'",
                            "mdi-format-list-bulleted",
                        ),
                        click=(
                            "history_filter_mode = "
                            + "(history_filter_mode === 'all') ? 'favorites' : 'all'"
                        ),
                        variant="text",
                        density="compact",
                        color=(
                            "history_filter_mode === 'favorites' ? 'red' : 'primary'",
                            "primary",
                        ),
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
                        # Favorite (heart) toggle; its own click, does not switch.
                        vuetify.VBtn(
                            icon=(
                                "s.pinned ? 'mdi-heart' : 'mdi-heart-outline'",
                                "mdi-heart-outline",
                            ),
                            click=(app.ctrl.toggle_pin_session, "[s.id]"),
                            size="x-small",
                            variant="text",
                            color=("s.pinned ? 'red' : 'grey'", "grey"),
                            classes="mr-1",
                        )
                        # Title: click to switch to this conversation.
                        html.Span(
                            "{{ s.title }}",
                            click=(app.ctrl.switch_session, "[s.id]"),
                            classes="flex-grow-1 text-truncate",
                            style="cursor: pointer;",
                        )

                    # The active conversation's follow-up prompts (its turns),
                    # clickable to jump to that point in the thread.
                    with html.Div(
                        v_if="s.active && conversation_navigation.length > 0",
                        classes="ml-6 mt-1 mb-1",
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
