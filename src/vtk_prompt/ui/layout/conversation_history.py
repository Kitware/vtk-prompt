# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, The PyTK developers.
# Distributed under a BSD-3-Clause license.
# See the LICENSE file for details.

"""Conversation History Component for VTK Prompt UI."""

from typing import Any

from trame.widgets import html
from trame.widgets import vuetify3 as vuetify


def build_conversation_history(app: Any) -> None:
    """Build the conversation history component with clickable conversation cards."""
    # Bottom: Conversation History
    with vuetify.VCard(classes="h-75 w-100 mt-2"):
        with vuetify.VCardTitle("Conversation History", classes="d-flex align-center"):
            vuetify.VSpacer()

            # Sort toggle button
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
                            + "(history_sort_order === 'newest')"
                            + " ? 'oldest' : 'newest'"
                        ),
                        variant="text",
                        density="compact",
                        color="primary",
                        disabled=("conversation_navigation.length === 0", False),
                        v_bind="props",
                    )

            # Filter toggle button
            with vuetify.VTooltip(text="Toggle favorites filter", location="bottom"):
                with vuetify.Template(v_slot_activator="{ props }"):
                    vuetify.VBtn(
                        icon=(
                            "history_filter_mode === 'favorites'"
                            + " ? 'mdi-heart' : 'mdi-heart-off'",
                            "mdi-format-list-bulleted",
                        ),
                        click=(
                            "history_filter_mode = (history_filter_mode === 'all')"
                            + " ? 'favorites' : 'all'"
                        ),
                        variant="text",
                        density="compact",
                        color=(
                            "history_filter_mode === 'favorites' ? 'red' : 'primary'",
                            "primary",
                        ),
                        disabled=(
                            "conversation_navigation.length === 0"
                            + " || favorited_conversations.length === 0",
                            False,
                        ),
                        v_bind="props",
                    )

        with vuetify.VCardText(style="height: calc(100% - 50px); overflow-y: auto;"):
            # Show message when no history
            vuetify.VAlert(
                text="No conversation history yet." + " Start by generating some VTK code!",
                type="info",
                variant="tonal",
                v_show="conversation_navigation.length === 0",
            )

            # Conversation history list
            with vuetify.VCard(
                v_for=(
                    "item in (history_sort_order === 'newest'"
                    + " ? conversation_navigation.slice().reverse()"
                    + ".map((pair, idx) => ({pair, originalIndex: "
                    + "conversation_navigation.length - 1 - idx}))"
                    + " : conversation_navigation.map((pair, idx) => "
                    + "({pair, originalIndex: idx})))"
                    + ".filter(item => history_filter_mode === 'all' || "
                    + "favorited_conversations.includes(item.originalIndex))"
                ),
                key="item.originalIndex",
                density="compact",
                v_show="conversation_navigation.length > 0",
                color=(
                    "conversation_index === item.originalIndex" + " ? 'primary' : 'secondary'",
                    "secondary",
                ),
                variant=(
                    "conversation_index === item.originalIndex" + " ? 'outlined' : 'default'",
                    "default",
                ),
            ):
                # Track favorited prompts
                with vuetify.VCardTitle(classes="text-end"):
                    vuetify.VIcon(
                        click=(
                            app.ctrl.toggle_favorite_conversation,
                            "[item.originalIndex]",
                        ),
                        icon=(
                            "favorited_conversations.includes(item.originalIndex) ? "
                            + "'mdi-heart' : 'mdi-heart-outline'",
                            "mdi-heart-outline",
                        ),
                        size="small",
                        color=(
                            "favorited_conversations.includes(item.originalIndex) ? "
                            + "'red' : 'grey'",
                            "grey",
                        ),
                    )
                with vuetify.VCardText(
                    click=(
                        app.ctrl.navigate_to_conversation,
                        "[item.originalIndex]",
                    ),
                    rounded=True,
                    classes="mb-2",
                ):
                    # User query preview
                    html.Span(
                        "{{ (item.pair.user.content.includes('</extra_instructions>')"
                        + " ? item.pair.user.content.split('</extra_instructions>')[1]"
                        + " : item.pair.user.content).trim() }}"
                    )
