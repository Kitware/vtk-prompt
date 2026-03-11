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

        with vuetify.VCardText(style="height: calc(100% - 50px); overflow-y: auto;"):
            # Show message when no history
            with vuetify.VAlert(
                text="No conversation history yet." + " Start by generating some VTK code!",
                type="info",
                variant="tonal",
                v_show="conversation_navigation.length === 0",
            ):
                pass

            # Conversation history list
            with vuetify.VCard(
                v_for=(
                    "(pair, idx) in (history_sort_order === 'newest'"
                    + " ? conversation_navigation.slice().reverse()"
                    + " : conversation_navigation)"
                ),
                key="idx",
                density="compact",
                v_show="conversation_navigation.length > 0",
                color=(
                    "conversation_index === idx ? 'primary' : 'secondary'",
                    "secondary",
                ),
                variant=(
                    "conversation_index === idx ? 'outlined' : 'default'",
                    "default",
                ),
            ):
                with vuetify.VCardText(
                    click=(app.ctrl.navigate_to_conversation, "[idx]"),
                    rounded=True,
                    classes="mb-2",
                ):
                    # User query preview
                    html.Span(
                        "{{ (pair.user.content.includes('</extra_instructions>')"
                        + " ? pair.user.content.split('</extra_instructions>')[1]"
                        + " : pair.user.content).trim() }}"
                    )
