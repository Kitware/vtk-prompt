"""
Toolbar Layout Module.

This module provides the toolbar layout construction for the VTK Prompt UI.
The toolbar contains file controls, download buttons, and theme switcher.
"""

from typing import Any

from trame.widgets import vuetify3 as vuetify


def build_toolbar(layout: Any) -> None:
    """Build the toolbar layout with file controls and settings."""
    with layout.toolbar:
        vuetify.VSpacer()

        # Conversation file input
        with vuetify.VTooltip(
            text=("conversation_file", "No file loaded"),
            location="bottom",
            disabled=("!conversation_object",),
        ):
            with vuetify.Template(v_slot_activator="{ props }"):
                vuetify.VFileInput(
                    label="Conversation File",
                    v_model=("conversation_object", None),
                    accept=".json",
                    variant="solo",
                    density="compact",
                    prepend_icon="mdi-forum-outline",
                    hide_details="auto",
                    classes="py-1 pr-1 mr-2 text-truncate",
                    open_on_focus=False,
                    clearable=False,
                    v_bind="props",
                    rules=["[utils.vtk_prompt.rules.json_file]"],
                    color="primary",
                    style="max-width: 25%;",
                )

        # Auto-run toggle button
        with vuetify.VTooltip(
            text=(
                "auto_run_conversation_file ? "
                + "'Auto-run conversation files on load' : "
                + "'Do not auto-run conversation files on load'",
                "Auto-run conversation files on load",
            ),
            location="bottom",
        ):
            with vuetify.Template(v_slot_activator="{ props }"):
                with vuetify.VBtn(
                    icon=True,
                    v_bind="props",
                    click="auto_run_conversation_file = !auto_run_conversation_file",
                    classes="mr-2",
                    color="primary",
                ):
                    vuetify.VIcon(
                        "mdi-autorenew",
                        v_show="auto_run_conversation_file",
                    )
                    vuetify.VIcon(
                        "mdi-autorenew-off",
                        v_show="!auto_run_conversation_file",
                    )

        # Download conversation button
        with vuetify.VTooltip(
            text="Download conversation file",
            location="bottom",
        ):
            with vuetify.Template(v_slot_activator="{ props }"):
                with vuetify.VBtn(
                    icon=True,
                    v_bind="props",
                    disabled=("!conversation",),
                    click="utils.download("
                    + "`vtk-prompt_${provider}_${model}.json`,"
                    + "trigger('save_conversation'),"
                    + "'application/json'"
                    + ")",
                    classes="mr-2",
                    color="primary",
                    density="compact",
                ):
                    vuetify.VIcon("mdi-file-download-outline")

        # Download config button
        with vuetify.VTooltip(
            text="Download config file",
            location="bottom",
        ):
            with vuetify.Template(v_slot_activator="{ props }"):
                with vuetify.VBtn(
                    icon=True,
                    v_bind="props",
                    click="utils.download("
                    + "`vtk-prompt_config.yml`,"
                    + "trigger('save_config'),"
                    + "'application/x-yaml'"
                    + ")",
                    classes="mr-4",
                    color="primary",
                    density="compact",
                ):
                    vuetify.VIcon("mdi-content-save-cog-outline")

        # Theme switcher
        vuetify.VSwitch(
            v_model=("theme_mode", "light"),
            hide_details=True,
            density="compact",
            classes="mr-2",
            true_value="light",
            false_value="dark",
            append_icon=("theme_mode === 'light' ? 'mdi-weather-sunny' : 'mdi-weather-night'",),
        )
