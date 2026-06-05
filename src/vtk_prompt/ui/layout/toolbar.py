"""
Toolbar Layout Module.

This module provides the toolbar layout construction for the VTK Prompt UI.
The toolbar contains file controls, download buttons, and theme switcher.
"""

from typing import Any

from trame.widgets import vuetify3 as vuetify


def build_toolbar(layout: Any, app: Any) -> None:
    """Build the toolbar layout with file controls and settings."""
    with layout.toolbar as toolbar:
        drawer_icon = toolbar.children[0]
        drawer_icon.hide()

        vuetify.VSpacer()

        # Settings buttons
        with vuetify.VTooltip(
            text="Load or download files",
            location="bottom",
        ):
            with vuetify.Template(v_slot_activator="{ props }"):
                with vuetify.VBtn(
                    icon=True,
                    v_bind="props",
                    click="advanced_settings_open = true; active_settings_tab = 'files';",
                    classes="mr-4",
                    color="primary",
                ):
                    vuetify.VIcon("mdi-file-cog-outline")

        with vuetify.VTooltip(
            text="Change model settings",
            location="bottom",
        ):
            with vuetify.Template(v_slot_activator="{ props }"):
                with vuetify.VBtn(
                    icon=True,
                    v_bind="props",
                    click="advanced_settings_open = true; active_settings_tab = 'model';",
                    classes="mr-4",
                    color="primary",
                ):
                    vuetify.VIcon("mdi-brain")

        with vuetify.VTooltip(
            text="Advanced settings",
            location="bottom",
        ):
            with vuetify.Template(v_slot_activator="{ props }"):
                with vuetify.VBtn(
                    icon=True,
                    v_bind="props",
                    click="advanced_settings_open = true; active_settings_tab = 'advanced';",
                    classes="mr-4",
                    color="primary",
                ):
                    vuetify.VIcon("mdi-cog-outline")

        # Theme switcher
        vuetify.VSwitch(
            v_model=("theme_mode", "light"),
            hide_details=True,
            classes="mr-2",
            true_value="light",
            false_value="dark",
            append_icon=("theme_mode === 'light' ? 'mdi-weather-sunny' : 'mdi-weather-night'",),
        )
