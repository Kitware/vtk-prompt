"""
Content Layout Module.

This module provides the main content area layout for the VTK Prompt UI.
The content area contains code panels, VTK viewer, and prompt input.
"""

from typing import Any

from trame.widgets import code
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify
from trame_vtk.widgets import vtk as vtk_widgets

from ..langs import PYTHON_TEXTMATE


def build_content(layout: Any, app: Any) -> None:
    """Build the main content area with code panels and VTK viewer."""
    with layout.content:
        with vuetify.VContainer(
            classes="fluid fill-height", style="min-width: 100%; padding: 0!important;"
        ):
            with vuetify.VRow(rows=12, classes="fill-height px-4 pt-1 pb-1"):
                # Left column - Generated code view
                with vuetify.VCol(cols=6):
                    # Prompt input
                    with vuetify.VCard(classes="h-25"):
                        with vuetify.VCardText(classes="h-100"):
                            with html.Div(classes="d-flex"):
                                # Cloud models chip
                                vuetify.VChip(
                                    "☁️ {{ provider }}/{{ model }}",
                                    small=True,
                                    color="blue",
                                    text_color="white",
                                    label=True,
                                    classes="mb-2",
                                    v_show="use_cloud_models",
                                )
                                # Local models chip
                                vuetify.VChip(
                                    (
                                        "🏠 "
                                        "{{ local_base_url.replace('http://', '')"
                                        ".replace('https://', '') }}/"
                                        "{{ local_model }}"
                                    ),
                                    small=True,
                                    color="green",
                                    text_color="white",
                                    label=True,
                                    classes="mb-2",
                                    v_show="!use_cloud_models",
                                )
                                vuetify.VSpacer()
                                # API token warning chip
                                vuetify.VChip(
                                    "API token is required for cloud models.",
                                    small=True,
                                    color="error",
                                    text_color="white",
                                    label=True,
                                    classes="mb-2",
                                    v_show="use_cloud_models && !api_token.trim()",
                                    prepend_icon="mdi-alert",
                                )

                            with html.Div(classes="d-flex", style="height: calc(100% - 75px);"):
                                with vuetify.VBtn(
                                    variant="tonal",
                                    icon=True,
                                    rounded="0",
                                    disabled=("!can_navigate_left",),
                                    classes="h-auto mr-1",
                                    click=app.ctrl.navigate_conversation_left,
                                ):
                                    vuetify.VIcon("mdi-arrow-left-circle")
                                # Query input
                                vuetify.VTextarea(
                                    label="Describe VTK visualization",
                                    v_model=("query_text", ""),
                                    rows=4,
                                    variant="outlined",
                                    placeholder=("e.g., Create a red sphere with lighting"),
                                    persistent_placeholder=True,
                                    hide_details=True,
                                    no_resize=True,
                                )
                                with vuetify.VBtn(
                                    color=(
                                        "conversation_index ==="
                                        + " conversation_navigation.length - 1"
                                        + " ? 'success' : 'default'",
                                        "default",
                                    ),
                                    variant="tonal",
                                    icon=True,
                                    rounded="0",
                                    disabled=("!can_navigate_right",),
                                    click=app.ctrl.navigate_conversation_right,
                                ):
                                    vuetify.VIcon(
                                        "mdi-arrow-right-circle",
                                        v_show="conversation_index <"
                                        + " conversation_navigation.length - 1",
                                    )
                                    vuetify.VIcon(
                                        "mdi-message-plus",
                                        v_show="conversation_index ==="
                                        + " conversation_navigation.length - 1",
                                    )

                            # Generate button
                            vuetify.VBtn(
                                "Generate Code",
                                color="primary",
                                block=True,
                                loading=("trame__busy", False),
                                click=app.ctrl.generate_code,
                                classes="my-2",
                                v_show="!use_cloud_models || api_token.trim()",
                            )
                            vuetify.VBtn(
                                "Set API Key",
                                color="error",
                                block=True,
                                click=(
                                    "advanced_settings_open = true;"
                                    + " active_settings_tab = 'model';"
                                ),
                                classes="mb-2",
                                v_show="use_cloud_models && !api_token.trim()",
                            )

                    # Generated code panel (editable + re-runnable)
                    with vuetify.VCard(classes="h-75 mt-2"):
                        with vuetify.VCardTitle(
                            "Generated Code", classes="d-flex align-center"
                        ):
                            vuetify.VSpacer()
                            # Undo across code versions (generations, runs, edits)
                            with vuetify.VTooltip(text="Undo code change", location="bottom"):
                                with vuetify.Template(v_slot_activator="{ props }"):
                                    with vuetify.VBtn(
                                        click=app.ctrl.undo_code,
                                        icon=True,
                                        variant="text",
                                        density="compact",
                                        color="secondary",
                                        classes="mr-1",
                                        v_bind="props",
                                        disabled=("code_history_pos < 1",),
                                    ):
                                        vuetify.VIcon("mdi-undo")
                            with vuetify.VTooltip(text="Redo code change", location="bottom"):
                                with vuetify.Template(v_slot_activator="{ props }"):
                                    with vuetify.VBtn(
                                        click=app.ctrl.redo_code,
                                        icon=True,
                                        variant="text",
                                        density="compact",
                                        color="secondary",
                                        classes="mr-2",
                                        v_bind="props",
                                        disabled=(
                                            "code_history_pos >= code_history.length - 1",
                                        ),
                                    ):
                                        vuetify.VIcon("mdi-redo")
                            # Run the (possibly edited) code without the LLM
                            vuetify.VBtn(
                                "Run",
                                click=app.ctrl.run_current_code,
                                prepend_icon="mdi-play",
                                size="small",
                                color="primary",
                                variant="flat",
                                disabled=("is_loading || !generated_code",),
                            )
                        with vuetify.VCardText(style="height: calc(100% - 50px);"):
                            code.Editor(
                                v_model=("generated_code", ""),
                                language="python",
                                theme="vs",
                                textmate=("code_textmate", PYTHON_TEXTMATE),
                                completion=app.jedi_complete,
                                hover=app.jedi_hover,
                                options=(
                                    "code_editor_options",
                                    {
                                        "automaticLayout": True,
                                        "minimap": {"enabled": False},
                                        "fontSize": 13,
                                        "scrollBeyondLastLine": False,
                                        "lineNumbers": "on",
                                        "tabSize": 4,
                                        # Render hover/suggest widgets at the
                                        # document body so the surrounding
                                        # VCard overflow does not clip them;
                                        # sticky lets long docstrings scroll.
                                        "fixedOverflowWidgets": True,
                                        "hover": {"enabled": True, "sticky": True},
                                    },
                                ),
                                style="height: 100%; width: 100%;",
                            )

                # Right column - VTK viewer and prompt
                with vuetify.VCol(cols=6):
                    with vuetify.VRow(no_gutters=True, classes="fill-height"):
                        # Top: VTK render view
                        with vuetify.VCard(classes="h-75 w-100"):
                            with vuetify.VCardTitle("VTK Visualization", classes="d-flex"):
                                vuetify.VSpacer()
                                # Token usage display
                                with vuetify.VChip(
                                    small=True,
                                    color="secondary",
                                    text_color="white",
                                    v_show="input_tokens > 0 || output_tokens > 0",
                                    classes="mr-2",
                                    density="compact",
                                ):
                                    html.Span(
                                        "Tokens: In: {{ input_tokens }} | Out: {{ output_tokens }}"
                                    )
                                # VTK control buttons
                                with vuetify.VTooltip(
                                    text="Clear Scene",
                                    location="bottom",
                                ):
                                    with vuetify.Template(v_slot_activator="{ props }"):
                                        with vuetify.VBtn(
                                            click=app.ctrl.clear_scene,
                                            icon=True,
                                            color="secondary",
                                            v_bind="props",
                                            classes="mr-2",
                                            density="compact",
                                            variant="text",
                                        ):
                                            vuetify.VIcon("mdi-reload")
                                with vuetify.VTooltip(
                                    text="Reset Camera",
                                    location="bottom",
                                ):
                                    with vuetify.Template(v_slot_activator="{ props }"):
                                        with vuetify.VBtn(
                                            click=app.ctrl.reset_camera,
                                            icon=True,
                                            color="secondary",
                                            v_bind="props",
                                            classes="mr-2",
                                            density="compact",
                                            variant="text",
                                        ):
                                            vuetify.VIcon("mdi-camera-retake-outline")
                            with vuetify.VCardText(style="height: calc(100% - 50px);"):
                                # VTK render window
                                view = vtk_widgets.VtkRemoteView(
                                    app.render_window,
                                    ref="view",
                                    classes="w-100 h-100",
                                    interactor_settings=[
                                        (
                                            "SetInteractorStyle",
                                            ["vtkInteractorStyleTrackballCamera"],
                                        ),
                                    ],
                                )
                                app.ctrl.view_update = view.update
                                app.ctrl.view_reset_camera = view.reset_camera

                                # Register custom controller methods
                                app.ctrl.on_tab_change = app.on_tab_change

                                # Ensure initial render
                                view.update()

                        # Explanation panel
                        with vuetify.VCard(classes="h-25 w-100 mt-2"):
                            vuetify.VCardTitle("Explanation", classes="text-h6")
                            with vuetify.VCardText(style="height: calc(100% - 50px);"):
                                vuetify.VTextarea(
                                    v_model=("generated_explanation", ""),
                                    readonly=True,
                                    solo=True,
                                    hide_details=True,
                                    no_resize=True,
                                    classes="overflow-y-auto fill-height",
                                    placeholder="Explanation will appear here...",
                                    auto_grow=True,
                                    density="compact",
                                    style="overflow-y: auto;",
                                )

        vuetify.VAlert(
            closable=True,
            v_show=("error_message", ""),
            density="compact",
            type="error",
            text=("error_message",),
            classes="h-auto position-absolute bottom-0 align-self-center mb-1",
            style="width: 30%; z-index: 1000;",
            icon="mdi-alert-outline",
        )

        # Toast notification snackbar for validation warnings
        with vuetify.VSnackbar(
            v_model=("toast_visible",),
            timeout=5000,
            color=("toast_color",),
            location="top",
            multi_line=True,
        ):
            vuetify.VIcon("mdi-alert", classes="mr-2")
            html.Span("{{ toast_message }}")
            with vuetify.Template(v_slot_actions=""):
                vuetify.VBtn(
                    "Close",
                    color="white",
                    variant="text",
                    click="toast_visible = false",
                )
