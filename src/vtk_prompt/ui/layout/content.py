"""
Content Layout Module.

This module provides the main content area layout for the VTK Prompt UI.
The content area contains code panels, VTK viewer, and prompt input.
"""

from typing import Any

from trame.widgets import html
from trame.widgets import vuetify3 as vuetify
from trame_code.widgets import code  # type: ignore[import-not-found]
from trame_vtk.widgets import vtk as vtk_widgets

from .conversation_history import build_conversation_history


def build_content(layout: Any, app: Any) -> None:
    """Build the main content area with code panels and VTK viewer."""
    with layout.content:
        with vuetify.VContainer(
            classes="fluid fill-height", style="min-width: 100%; padding: 0!important;"
        ):
            with vuetify.VRow(rows=12, classes="fill-height px-4 pt-1 pb-1"):
                # Left column - Prompt and conversation history
                with vuetify.VCol(cols=3, classes="fill-height"):
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
                                # Query input
                                vuetify.VTextarea(
                                    label="Describe VTK visualization",
                                    v_model=("query_text", "Create a red sphere with lighting"),
                                    rows=4,
                                    variant="outlined",
                                    hide_details=True,
                                    no_resize=True,
                                    clearable=True,
                                )
                            # Generate button
                            vuetify.VBtn(
                                "Generate Code",
                                color="primary",
                                block=True,
                                loading=("trame__busy", False),
                                click=app.ctrl.generate_code,
                                classes="my-2",
                                v_show="api_token.trim()",
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

                    # Bottom: Conversation History
                    build_conversation_history(app)

                # Middle column - Generated code view
                with vuetify.VCol(cols=4):
                    # Generated code panel
                    with vuetify.VCard(readonly=True, classes="h-100"):
                        with vuetify.VCardTitle(classes="d-flex align-center"):
                            html.Span("Generated Code")
                            vuetify.VSpacer()
                            with vuetify.VTooltip(text="Download as .py", location="bottom"):
                                with vuetify.Template(v_slot_activator="{ props }"):
                                    with vuetify.VBtn(
                                        icon=True,
                                        density="compact",
                                        variant="text",
                                        v_bind="props",
                                        click=(
                                            "trame.utils.vtk_prompt.download_generated_code(trame)"
                                        ),
                                        disabled=("!generated_code",),
                                    ):
                                        vuetify.VIcon("mdi-download")
                        with vuetify.VCardText(style="height: calc(100% - 50px);"):
                            code.Editor(
                                model_value=("generated_code", ""),
                                language="python",
                                theme=("theme_mode === 'dark' ? 'vs-dark' : 'vs'",),
                                options=("editor_options",),
                                style="width: 100%; height: calc(100% - 75px);",
                            )

                # Right column - VTK viewer and prompt
                with vuetify.VCol(cols=5, classes="mb-2"):
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
