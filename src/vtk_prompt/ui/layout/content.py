"""
Content Layout Module.

This module provides the main content area layout for the VTK Prompt UI.
The content area contains code panels, VTK viewer, and prompt input.
"""

from typing import Any

from trame.widgets import html, vuetify3 as vuetify
from trame_vtk.widgets import vtk as vtk_widgets


def build_content(layout: Any, app: Any) -> None:
    """Build the main content area with code panels and VTK viewer."""
    with layout.content:
        with vuetify.VContainer(
            classes="fluid fill-height", style="min-width: 100%; padding: 0!important;"
        ):
            with vuetify.VRow(rows=12, classes="fill-height px-4 pt-1 pb-1"):
                # Left column - Generated code view
                with vuetify.VCol(cols=7, classes="fill-height pa-0"):
                    with vuetify.VExpansionPanels(
                        v_model=("explanation_expanded", [0, 1]),
                        classes="fill-height pb-1 pr-1",
                        multiple=True,
                    ):
                        # Explanation panel
                        with vuetify.VExpansionPanel(
                            classes=("flex-grow-1 flex-shrink-0 d-flex flex-column pa-0 mt-0"),
                            style="max-height: 25%;",
                        ):
                            vuetify.VExpansionPanelTitle("Explanation", classes="text-h6")
                            with vuetify.VExpansionPanelText(
                                classes="fill-height flex-shrink-1",
                                style="overflow: hidden;",
                            ):
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

                        # Generated code panel
                        with vuetify.VExpansionPanel(
                            classes=(
                                "fill-height flex-grow-2 flex-shrink-0" + " d-flex flex-column mt-1"
                            ),
                            readonly=True,
                            style=(
                                "explanation_expanded.length > 1 ? "
                                + "'max-height: 75%;' : 'max-height: 95%;'",
                                "box-sizing: border-box;",
                            ),
                        ):
                            vuetify.VExpansionPanelTitle(
                                "Generated Code",
                                collapse_icon=False,
                                classes="text-h6",
                            )
                            with vuetify.VExpansionPanelText(
                                style="overflow: hidden; height: 90%;",
                                classes="flex-grow-1",
                            ):
                                vuetify.VTextarea(
                                    v_model=("generated_code", ""),
                                    readonly=True,
                                    solo=True,
                                    hide_details=True,
                                    no_resize=True,
                                    classes="overflow-y-auto fill-height",
                                    style="font-family: monospace;",
                                    placeholder="Generated VTK code will appear here...",
                                )

                # Right column - VTK viewer and prompt
                with vuetify.VCol(cols=5, classes="fill-height pa-0"):
                    with vuetify.VRow(no_gutters=True, classes="fill-height"):
                        # Top: VTK render view
                        with vuetify.VCol(
                            cols=12,
                            classes="flex-grow-1 flex-shrink-0 pa-0",
                            style="min-height: calc(100% - 256px);",
                        ):
                            with vuetify.VCard(classes="fill-height"):
                                with vuetify.VCardTitle(
                                    "VTK Visualization", classes="d-flex align-center"
                                ):
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
                                            "Tokens: In: {{ input_tokens }} | "
                                            "Out: {{ output_tokens }}"
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
                                with vuetify.VCardText(style="height: 90%;"):
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

                        # Bottom: Prompt input
                        with vuetify.VCol(
                            cols=12,
                            classes="flex-grow-0 flex-shrink-0",
                            style="height: 256px;",
                        ):
                            with vuetify.VCard(classes="fill-height"):
                                with vuetify.VCardText(
                                    classes="d-flex flex-column",
                                    style="height: 100%;",
                                ):
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

                                    with html.Div(
                                        classes="d-flex mb-2",
                                        style="height: 100%;",
                                    ):
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
                                            hide_details=True,
                                            no_resize=True,
                                            disabled=(
                                                "is_viewing_history",
                                                False,
                                            ),
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
                                            classes="h-auto ml-1",
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
                                        classes="mb-2",
                                        disabled=(
                                            "is_viewing_history ||"
                                            + " !query_text.trim() ||"
                                            + " (use_cloud_models && !api_token.trim())",
                                        ),
                                    )

        # Error alert
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
