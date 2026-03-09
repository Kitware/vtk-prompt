"""
Generation Controllers Module.

This module provides controller functions for VTK code generation, execution,
and scene manipulation in the VTK Prompt UI.
"""

from typing import Any

from .. import get_logger
from ..rendering import clear_scene as clear_vtk_scene
from ..rendering import (
    execute_vtk_code,
)
from ..rendering import reset_camera as reset_vtk_camera

logger = get_logger(__name__)

EXPLAIN_RENDERER = (
    "# renderer is a vtkRenderer injected by this webapp"
    + "\n"
    + "# Use your own vtkRenderer in your application"
)


def generate_code(app: Any) -> None:
    """Generate VTK code from user query."""
    generate_and_execute_code(app)


def generate_and_execute_code(app: Any) -> None:
    """Generate VTK code using AI API and execute it."""
    app.state.is_loading = True
    app.state.error_message = ""

    try:
        if not app._conversation_loading:
            # Use custom prompt if provided, otherwise use built-in YAML prompts
            if app.custom_prompt_data:
                # Use the query text directly when using custom prompts
                enhanced_query = app.state.query_text
                logger.debug("Using custom prompt file")
            else:
                # Let the client handle prompt selection based on RAG and UI mode
                enhanced_query = app.state.query_text
                logger.debug("Using UI mode - client will select appropriate prompt")

            # Reinitialize client with current settings
            app._init_prompt_client()
            if hasattr(app.state, "error_message") and app.state.error_message:
                return

            result = app.prompt_client.query(
                enhanced_query,
                api_key=app._get_api_key(),
                model=app._get_model(),
                base_url=app._get_base_url(),
                max_tokens=int(app.state.max_tokens),
                temperature=float(app.state.temperature),
                top_k=int(app.state.top_k),
                rag=app.state.use_rag,
                retry_attempts=int(app.state.retry_attempts),
                provider=app.state.provider,
                custom_prompt=app.custom_prompt_data,
                ui_mode=True,  # This tells the client to use UI-specific components
            )
            # Keep UI in sync with conversation
            app.state.conversation = app.prompt_client.conversation

            # Handle result with optional validation warnings
            validation_warnings: list[str] = []
            if isinstance(result, tuple):
                if len(result) == 4:
                    # Result includes validation warnings
                    generated_explanation, generated_code, usage, validation_warnings = result
                elif len(result) == 3:
                    generated_explanation, generated_code, usage = result
                else:
                    generated_explanation = str(result)
                    generated_code = ""
                    usage = None

                if usage:
                    app.state.input_tokens = usage.prompt_tokens
                    app.state.output_tokens = usage.completion_tokens
                else:
                    app.state.input_tokens = 0
                    app.state.output_tokens = 0
            else:
                # Handle string result
                generated_explanation = str(result)
                generated_code = ""
                app.state.input_tokens = 0
                app.state.output_tokens = 0

            # Display validation warnings as toast notifications
            if validation_warnings:
                for warning in validation_warnings:
                    trigger_warning_toast(app, warning)

            app.state.generated_explanation = generated_explanation
            app.state.generated_code = EXPLAIN_RENDERER + "\n" + generated_code

            # Update navigation after new conversation entry
            from .conversation import build_conversation_navigation

            build_conversation_navigation(app)

        app._conversation_loading = False
        # Execute the generated code using the existing run_code method
        # But we need to modify it to work with our renderer
        execute_with_renderer(app, app.state.generated_code)
    except ValueError as e:
        if "max_tokens" in str(e):
            app.state.error_message = (
                f"{str(e)} Current: {app.state.max_tokens}. Try increasing max tokens."
            )
        else:
            app.state.error_message = f"Error generating code: {str(e)}"
    except Exception as e:
        app.state.error_message = f"Error generating code: {str(e)}"
    finally:
        app.state.is_loading = False


def execute_with_renderer(app: Any, code_string: str) -> None:
    """Execute VTK code with our renderer."""
    success, error_message = execute_vtk_code(code_string, app.renderer, app.render_window)

    if not success and error_message:
        app.state.error_message = error_message

    # Always update view
    try:
        app.ctrl.view_update()
    except Exception as e:
        logger.warning("View update error: %s", e)


def clear_scene(app: Any) -> None:
    """Clear the VTK scene and restore default axes."""
    try:
        clear_vtk_scene(app.renderer, app.render_window)
        app.ctrl.view_update()
    except Exception as e:
        logger.error("Error clearing scene: %s", e)


def reset_camera(app: Any) -> None:
    """Reset camera view."""
    try:
        reset_vtk_camera(app.renderer, app.render_window)
        app.ctrl.view_update()
    except Exception as e:
        logger.error("Error resetting camera: %s", e)


def trigger_warning_toast(app: Any, message: str) -> None:
    """Display a warning toast notification.

    Args:
        app: VTKPromptApp instance
        message: Warning message to display
    """
    app.state.toast_message = message
    app.state.toast_color = "warning"
    app.state.toast_visible = True
    logger.warning("Toast notification: %s", message)
