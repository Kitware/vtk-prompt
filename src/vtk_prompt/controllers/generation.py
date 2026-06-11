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
                # Let the client handle prompt selection based on mcp_url and UI mode
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
            push_code_snapshot(app, app.state.generated_code)

            # Update navigation after new conversation entry
            from .conversation import build_conversation_navigation

            build_conversation_navigation(app)

        app._conversation_loading = False
        success, exec_error = execute_with_renderer(app, app.state.generated_code)

        # If execution failed and vtk-mcp is configured, retry with the error fed back
        if not success and exec_error and getattr(app.state, "mcp_url", "").strip():
            logger.debug("Execution error, retrying with vtk-mcp: %s", exec_error)
            app.state.error_message = ""
            retry_result = app.prompt_client.query(
                execution_error=exec_error,
                api_key=app._get_api_key(),
                model=app._get_model(),
                base_url=app._get_base_url(),
                max_tokens=int(app.state.max_tokens),
                temperature=float(app.state.temperature),
                top_k=int(app.state.top_k),
                retry_attempts=1,
                provider=app.state.provider,
                custom_prompt=app.custom_prompt_data,
                ui_mode=True,
            )
            app.state.conversation = app.prompt_client.conversation
            if isinstance(retry_result, tuple) and len(retry_result) >= 2:
                _, retry_code = retry_result[0], retry_result[1]
                if retry_code:
                    app.state.generated_code = EXPLAIN_RENDERER + "\n" + retry_code
                    push_code_snapshot(app, app.state.generated_code)
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


def execute_with_renderer(app: Any, code_string: str) -> tuple[bool, str | None]:
    """Execute VTK code with our renderer. Returns (success, error_message)."""
    success, error_message = execute_vtk_code(code_string, app.renderer, app.render_window)

    if not success and error_message:
        app.state.error_message = error_message

    # Always update view
    try:
        app.ctrl.view_update()
    except Exception as e:
        logger.warning("View update error: %s", e)

    return success, error_message


def run_current_code(app: Any) -> None:
    """Execute the current (possibly hand-edited) code without calling the LLM.

    This is the "Run" action on the editable code panel: it takes whatever is in
    app.state.generated_code and renders it. A snapshot is recorded so a run after
    manual edits is reachable via undo/redo.
    """
    app.state.error_message = ""
    app.state.is_loading = True
    try:
        push_code_snapshot(app, app.state.generated_code)
        execute_with_renderer(app, app.state.generated_code)
    finally:
        app.state.is_loading = False


def push_code_snapshot(app: Any, code_string: str) -> None:
    """Record a code version on the history stack (drops any redo tail).

    No-op when the snapshot is identical to the current position, so repeated
    runs of unchanged code do not bloat the history.
    """
    history = list(app.state.code_history or [])
    pos = app.state.code_history_pos

    # If we branched off after an undo, discard the now-stale redo tail.
    if 0 <= pos < len(history) - 1:
        history = history[: pos + 1]

    if history and history[-1] == code_string:
        return  # nothing changed

    history.append(code_string)
    app.state.code_history = history
    app.state.code_history_pos = len(history) - 1


def undo_code(app: Any) -> None:
    """Step back to the previous code version and re-render it."""
    history = app.state.code_history or []
    pos = app.state.code_history_pos
    if pos > 0:
        pos -= 1
        app.state.code_history_pos = pos
        app.state.generated_code = history[pos]
        execute_with_renderer(app, app.state.generated_code)


def redo_code(app: Any) -> None:
    """Step forward to the next code version and re-render it."""
    history = app.state.code_history or []
    pos = app.state.code_history_pos
    if pos < len(history) - 1:
        pos += 1
        app.state.code_history_pos = pos
        app.state.generated_code = history[pos]
        execute_with_renderer(app, app.state.generated_code)


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
