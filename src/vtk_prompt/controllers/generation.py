"""
Generation Controllers Module.

This module provides controller functions for VTK code generation, execution,
and scene manipulation in the VTK Prompt UI.
"""

import asyncio
from typing import Any

from trame.app import asynchronous

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


def _unpack_result(result: Any) -> tuple[str, str]:
    """Reduce a query result to (explanation, code) regardless of its shape."""
    if isinstance(result, tuple):
        if len(result) >= 2:
            return str(result[0]), str(result[1])
        return str(result[0]) if result else "", ""
    return str(result), ""


def _deliver_to_background_session(
    app: Any, session_id: str, messages: list, result: Any
) -> None:
    """Store a finished generation in a conversation the user is not viewing.

    The visible conversation keeps the render window; this only updates the
    originating session record and flags it so the drawer can show that it has a
    new result waiting.
    """
    from . import sessions as sessions_mod

    sess = sessions_mod.sessions_by_id(app).get(session_id)
    if sess is None:
        return  # conversation was deleted while the query ran
    explanation, code = _unpack_result(result)
    display_code = EXPLAIN_RENDERER + "\n" + code if code else ""

    sess["messages"] = list(messages)
    if display_code:
        history = list(sess.get("code_history") or [])
        labels = list(sess.get("code_history_labels") or [])
        if not history or history[-1] != display_code:
            history.append(display_code)
            labels.append(sess.get("pending_prompt") or "Generated")
        sess["code_history"] = history
        sess["code_history_labels"] = labels
        sess["code_history_pos"] = len(history) - 1
    sess["explanation"] = explanation
    sess.pop("error_message", None)
    sess["unseen"] = True
    sess.pop("pending_prompt", None)
    sessions_mod.finish_background_session(app, sess)


def _deliver_error(app: Any, session_id: str, message: str) -> None:
    """Attach a failure to the conversation that caused it.

    An error belongs to its conversation, so a background failure is stored on
    that session and surfaces when the user switches to it instead of
    interrupting whatever they are looking at now.
    """
    if _is_visible(app, session_id):
        app.state.error_message = message
        return
    from . import sessions as sessions_mod

    sess = sessions_mod.sessions_by_id(app).get(session_id)
    if sess is None:
        return
    sess["error_message"] = message
    sess["unseen"] = True
    sess.pop("pending_prompt", None)
    sessions_mod.finish_background_session(app, sess)


def _generating_sessions(app: Any) -> set:
    """Ids of conversations with a generation in flight (one per conversation)."""
    if not hasattr(app, "_generating_session_ids"):
        app._generating_session_ids = set()
    return app._generating_session_ids


def conversation_token(app: Any, session_id: str) -> int:
    """Return the current generation token for one conversation.

    A generation captures this at start and its result is delivered only if the
    token still matches. Bumped when the conversation is reset or a new
    generation starts in it, so an obsolete result is dropped. Crucially, merely
    switching between conversations does NOT bump anyone's token, so returning to
    a conversation does not orphan its own in-flight generation.
    """
    tokens = getattr(app, "_conversation_tokens", None)
    if tokens is None:
        tokens = app._conversation_tokens = {}
    return tokens.get(session_id, 0)


def bump_conversation_token(app: Any, session_id: str) -> None:
    """Invalidate any in-flight generation belonging to one conversation."""
    tokens = getattr(app, "_conversation_tokens", None)
    if tokens is None:
        tokens = app._conversation_tokens = {}
    tokens[session_id] = tokens.get(session_id, 0) + 1


def _is_visible(app: Any, session_id: str) -> bool:
    """Whether the given conversation is the one currently on screen."""
    return (getattr(app.state, "current_session_id", "") or "") == session_id


def generate_code(app: Any) -> None:
    """Generate VTK code from user query.

    Schedules the work as a background task so the (slow) LLM request runs off
    the event loop and the 3D view stays interactive while it is in flight. A
    synchronous re-entry guard prevents overlapping generations from a second
    click (the button no longer freezes the UI, so double-clicks are possible).
    """
    session_id = getattr(app.state, "current_session_id", "") or ""
    if session_id in _generating_sessions(app):
        return  # this conversation is already generating; others may proceed
    # Mirror the send button's disabled condition so Ctrl+Enter (which bypasses
    # the button) cannot submit an empty prompt or run without a cloud token.
    if not (getattr(app.state, "query_text", "") or "").strip():
        return
    if getattr(app.state, "use_cloud_models", True) and not (
        getattr(app.state, "api_token", "") or ""
    ).strip():
        return
    bump_conversation_token(app, session_id)
    _generating_sessions(app).add(session_id)
    from . import sessions as sessions_mod

    sessions_mod.refresh_sessions_list(app)  # show the spinner on this conversation
    asynchronous.create_task(generate_and_execute_code(app, session_id))


async def generate_and_execute_code(app: Any, origin_session_id: str = "") -> None:
    """Generate VTK code using AI API and execute it.

    Only the blocking network call (prompt_client.query) is offloaded to a
    worker thread via asyncio.to_thread; the code after each await resumes on
    the event loop (main) thread, so all VTK execution and rendering stays
    main-thread-bound as VTK/OpenGL requires.
    """
    if _is_visible(app, origin_session_id):
        app.state.is_loading = True
        app.state.error_message = ""
    app.state.flush()  # show the spinner immediately, before the slow request

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

            # Capture the prompt for inline display, then clear the input box so
            # the sent text does not linger (Claude-style).
            app.state.current_prompt = enhanced_query
            app.state.query_text = ""
            # Record the prompt in the conversation immediately, so switching away
            # mid-generation still snapshots what this conversation is about.
            if enhanced_query:
                app.state.conversation = list(app.state.conversation or []) + [
                    {"role": "user", "content": enhanced_query}
                ]
            app.state.flush()

            # Reinitialize client with current settings
            app._init_prompt_client()
            if hasattr(app.state, "error_message") and app.state.error_message:
                return
            # Tie this generation to its conversation by token. The result is
            # delivered only if this conversation has not been reset or
            # re-generated meanwhile. Switching conversations does not change it.
            token = conversation_token(app, origin_session_id)

            # Refine the CURRENT editor code (including manual edits), not the
            # model's previous output, so generation mutates what is on screen.
            from .conversation import sync_editor_code_into_conversation

            sync_editor_code_into_conversation(app)

            # This generation works on its own copy; it is adopted as the
            # conversation only if this is still the active one when it finishes.
            messages = list(app.state.conversation or [])
            result = await asyncio.to_thread(
                app.prompt_client.query,
                enhanced_query,
                conversation=messages,
                api_key=app._get_api_key(),
                model=app._get_model(),
                base_url=app._get_base_url(),
                max_tokens=int(app.state.max_tokens),
                temperature=float(app.state.temperature),
                top_k=int(app.state.top_k),
                retry_attempts=int(app.state.retry_attempts),
                log_tool_calls=bool(app.state.log_tool_calls),
                agentic_retrieval=bool(app.state.agentic_retrieval),
                provider=app.state.provider,
                custom_prompt=app.custom_prompt_data,
                ui_mode=True,  # This tells the client to use UI-specific components
            )
            if conversation_token(app, origin_session_id) != token:
                # The originating conversation was reset or re-generated; drop this.
                return
            if not _is_visible(app, origin_session_id):
                # The user moved on. Deliver into the originating conversation
                # without touching the visible one or the render window.
                _deliver_to_background_session(app, origin_session_id, messages, result)
                return
            # Keep UI in sync with conversation
            app.state.conversation = messages

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
            push_code_snapshot(
                app, app.state.generated_code, label=app.state.query_text or "Generated"
            )

            # Update navigation after new conversation entry
            from .conversation import build_conversation_navigation, record_turn_checkpoint

            build_conversation_navigation(app)
            record_turn_checkpoint(app)

            from .sessions import touch_current_session

            touch_current_session(app)

        app._conversation_loading = False
        success, exec_error = execute_with_renderer(app, app.state.generated_code)

        # If execution failed and vtk-mcp is configured, retry with the error fed back
        if not success and exec_error and getattr(app.state, "mcp_url", "").strip():
            logger.debug("Execution error, retrying with vtk-mcp: %s", exec_error)
            app.state.error_message = ""
            retry_messages = list(app.state.conversation or [])
            retry_result = await asyncio.to_thread(
                app.prompt_client.query,
                conversation=retry_messages,
                execution_error=exec_error,
                api_key=app._get_api_key(),
                model=app._get_model(),
                base_url=app._get_base_url(),
                max_tokens=int(app.state.max_tokens),
                temperature=float(app.state.temperature),
                top_k=int(app.state.top_k),
                retry_attempts=1,
                log_tool_calls=bool(app.state.log_tool_calls),
                agentic_retrieval=bool(app.state.agentic_retrieval),
                provider=app.state.provider,
                custom_prompt=app.custom_prompt_data,
                ui_mode=True,
            )
            app.state.conversation = retry_messages
            if isinstance(retry_result, tuple) and len(retry_result) >= 2:
                _, retry_code = retry_result[0], retry_result[1]
                if retry_code:
                    app.state.generated_code = EXPLAIN_RENDERER + "\n" + retry_code
                    from .conversation import record_turn_checkpoint

                    push_code_snapshot(
                        app, app.state.generated_code, label=app.state.query_text or "Generated"
                    )
                    record_turn_checkpoint(app)

                    from .sessions import touch_current_session

                    touch_current_session(app)
                    execute_with_renderer(app, app.state.generated_code)
    except ValueError as e:
        if "max_tokens" in str(e):
            msg = f"{str(e)} Current: {app.state.max_tokens}. Try increasing max tokens."
        else:
            msg = f"Error generating code: {str(e)}"
        _deliver_error(app, origin_session_id, msg)
    except Exception as e:
        _deliver_error(app, origin_session_id, f"Error generating code: {str(e)}")
    finally:
        _generating_sessions(app).discard(origin_session_id)
        from . import sessions as sessions_mod

        sessions_mod.refresh_sessions_list(app)  # clear this conversation's spinner
        # Only the visible conversation owns the in-pane spinner.
        if _is_visible(app, origin_session_id):
            app.state.is_loading = False
        app.state.flush()  # push final state (result/error, spinner off) to client


def _editor_line_for(displayed_code: str, line_text: str | None) -> int:
    """Find the 1-based line of ``line_text`` in the code shown in the editor."""
    if not line_text:
        return 0
    target = line_text.strip()
    for i, line in enumerate(displayed_code.splitlines(), start=1):
        if line.strip() == target:
            return i
    return 0


def _format_exec_error(displayed_code: str, error_message: str, line_text: str | None) -> str:
    """Prefix a run error with the editor line and the offending source text."""
    if not line_text:
        return error_message
    editor_line = _editor_line_for(displayed_code, line_text)
    where = f"Line {editor_line}: {line_text}" if editor_line else f"At: {line_text}"
    return f"{where}\n{error_message}"


def execute_with_renderer(app: Any, code_string: str) -> tuple[bool, str | None]:
    """Execute VTK code with our renderer. Returns (success, error_message)."""
    # Resolve bare data-file references (e.g. 'cow.g') to fetched local paths so
    # example-style code runs. No-op unless a VTK data tree is configured. Only
    # the executed copy is rewritten; the stored/displayed code keeps bare names.
    from ..data.resolver import stage_code

    exec_code = stage_code(code_string)
    success, error_message, error_line_text = execute_vtk_code(
        exec_code, app.renderer, app.render_window
    )

    if not success and error_message:
        app.state.error_message = _format_exec_error(
            code_string, error_message, error_line_text
        )

    if success:
        app.state.rendered_code = code_string

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
        push_code_snapshot(app, app.state.generated_code, label="Run")
        execute_with_renderer(app, app.state.generated_code)
    finally:
        app.state.is_loading = False


def push_code_snapshot(app: Any, code_string: str, label: str = "") -> None:
    """Record a labeled code version on the single per-conversation timeline.

    Drops any redo tail, and is a no-op when identical to the current position so
    repeated runs of unchanged code do not bloat the history. The parallel label
    list records what produced each version (a prompt, or "Manual edit").
    """
    history = list(app.state.code_history or [])
    labels = list(app.state.code_history_labels or [])
    pos = app.state.code_history_pos

    # If we branched off after an undo, discard the now-stale redo tail.
    if 0 <= pos < len(history) - 1:
        history = history[: pos + 1]
        labels = labels[: pos + 1]

    if history and history[-1] == code_string:
        return  # nothing changed

    history.append(code_string)
    labels.append(label)
    app.state.code_history = history
    app.state.code_history_labels = labels
    app.state.code_history_pos = len(history) - 1


def undo_code(app: Any) -> None:
    """Step the editor back to the previous code version (does not re-run)."""
    history = app.state.code_history or []
    pos = app.state.code_history_pos
    if pos > 0:
        pos -= 1
        app.state.code_history_pos = pos
        app.state.generated_code = history[pos]


def redo_code(app: Any) -> None:
    """Step the editor forward to the next code version (does not re-run)."""
    history = app.state.code_history or []
    pos = app.state.code_history_pos
    if pos < len(history) - 1:
        pos += 1
        app.state.code_history_pos = pos
        app.state.generated_code = history[pos]


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
