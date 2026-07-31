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


def generate_code(app: Any) -> None:
    """Generate VTK code from user query.

    Schedules the work as a background task so the (slow) LLM request runs off
    the event loop and the 3D view stays interactive while it is in flight. A
    synchronous re-entry guard prevents overlapping generations from a second
    click (the button no longer freezes the UI, so double-clicks are possible).
    """
    if getattr(app, "_generating", False):
        return
    # Mirror the send button's disabled condition so Ctrl+Enter (which bypasses
    # the button) cannot submit an empty prompt or run without a cloud token.
    if not (getattr(app.state, "query_text", "") or "").strip():
        return
    if getattr(app.state, "use_cloud_models", True) and not (
        getattr(app.state, "api_token", "") or ""
    ).strip():
        return
    app._generating = True
    asynchronous.create_task(generate_and_execute_code(app))


async def generate_and_execute_code(app: Any) -> None:
    """Generate VTK code using AI API and execute it.

    Only the blocking network call (prompt_client.query) is offloaded to a
    worker thread via asyncio.to_thread; the code after each await resumes on
    the event loop (main) thread, so all VTK execution and rendering stays
    main-thread-bound as VTK/OpenGL requires.
    """
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
            app.state.flush()

            # Reinitialize client with current settings
            app._init_prompt_client()
            if hasattr(app.state, "error_message") and app.state.error_message:
                return
            # Tie this generation to the active conversation. A reset or session
            # switch during the offloaded query bumps the epoch, so a stale result
            # is discarded rather than written back over the new conversation.
            epoch = getattr(app, "_conversation_epoch", 0)

            # Refine the CURRENT editor code (including manual edits), not the
            # model's previous output, so generation mutates what is on screen.
            from .conversation import sync_editor_code_into_conversation

            sync_editor_code_into_conversation(app)

            result = await asyncio.to_thread(
                app.prompt_client.query,
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
            if getattr(app, "_conversation_epoch", 0) != epoch:
                return  # conversation was reset/switched while the query ran
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
            retry_result = await asyncio.to_thread(
                app.prompt_client.query,
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
            app.state.error_message = (
                f"{str(e)} Current: {app.state.max_tokens}. Try increasing max tokens."
            )
        else:
            app.state.error_message = f"Error generating code: {str(e)}"
    except Exception as e:
        app.state.error_message = f"Error generating code: {str(e)}"
    finally:
        app.state.is_loading = False
        app._generating = False
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


def _classify_line(text: str) -> str:
    """Tag a captured line by severity so the console can colour it.

    VTK (C++) and Python both use recognisable markers: "ERROR"/"Traceback" for
    failures, "Warning" for warnings. Everything else is ordinary output.
    """
    low = text.lower()
    if ("error" in low) or ("traceback" in low) or low.startswith("  file "):
        return "err"
    if ("warning" in low) or ("warn:" in low) or ("deprecat" in low):
        return "warn"
    return "out"


def _append_console(
    app: Any, stdout: str, stderr: str = "", error: str | None = None
) -> None:
    """Record a run's captured output as a collapsible group in the console.

    Each run is one entry {stamp, lines:[{kind,text}], summary}. A run that
    produced no output is skipped, so the console never shows an empty marker.
    """
    import time

    def _cap(s: str) -> str:
        # A single very long line (e.g. print(dir(vtk))) is unwieldy; keep the
        # console readable by truncating with an indicator.
        return s if len(s) <= 2000 else s[:2000] + " ... [truncated]"

    entries: list[dict] = []
    # stdout is ordinary output; stderr is a warning; a raised exception is an
    # error. Classify by origin rather than by scanning text for words like
    # "error" (a printed class name such as vtkErrorCode is not an error).
    for raw in (stdout or "").splitlines():
        entries.append({"kind": "out", "text": _cap(raw)})
    for raw in (stderr or "").splitlines():
        # Everything on stderr is at least a warning; VTK also writes hard
        # errors there, so upgrade to error when the text says so.
        kind = "err" if _classify_line(raw) == "err" else "warn"
        entries.append({"kind": kind, "text": _cap(raw)})
    if error:
        for raw in error.splitlines():
            entries.append({"kind": "err", "text": _cap(raw)})
    if not entries:
        return  # nothing to show for this run

    n_err = sum(1 for e in entries if e["kind"] == "err")
    n_warn = sum(1 for e in entries if e["kind"] == "warn")
    parts = [f"{len(entries)} line" + ("s" if len(entries) != 1 else "")]
    if n_err:
        parts.append(f"{n_err} error" + ("s" if n_err != 1 else ""))
    if n_warn:
        parts.append(f"{n_warn} warning" + ("s" if n_warn != 1 else ""))
    stamp = time.strftime("%H:%M:%S")
    level = "err" if n_err else ("warn" if n_warn else "out")
    runs = list(getattr(app.state, "console_log", []) or [])
    runs.append(
        {"stamp": stamp, "lines": entries, "summary": ", ".join(parts), "level": level}
    )
    app.state.console_log = runs[-100:]

    # Flat, render-friendly view: a header line per run then its output lines.
    # A single non-nested list keeps the UI markup simple and robust.
    _class = {
        "err": "text-error",
        "warn": "text-warning",
        "run": "text-medium-emphasis font-weight-medium",
        "out": "text-high-emphasis",
    }
    flat = list(getattr(app.state, "console_lines", []) or [])
    header = {"kind": "run", "text": f"\u25b6 {stamp}  \u2014  {', '.join(parts)}"}
    for item in [header, *entries]:
        item["cls"] = _class.get(item["kind"], "text-high-emphasis")
        flat.append(item)
    app.state.console_lines = flat[-1000:]


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

    # Surface anything the code printed (and any error) in the console panel.
    from ..rendering.code_executor import last_console_output

    _stdout, _stderr = last_console_output()
    _append_console(
        app, _stdout, _stderr, error_message if not success else None
    )

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
