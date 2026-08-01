"""VTK Code Execution Module."""

import contextlib
import io
import traceback

import vtk

from .. import get_logger
from ..utils.helpers import ensure_vtk_importable

logger = get_logger(__name__)

# Output captured from the most recent run of generated code. The executor keeps
# its (success, error, line) return contract; callers read the console text from
# here so a print() in generated code is visible in the app, not just the server
# terminal.
_last_stdout: str = ""
_last_stderr: str = ""


def last_console_output() -> tuple[str, str]:
    """(stdout, stderr) captured from the most recent execute_vtk_code call.

    Kept as separate streams so callers can classify by origin: stdout is
    ordinary output, stderr is a warning/error. This avoids guessing severity
    from line content (e.g. a printed class name like vtkErrorCode is not an
    error).
    """
    return _last_stdout, _last_stderr


class _NoOpRenderWindow:
    """Stand-in for a render window that generated code constructs itself.

    Handing back the app's real window let scripts call AddRenderer/SetSize/
    SetOffScreenRendering on it, which stacked renderers and corrupted the view
    (black screen after a few switches). This absorbs those calls harmlessly; the
    scene still reaches the app because the code draws into the injected renderer.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __getattr__(self, name: str):
        def _noop(*args: object, **kwargs: object):
            return None

        return _noop


class _NoOpInteractor:
    """Stand-in for vtkRenderWindowInteractor used while running generated code.

    Standalone VTK scripts end with interactor.Start(), which opens a native
    window and blocks the event loop until the user presses q. Inside the app the
    scene belongs to the trame view, so the interactor is replaced by an inert
    object that accepts the usual calls and does nothing.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __getattr__(self, name: str):
        def _noop(*args: object, **kwargs: object) -> None:
            return None

        return _noop


def execute_vtk_code(
    code_string: str, renderer: vtk.vtkRenderer, render_window: vtk.vtkRenderWindow
) -> tuple[bool, str | None, str | None]:
    """Execute VTK code with renderer context.

    Clears previous actors, cleans the code string, executes it with the renderer
    available in the global scope, and resets the camera.
    """
    try:
        # Clear previous actors
        renderer.RemoveAllViewProps()

        # Ensure vtk is importable without clobbering module-specific imports
        code_segment = ensure_vtk_importable(code_string)

        # Create execution globals with renderer available.
        # Notes:
        # - __name__ is set to "__main__" so generated scripts guarded by
        #   `if __name__ == "__main__":` actually run. Without it, a bare
        #   __name__ resolves via builtins to "builtins", the guard is False,
        #   and the script body (e.g. a main()) never executes -> blank view.
        # - render_window is injected alongside renderer for code that uses it.
        # - A single namespace (globals only) is used so top-level defs and the
        #   guard share one scope and functions can see the injected names.
        exec_globals = {
            "vtk": vtk,
            "renderer": renderer,
            "render_window": render_window,
            "__name__": "__main__",
        }

        # Keep generated code inside the app (a script that builds its own window
        # or interactor would otherwise pop up a native window and block on
        # Start()), and capture stdout/stderr separately so the console can
        # colour output by stream. Restore vtk afterwards.
        global _last_stdout, _last_stderr
        real_window_cls = vtk.vtkRenderWindow
        real_interactor_cls = vtk.vtkRenderWindowInteractor
        vtk.vtkRenderWindow = _NoOpRenderWindow  # type: ignore[assignment,misc]
        vtk.vtkRenderWindowInteractor = _NoOpInteractor  # type: ignore[assignment,misc]
        out_buf, err_buf = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(
                err_buf
            ):
                exec(code_segment, exec_globals)

                # Reset camera and render
                try:
                    renderer.ResetCamera()
                    render_window.Render()
                except Exception as render_error:
                    logger.warning("Render error: %s", render_error)
        finally:
            vtk.vtkRenderWindow = real_window_cls  # type: ignore[assignment,misc]
            vtk.vtkRenderWindowInteractor = real_interactor_cls  # noqa: E501  # type: ignore[assignment,misc]
        _last_stdout, _last_stderr = out_buf.getvalue(), err_buf.getvalue()

        return True, None, None

    except (Exception, SystemExit) as e:
        _last_stdout = locals().get("out_buf", io.StringIO()).getvalue()
        _last_stderr = locals().get("err_buf", io.StringIO()).getvalue()
        # SystemExit is NOT an Exception subclass: generated code that calls
        # sys.exit() or argparse.parse_args() (common in VTK example scripts that
        # read command-line data files) would otherwise propagate out and kill the
        # whole trame app. Trap it here and report it as a normal code error.
        if isinstance(e, SystemExit):
            error_message = (
                "Error executing code: the generated code runs as a standalone "
                "script (it uses argparse/sys.exit and expects command-line "
                "arguments), so it cannot run inside the app as written."
            )
        else:
            error_message = f"Error executing code: {str(e)}"
        logger.error(error_message)
        # Identify the offending line *within the executed code* and return its
        # text (not its number). The executed code differs from what the editor
        # shows: an explanation banner is prepended, a markdown fence may be
        # stripped, an import may be added, and data literals may be rewritten.
        # The caller matches on text to locate the editor line robustly.
        line_text = None
        segment = locals().get("code_segment")
        if isinstance(segment, str):
            lineno = None
            if isinstance(e, SyntaxError) and e.lineno:
                lineno = e.lineno
            else:
                for frame, frame_line in traceback.walk_tb(e.__traceback__):
                    if frame.f_code.co_filename == "<string>":
                        lineno = frame_line
            if lineno is not None:
                lines = segment.splitlines()
                if 1 <= lineno <= len(lines):
                    line_text = lines[lineno - 1].strip()
        return False, error_message, line_text
