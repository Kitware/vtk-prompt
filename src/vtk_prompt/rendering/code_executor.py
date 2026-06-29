"""VTK Code Execution Module."""

import traceback

import vtk

from .. import get_logger
from ..utils.helpers import ensure_vtk_importable

logger = get_logger(__name__)


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

        # Execute the code
        exec(code_segment, exec_globals)

        # Reset camera and render
        try:
            renderer.ResetCamera()
            render_window.Render()
        except Exception as render_error:
            logger.warning("Render error: %s", render_error)

        return True, None, None

    except Exception as e:
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
