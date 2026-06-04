"""VTK Code Execution Module."""

import vtk

from .. import get_logger
from ..utils.helpers import ensure_vtk_importable

logger = get_logger(__name__)


def execute_vtk_code(
    code_string: str, renderer: vtk.vtkRenderer, render_window: vtk.vtkRenderWindow
) -> tuple[bool, str | None]:
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

        return True, None

    except Exception as e:
        error_message = f"Error executing code: {str(e)}"
        logger.error(error_message)
        return False, error_message
