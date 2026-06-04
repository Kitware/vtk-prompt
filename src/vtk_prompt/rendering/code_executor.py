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

        # Create execution globals with renderer available
        exec_globals = {
            "vtk": vtk,
            "renderer": renderer,
        }

        # Execute the code
        exec(code_segment, exec_globals, {})

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
