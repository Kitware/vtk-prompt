"""VTK Code Execution Module."""

import vtk

from .. import get_logger

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

        # Clean the code
        pos = code_string.find("import vtk")
        if pos != -1:
            code_string = code_string[pos:]

        # Ensure vtk is imported
        code_segment = code_string
        if "import vtk" not in code_segment:
            code_segment = "import vtk\n" + code_segment

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
