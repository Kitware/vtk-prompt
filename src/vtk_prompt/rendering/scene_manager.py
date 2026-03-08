"""
VTK Scene Management Module.
"""

import vtk

from .. import get_logger

logger = get_logger(__name__)


def setup_vtk_renderer() -> tuple[
    vtk.vtkRenderer, vtk.vtkRenderWindow, vtk.vtkRenderWindowInteractor
]:
    """Initialize VTK renderer, render window, and interactor with proper configuration.

    Creates and configures:
    - vtkRenderer with dark background
    - vtkRenderWindow with offscreen rendering enabled
    - vtkRenderWindowInteractor with trackball camera style
    """
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.1)

    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.OffScreenRenderingOn()  # Prevent external window
    render_window.SetSize(800, 600)

    # Create interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

    return renderer, render_window, render_window_interactor


def add_default_scene(renderer: vtk.vtkRenderer) -> None:
    """Add default coordinate axes to prevent empty scene segfaults.

    Creates simple XYZ axes and adds them to the renderer. This prevents
    issues with empty scenes and provides visual reference.
    """
    try:
        # Create simple axes
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1, 1, 1)
        axes.SetShaftType(0)  # Line shaft
        axes.SetCylinderRadius(0.02)

        # Add to renderer
        renderer.AddActor(axes)

        # Reset camera to show axes
        renderer.ResetCamera()
    except Exception as e:
        logger.warning("Could not add default scene: %s", e)


def clear_scene(renderer: vtk.vtkRenderer, render_window: vtk.vtkRenderWindow) -> None:
    """Clear the VTK scene and restore default axes.

    Removes all actors from the renderer, adds default coordinate axes,
    resets the camera, and renders the scene.
    """
    try:
        renderer.RemoveAllViewProps()
        add_default_scene(renderer)
        renderer.ResetCamera()
        render_window.Render()
    except Exception as e:
        logger.error("Error clearing scene: %s", e)


def reset_camera(renderer: vtk.vtkRenderer, render_window: vtk.vtkRenderWindow) -> None:
    """Reset camera view to show all actors.

    Resets the camera to frame all actors in the scene and renders.
    """
    try:
        renderer.ResetCamera()
        render_window.Render()
    except Exception as e:
        logger.error("Error resetting camera: %s", e)
