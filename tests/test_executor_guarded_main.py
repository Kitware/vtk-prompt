"""Regression test: webapp executor must run guarded-main generated scripts.

LLMs frequently emit standalone scripts wrapped in:

    def main(): ...
    if __name__ == "__main__":
        main()

The webapp executes these with an injected ``renderer``. If __name__ is not set
to "__main__" in the exec namespace, a bare __name__ resolves via builtins to
"builtins", the guard is False, main() never runs, and nothing reaches the
renderer (blank view, no error). This test guards against that regression.
"""

import pytest

vtk = pytest.importorskip("vtk")

from vtk_prompt.rendering.code_executor import execute_vtk_code  # noqa: E402

GUARDED_SPHERE = """
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper


def main():
    sphere = vtkSphereSource()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)
    renderer.AddActor(actor)


if __name__ == "__main__":
    main()
"""


def test_guarded_main_script_adds_actor_to_renderer():
    try:
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.AddRenderer(renderer)
    except Exception:  # pragma: no cover - no VTK rendering backend available
        pytest.skip("VTK render window unavailable in this environment")

    ok, err = execute_vtk_code(GUARDED_SPHERE, renderer, render_window)
    assert ok, err
    assert renderer.GetActors().GetNumberOfItems() == 1
