"""Tests for vtk_prompt.utils.helpers."""

from vtk_prompt.utils.helpers import ensure_vtk_importable


def test_leading_import_vtk_unchanged():
    code = "import vtk\nsrc = vtk.vtkSphereSource()\n"
    assert ensure_vtk_importable(code).splitlines()[0] == "import vtk"


def test_vtkmodules_import_not_truncated():
    # Regression: a bare "import vtk" substring search used to truncate this to
    # "import vtkSphereSource", failing at runtime with No module named.
    code = "from vtkmodules.vtkFiltersSources import vtkSphereSource\nsrc = vtkSphereSource()\n"
    out = ensure_vtk_importable(code)
    assert out.splitlines()[0] == "from vtkmodules.vtkFiltersSources import vtkSphereSource"
    # never decapitated into a standalone module import
    assert "import vtkSphereSource" not in out.splitlines()


def test_parametric_torus_import_not_truncated():
    code = (
        "from vtkmodules.vtkCommonComputationalGeometry import vtkParametricTorus\n"
        "t = vtkParametricTorus()\n"
    )
    out = ensure_vtk_importable(code)
    assert out.startswith("from vtkmodules.vtkCommonComputationalGeometry import vtkParametricTorus")


def test_prepends_import_when_absent():
    code = "s = vtk.vtkConeSource()\n"
    assert ensure_vtk_importable(code).splitlines()[0] == "import vtk"


def test_strips_markdown_fence():
    code = "```python\nfrom vtkmodules.vtkFiltersSources import vtkSphereSource\n```\n"
    out = ensure_vtk_importable(code)
    assert out.splitlines()[0] == "from vtkmodules.vtkFiltersSources import vtkSphereSource"
    assert "```" not in out


def test_import_vtk_as_alias_recognized():
    code = "import vtk as v\nv.vtkSphereSource()\n"
    out = ensure_vtk_importable(code)
    assert out.count("import vtk") == 1  # not prepended a second time
