"""Tests for in-process jedi-backed Python/VTK completion."""

from vtk_prompt.completion import complete_python, hover_python


def test_completes_vtk_instance_methods():
    code = (
        "from vtkmodules.vtkFiltersSources import vtkSphereSource\n"
        "s = vtkSphereSource()\n"
        "s.SetTh"
    )
    labels = [c["label"] for c in complete_python(code, line=3, column=len("s.SetTh"))]
    assert "SetThetaResolution" in labels


def test_completes_vtk_class_names_via_preamble():
    code = "vtk.vtkConeS"
    labels = [c["label"] for c in complete_python(code, line=1, column=len("vtk.vtkConeS"))]
    assert "vtkConeSource" in labels


def test_payload_shape():
    code = "x = 1\nx."
    out = complete_python(code, line=2, column=2)
    assert all({"label", "kind", "detail"} <= set(item) for item in out)


def test_bad_input_returns_empty_not_raise():
    assert complete_python({"not": "a string"}, 1, 0) == []  # type: ignore[arg-type]


def test_member_completion_includes_inherited_and_is_not_truncated():
    # vtkSphereSource exposes ~200+ members including inherited ones; the result
    # must not be capped below that, or Monaco hides methods the user types.
    code = (
        "from vtkmodules.vtkFiltersSources import vtkSphereSource\n"
        "s = vtkSphereSource()\n"
        "s."
    )
    out = complete_python(code, line=3, column=2)
    labels = {c["label"] for c in out}
    assert len(out) > 100
    assert "SetThetaResolution" in labels  # defined on the class itself
    assert "Update" in labels  # inherited from a superclass (vtkAlgorithm)


def test_hover_returns_signature_and_vtk_prose():
    code = (
        "from vtkmodules.vtkFiltersSources import vtkSphereSource\n"
        "s = vtkSphereSource()\n"
        "s.SetRadius"
    )
    info = hover_python(code, line=3, column=len("s.SetRadius"))
    assert info is not None
    assert any("SetRadius" in s for s in info["signatures"])
    assert "radius" in info["prose"].lower()  # real VTK prose, not just the signature
