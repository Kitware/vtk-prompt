"""Tests for vtkmodules import suggestions."""

from vtk_prompt.data import imports


def test_moves_class_to_the_module_that_defines_it():
    code = "from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkNamedColors\n"
    error = "cannot import name 'vtkNamedColors' from 'vtkmodules.vtkFiltersSources'"

    (pick,) = imports.suggestions(code, error)

    assert pick["name"] == "vtkNamedColors"
    assert pick["line"] == 1
    fixed = imports.apply_suggestion(code, pick)
    assert "from vtkmodules.vtkCommonColor import vtkNamedColors" in fixed
    # the unrelated name stays importable from where it was
    assert "from vtkmodules.vtkFiltersSources import vtkSphereSource" in fixed


def test_unknown_module_is_resolved_per_name():
    code = "from vtkmodules.vtkParametricFunctions import vtkParametricTorus\n"
    error = "No module named 'vtkmodules.vtkParametricFunctions'"

    (pick,) = imports.suggestions(code, error)

    assert "vtkCommonComputationalGeometry" in pick["new"]


def test_misspelled_class_offers_near_misses_and_drops_the_bad_name():
    code = "from vtkmodules.vtkFiltersSources import vtkConeSource, vtkSphereSorce\n"
    error = "cannot import name 'vtkSphereSorce' from 'vtkmodules.vtkFiltersSources'"

    picks = imports.suggestions(code, error)

    assert picks, "expected at least one near-miss suggestion"
    assert "vtkSphereSource" in picks[0]["reason"]
    fixed = imports.apply_suggestion(code, picks[0])
    assert "vtkSphereSorce" not in fixed
    assert "vtkConeSource" in fixed


def test_non_import_errors_are_ignored():
    assert imports.suggestions("x = 1\n", "NameError: name 'y' is not defined") == []


def test_no_suggestion_when_the_import_line_is_absent():
    error = "cannot import name 'vtkNamedColors' from 'vtkmodules.vtkFiltersSources'"
    assert imports.suggestions("x = 1\n", error) == []


def test_apply_is_a_no_op_when_the_code_moved_on():
    code = "from vtkmodules.vtkFiltersSources import vtkSphereSource\n"
    stale = {"line": 1, "old": "from somewhere import other", "new": "x"}
    assert imports.apply_suggestion(code, stale) == code
