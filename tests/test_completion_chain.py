"""Tests for docstring-based resolution through VTK call chains.

jedi cannot infer through VTK's C-extension methods: it renders the return type
in a signature because it parses the docstring, but it does not use that for
inference. So ``tetra.GetPointIds().SetId`` yielded no hover and no completions
even though ``tetra.GetPointIds`` resolved fine. These cover the fallback that
walks the chain by reading the ``->`` annotations.
"""

from vtk_prompt.completion import complete_python, hover_python

TETRA = "import vtk\ntetra = vtk.vtkTetra()\ntetra.GetPointIds().SetId(0, 0)"
ACTOR = "import vtk\na = vtk.vtkActor()\na.GetProperty().SetColor(1, 0, 0)"
DEEP = (
    "import vtk\n"
    "p = vtk.vtkPolyData()\n"
    "p.GetPointData().GetArray(0).GetNumberOfTuples()"
)


def _hover_on(code, token):
    """Hover with the cursor inside ``token`` on whichever line holds it."""
    for lineno, text in enumerate(code.splitlines(), 1):
        if token in text:
            return hover_python(code, lineno, text.index(token) + 2)
    raise AssertionError(f"{token!r} not in code")


def test_hover_through_one_call():
    info = _hover_on(TETRA, "SetId")
    assert info is not None
    assert info["name"] == "SetId"
    assert any("SetId(" in s for s in info["signatures"])
    assert info["prose"]


def test_hover_through_call_on_another_class():
    info = _hover_on(ACTOR, "SetColor")
    assert info is not None
    assert any("SetColor(" in s for s in info["signatures"])


def test_hover_through_multiple_chained_calls():
    info = _hover_on(DEEP, "GetNumberOfTuples")
    assert info is not None
    assert info["name"] == "GetNumberOfTuples"


def test_hover_payload_shape_matches_jedi_path():
    chained = _hover_on(TETRA, "SetId")
    direct = _hover_on("import vtk\nids = vtk.vtkIdList()\nids.SetId(0, 0)", "SetId")
    assert direct is not None
    assert set(chained) == set(direct)


def test_completion_after_call_returns_members():
    code = "import vtk\ntetra = vtk.vtkTetra()\ntetra.GetPointIds()."
    labels = [c["label"] for c in complete_python(code, 3, len("tetra.GetPointIds()."))]
    assert "SetId" in labels
    assert "GetNumberOfIds" in labels


def test_completion_payload_shape_after_call():
    code = "import vtk\na = vtk.vtkActor()\na.GetProperty()."
    out = complete_python(code, 3, len("a.GetProperty()."))
    assert out
    assert all({"label", "kind", "detail"} <= set(item) for item in out)


def test_direct_object_still_uses_jedi():
    """The fallback must not shadow the path that already worked."""
    code = "import vtk\nc = vtk.vtkConeSource()\nc.SetRad"
    labels = [c["label"] for c in complete_python(code, 3, len("c.SetRad"))]
    assert "SetRadius" in labels


def test_chain_off_a_primitive_resolves_to_nothing():
    """``-> int`` gives nothing to chain from, and must not guess."""
    code = "import vtk\nvtk.vtkTetra().GetNumberOfPoints().Foo"
    assert hover_python(code, 2, code.splitlines()[1].index("Foo") + 1) is None


def test_unknown_root_is_not_resolved():
    code = "mystery.GetPointIds().SetId(0, 0)"
    assert hover_python(code, 1, code.index("SetId") + 2) is None


def test_out_of_range_position_returns_none():
    assert hover_python("import vtk\n", 99, 99) is None
    assert complete_python("import vtk\n", 99, 99) == []


def test_injected_object_resolves_without_jedi():
    """The reported bug: renderer.AddActor lost its hover in longer scripts.

    jedi's help() costs hundreds of ms warm and seconds cold, so the editor
    cancelled the request. Resolution for a VTK receiver must not touch jedi.
    """
    import vtk

    from vtk_prompt import completion as c

    c.register_runtime_objects(renderer=vtk.vtkRenderer())
    code = "renderer.AddActor(actor)"

    called = []
    real = c.jedi.Interpreter

    class Spy:
        def __init__(self, *a, **k):
            called.append(1)
            self._inner = real(*a, **k)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    c.jedi.Interpreter = Spy
    try:
        info = hover_python(code, 1, code.index("AddActor") + 2)
    finally:
        c.jedi.Interpreter = real

    assert info is not None
    assert info["name"] == "AddActor"
    assert not called, "fast path must not invoke jedi for a VTK receiver"


def test_local_vtk_assignment_resolves_without_jedi():
    """`cone = vtk.vtkConeSource()` is the shape generated scripts use."""
    code = "import vtk\ncone = vtk.vtkConeSource()\ncone.SetResolution(20)"
    info = _hover_on(code, "SetResolution")
    assert info is not None
    assert any("SetResolution(" in s for s in info["signatures"])


def test_hover_stays_fast_on_a_realistic_script():
    """Guards the actual failure mode: correct but too slow to be delivered."""
    import time

    code = (
        "import vtk\n"
        "cone = vtk.vtkConeSource()\n"
        "mapper = vtk.vtkPolyDataMapper()\n"
        "mapper.SetInputConnection(cone.GetOutputPort())\n"
        "actor = vtk.vtkActor()\n"
        "actor.GetProperty().SetColor(0.2, 0.4, 0.9)\n"
    )
    line = 6
    col = code.splitlines()[5].index("SetColor") + 2

    start = time.perf_counter()
    info = hover_python(code, line, col)
    elapsed = time.perf_counter() - start

    assert info is not None
    # The docstring path runs in ~1ms; jedi took seconds cold. A generous
    # ceiling still catches a regression back onto the slow path.
    assert elapsed < 0.5, f"hover took {elapsed:.2f}s - likely back on jedi"


def test_non_vtk_code_still_resolves():
    """The fast path must not shadow jedi for ordinary Python."""
    code = "import os\nos.path.join('a', 'b')"
    assert hover_python(code, 2, code.splitlines()[1].index("join") + 2) is not None
    assert len(complete_python("import json\njson.", 2, 5)) > 0


def test_module_root_is_not_treated_as_an_instance():
    """`vtk` is in the namespace as a module; type(module) must not be used.

    Were the fast path to claim it, completion would list the *module type's*
    attributes instead of VTK's classes.
    """
    from vtk_prompt.completion import _live_class, _resolve_receiver

    assert _live_class("vtk") is None
    assert _resolve_receiver("import vtk\nvtk.", 2, 4, allow_jedi=False) is None
    # jedi still answers it, and prefix-filtered lookups find real classes.
    labels = [c["label"] for c in complete_python("import vtk\nvtk.vtkConeS", 2, 12)]
    assert "vtkConeSource" in labels
