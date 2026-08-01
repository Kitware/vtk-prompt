"""Tests for near-miss data-file suggestions in the resolver."""

from vtk_prompt.data import resolver


def _index(monkeypatch, names):
    monkeypatch.setattr(resolver, "_load_index", lambda: {n: "hash" for n in names})
    monkeypatch.setattr(resolver.uploads, "uploaded_names", lambda: [])


def test_stem_match_is_suggested(monkeypatch):
    _index(monkeypatch, ["can.ex2", "can.exdg", "other.vtk"])
    out = resolver.suggestions("r.SetFileName('can.ex')")
    assert out and out[0]["name"] == "can.ex"
    assert "can.ex2" in out[0]["matches"]


def test_valid_reference_has_no_suggestion(monkeypatch):
    _index(monkeypatch, ["can.ex2"])
    assert resolver.suggestions("r.SetFileName('can.ex2')") == []


def test_paths_and_code_are_ignored(monkeypatch):
    _index(monkeypatch, ["can.ex2"])
    assert resolver.suggestions("open('/abs/can.ex')") == []
    assert resolver.suggestions("print('hello world')") == []
    assert resolver.suggestions("f'{name}.vtk'") == []


def test_no_index_no_suggestions(monkeypatch):
    _index(monkeypatch, [])
    assert resolver.suggestions("r.SetFileName('can.ex')") == []


def test_apply_swaps_reference_in_current_code(monkeypatch):
    import types
    from vtk_prompt.controllers import generation

    calls = []
    app = types.SimpleNamespace()
    app.state = types.SimpleNamespace(
        generated_code="r.SetFileName('can.ex')",
        code_history=["r.SetFileName('can.ex')"],
        code_history_labels=["gen"],
        code_history_pos=0,
        data_suggestions=[{"missing": "can.ex", "suggestion": "can.ex2"}],
        error_message="nope",
    )
    monkeypatch.setattr(generation, "push_code_snapshot", lambda a, c, label="": calls.append(("snap", c)))
    monkeypatch.setattr(generation, "execute_with_renderer", lambda a, c: calls.append(("run", c)))
    generation.apply_data_suggestion(app, "can.ex", "can.ex2")
    assert "'can.ex2'" in app.state.generated_code
    assert app.state.data_suggestions == []
    assert app.state.error_message == ""
    assert ("run", "r.SetFileName('can.ex2')") in calls
