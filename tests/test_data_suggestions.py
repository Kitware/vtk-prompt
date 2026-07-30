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
