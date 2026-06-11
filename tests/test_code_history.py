"""Tests for the editable code panel's version history (undo/redo)."""

import types

from vtk_prompt.controllers import generation


def _app():
    app = types.SimpleNamespace()
    app.state = types.SimpleNamespace(
        code_history=[],
        code_history_pos=-1,
        generated_code="",
        error_message="",
        is_loading=False,
    )
    return app


def test_push_builds_history_and_dedups_head():
    app = _app()
    generation.push_code_snapshot(app, "v1")
    generation.push_code_snapshot(app, "v2")
    assert app.state.code_history == ["v1", "v2"]
    assert app.state.code_history_pos == 1
    # pushing the same code at the head is a no-op
    generation.push_code_snapshot(app, "v2")
    assert app.state.code_history == ["v1", "v2"]
    assert app.state.code_history_pos == 1


def test_undo_then_redo_resets_position_and_rerenders(monkeypatch):
    rendered = []
    monkeypatch.setattr(
        generation,
        "execute_with_renderer",
        lambda app, code: (rendered.append(code), (True, None))[1],
    )
    app = _app()
    for v in ("v1", "v2", "v3"):
        generation.push_code_snapshot(app, v)
    assert app.state.code_history_pos == 2

    generation.undo_code(app)
    assert (app.state.generated_code, app.state.code_history_pos) == ("v2", 1)
    generation.undo_code(app)
    assert (app.state.generated_code, app.state.code_history_pos) == ("v1", 0)
    generation.undo_code(app)  # already at the bottom -> no-op
    assert app.state.code_history_pos == 0

    generation.redo_code(app)
    assert (app.state.generated_code, app.state.code_history_pos) == ("v2", 1)

    # each successful undo/redo re-renders the restored code
    assert rendered == ["v2", "v1", "v2"]


def test_editing_after_undo_drops_the_redo_tail():
    app = _app()
    for v in ("v1", "v2", "v3"):
        generation.push_code_snapshot(app, v)
    # simulate having undone back to v1
    app.state.code_history_pos = 0
    app.state.generated_code = "v1"
    # an edit + run from here should branch, discarding v2/v3
    generation.push_code_snapshot(app, "v1-edited")
    assert app.state.code_history == ["v1", "v1-edited"]
    assert app.state.code_history_pos == 1
