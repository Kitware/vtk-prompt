"""Tests for console line severity classification."""

import pytest

from vtk_prompt.controllers.generation import _classify_line


@pytest.mark.parametrize(
    "text,kind",
    [
        ("blabla True", "out"),
        ("Building sphere...", "out"),
        ("Warning: deprecated API", "warn"),
        ("DeprecationWarning: use X", "warn"),
        ("vtkDebugLeaks Warning: leaked", "warn"),
        ("ERROR: could not open file", "err"),
        ("Exodus Library Warning/Error: [x]", "err"),
        ("Traceback (most recent call last):", "err"),
        ('  File "<string>", line 3', "err"),
    ],
)
def test_classify_line(text, kind):
    assert _classify_line(text) == kind


def test_stdout_is_never_error_even_with_error_words(monkeypatch):
    """A printed class name like vtkErrorCode must not be flagged as an error."""
    import types
    from vtk_prompt.controllers import generation

    app = types.SimpleNamespace()
    app.state = types.SimpleNamespace(console_log=[], console_lines=[])
    generation._append_console(
        app, stdout="['vtkErrorCode', 'vtkWarningObserver']", stderr="", error=None
    )
    run = app.state.console_log[-1]
    assert run["level"] == "out"
    assert all(line["kind"] == "out" for line in run["lines"])


def test_stderr_is_warning_and_exception_is_error():
    import types
    from vtk_prompt.controllers import generation

    app = types.SimpleNamespace()
    app.state = types.SimpleNamespace(console_log=[], console_lines=[])
    generation._append_console(app, stdout="", stderr="deprecated call", error=None)
    assert app.state.console_log[-1]["level"] == "warn"

    app.state.console_log = []
    app.state.console_lines = []
    generation._append_console(app, stdout="", stderr="", error="Traceback: boom")
    assert app.state.console_log[-1]["level"] == "err"
