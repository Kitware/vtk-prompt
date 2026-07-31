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
