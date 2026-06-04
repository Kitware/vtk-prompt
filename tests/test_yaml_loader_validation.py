"""Regression tests for YAMLPromptLoader variable validation.

Retrieved code context (context_snippets) must bypass the scalar-oriented
safety guards: example code legitimately contains dunders like ``__main__`` and
can exceed the 10000-char cap. Small scalar variables stay validated.
"""

import pytest

from vtk_prompt.prompts.yaml_prompt_loader import YAMLPromptLoader


def test_context_snippets_allows_code_with_dunders_and_over_length():
    loader = YAMLPromptLoader()
    code = 'if __name__ == "__main__":\n    main()\n' + "x = 1\n" * 5000  # >10000 chars
    out = loader.substitute_yaml_variables(
        "{{context_snippets}}", {"context_snippets": code}
    )
    assert out == code  # passed through untouched, no ValueError


def test_scalar_variable_still_rejects_dunder_content():
    loader = YAMLPromptLoader()
    with pytest.raises(ValueError):
        loader.substitute_yaml_variables("{{VTK_VERSION}}", {"VTK_VERSION": "__import__"})
