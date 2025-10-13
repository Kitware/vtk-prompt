"""
Test suite for VTK prompt assembly system.

Tests focusing on key prompt functionality.
"""

import pytest
import re
from vtk_prompt.prompts import assemble_vtk_prompt, PYTHON_VERSION


def _assert_basic_structure(result):
    """Helper: Assert basic prompt structure is correct."""
    assert "messages" in result
    assert isinstance(result["messages"], list)
    assert len(result["messages"]) >= 3
    assert all("role" in msg and "content" in msg for msg in result["messages"])
    assert result.get("model") == "openai/gpt-5"


def _get_content(result):
    """Helper: Get combined content from all messages."""
    return " ".join([msg["content"] for msg in result["messages"]])


class TestPromptAssembly:
    """Test prompt assembly for different scenarios."""

    @pytest.mark.parametrize(
        "ui_mode,expected_ui_content",
        [
            (False, False),  # CLI mode - no UI content
            (True, True),  # UI mode - has UI content
        ],
    )
    def test_default_values(self, ui_mode, expected_ui_content):
        """Test default values work for both CLI and UI modes."""
        result = assemble_vtk_prompt("create a sphere", ui_mode=ui_mode)

        _assert_basic_structure(result)
        # Check default model parameters
        assert result.get("modelParameters", {}).get("temperature") == 0.5
        assert result.get("modelParameters", {}).get("max_tokens") == 10000

        content = _get_content(result)
        assert "create a sphere" in content

        # Check that VTK version is present but don't assume a specific version
        assert re.search(r"9\.\d+\.\d+", content), "VTK version should be present in content"
        assert PYTHON_VERSION in content
        assert "DO NOT READ OUTSIDE DATA" in content

        # UI-specific content check
        ui_content_present = "injected vtkrenderer object named renderer" in content
        assert ui_content_present == expected_ui_content

    @pytest.mark.parametrize(
        "ui_mode,rag_enabled",
        [
            (False, True),  # CLI with RAG
            (True, False),  # UI without RAG
            (True, True),  # UI with RAG
        ],
    )
    def test_feature_combinations(self, ui_mode, rag_enabled):
        """Test different combinations of UI and RAG features."""
        kwargs = {
            "ui_mode": ui_mode,
            "VTK_VERSION": "9.3.0",  # Override version
        }

        if rag_enabled:
            kwargs.update({"rag_enabled": True, "context_snippets": "example RAG content"})

        result = assemble_vtk_prompt("create a cube", **kwargs)

        _assert_basic_structure(result)
        content = _get_content(result)

        # Check overridden version
        assert "9.3.0" in content
        assert "create a cube" in content

        # Check feature-specific content
        if ui_mode:
            assert "injected vtkrenderer object named renderer" in content
        if rag_enabled:
            assert "example RAG content" in content

    def test_parameter_overrides(self):
        """Test that parameter overrides work correctly."""
        result = assemble_vtk_prompt(
            "create a torus",
            ui_mode=True,
            rag_enabled=True,
            context_snippets="torus example code",
            VTK_VERSION="9.1.0",
            PYTHON_VERSION=">=3.12",
        )

        _assert_basic_structure(result)
        content = _get_content(result)

        # All overridden values should be present
        assert "create a torus" in content
        assert "torus example code" in content  # RAG
        assert "injected vtkrenderer object named renderer" in content  # UI
        assert "9.1.0" in content  # Overridden VTK
        assert ">=3.12" in content  # Overridden Python

    def test_error_conditions(self):
        """Test error handling and edge cases."""
        # RAG without context should raise error
        with pytest.raises(ValueError, match="context_snippets required when rag_enabled=True"):
            assemble_vtk_prompt("test", rag_enabled=True)

        # Empty request should work
        result = assemble_vtk_prompt("")
        _assert_basic_structure(result)
        assert "Request: " in _get_content(result)
