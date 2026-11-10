"""Tests for the volume rendering prompt component and its trigger."""

from vtk_prompt.prompts.prompt_component_assembler import (
    _wants_volume_rendering,
    assemble_vtk_prompt,
)


def test_triggers_on_volume_wording():
    assert _wants_volume_rendering("volume render the head", None)
    assert _wants_volume_rendering("set up a transfer function", None)


def test_triggers_on_a_volumetric_file_the_request_names():
    assert _wants_volume_rendering("show me LIDCFull.vti", ["LIDCFull.vti"])


def test_ignores_an_uploaded_volume_the_request_does_not_name():
    assert not _wants_volume_rendering("create a red sphere", ["LIDCFull.vti"])


def test_surface_requests_are_left_alone():
    assert not _wants_volume_rendering("show me suzanne.vtp", ["suzanne.vtp"])
    assert not _wants_volume_rendering("create a cone", None)


def test_guidance_reaches_the_prompt():
    data = assemble_vtk_prompt(
        "volume render the chest CT in LIDCFull.vti",
        uploaded_files=["LIDCFull.vti"],
    )
    text = " ".join(m["content"] for m in data["messages"])
    assert "SetGradientOpacity" in text
    assert "Hounsfield" in text


def test_guidance_stays_out_of_unrelated_prompts():
    data = assemble_vtk_prompt("create a red sphere")
    text = " ".join(m["content"] for m in data["messages"])
    assert "SetGradientOpacity" not in text
