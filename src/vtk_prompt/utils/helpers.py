"""
Helpers Module.

This module provides general helper functions and constants for the VTK Prompt UI application.
"""

import re

# Constants
EXPLAIN_RENDERER = (
    "# renderer is a vtkRenderer injected by this webapp"
    + "\n"
    + "# Use your own vtkRenderer in your application"
)

EXPLANATION_PATTERN = r"<explanation>(.*?)</explanation>"
CODE_PATTERN = r"<code>(.*?)</code>"
EXTRA_INSTRUCTIONS_TAG = "</extra_instructions>"


def ensure_vtk_importable(code: str) -> str:
    """Ensure generated VTK code can run without clobbering module-specific imports.

    Strips a surrounding markdown fence if present, then prepends a top-level
    ``import vtk`` only when the snippet has neither a real ``import vtk`` line nor
    any ``vtkmodules`` import.

    A naive ``"import vtk" in code`` substring test is intentionally avoided: it
    matches inside ``import vtkSphereSource``, which previously caused valid
    ``from vtkmodules.X import vtkY`` code to be truncated to ``import vtkY`` and
    fail at runtime with ``No module named 'vtkY'``.
    """
    code = code.strip()

    # Strip a single surrounding markdown code fence if the model emitted one
    if code.startswith("```"):
        code = re.sub(r"^```[A-Za-z0-9_-]*\n?", "", code)
        code = re.sub(r"\n?```$", "", code).strip()

    has_vtk_import = bool(
        re.search(r"^\s*import\s+vtk(\s+as\s+\w+)?\s*$", code, re.MULTILINE)
        or re.search(r"\bvtkmodules\b", code)
    )
    if not has_vtk_import:
        code = "import vtk\n" + code

    return code
