"""
Helpers Module.

This module provides general helper functions and constants for the VTK Prompt UI application.
"""

# Constants
EXPLAIN_RENDERER = (
    "# renderer is a vtkRenderer injected by this webapp"
    + "\n"
    + "# Use your own vtkRenderer in your application"
)

EXPLANATION_PATTERN = r"<explanation>(.*?)</explanation>"
CODE_PATTERN = r"<code>(.*?)</code>"
EXTRA_INSTRUCTIONS_TAG = "</extra_instructions>"
