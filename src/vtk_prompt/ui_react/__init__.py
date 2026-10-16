"""React client entry point.

Reuses VTKPromptApp's controllers and state wholesale; only the UI differs.
"""

from .shell import build_ui

__all__ = ["build_ui"]
