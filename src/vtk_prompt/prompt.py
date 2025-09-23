"""
VTK Code Generation with LLM Integration (Backward Compatibility Module).

This module maintains backward compatibility for the VTKPromptClient and main CLI function.
The actual implementations have been moved to:
- client.py: VTKPromptClient class
- cli.py: CLI interface

This module re-exports them to maintain existing import patterns.

Deprecated: Direct imports from this module are deprecated.
Please import from .client or .cli instead for new code.
"""

from .cli import main

# Re-export for backward compatibility
from .client import VTKPromptClient

__all__ = ["VTKPromptClient", "main"]
