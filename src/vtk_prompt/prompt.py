"""
VTK Code Generation with LLM Integration (Backward Compatibility Module).

This module maintains backward compatibility for the main CLI function.
The actual implementation has been moved to cli.py.

This module re-exports it to maintain existing import patterns.

Deprecated: Direct imports from this module are deprecated.
Please import from .cli instead for new code.
"""

from .cli import main

__all__ = ["main"]
