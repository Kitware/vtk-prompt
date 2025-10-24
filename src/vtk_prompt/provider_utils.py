"""
Provider utilities for managing curated model lists and provider configurations.

This module provides curated lists of models that work well for VTK code generation,
rather than dynamically fetching all available models from providers.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Curated models for each provider - selected for VTK code generation quality
OPENAI_MODELS = ["gpt-5", "gpt-4.1", "o4-mini", "o3"]

ANTHROPIC_MODELS = [
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
]

GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

NIM_MODELS = [
    "meta/llama3-70b-instruct",
    "meta/llama3-8b-instruct",
    "microsoft/phi-3-medium-4k-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
]


# Models that don't support temperature control (must use temperature=1.0)
TEMPERATURE_UNSUPPORTED_MODELS = ["gpt-5", "o4-mini", "o3"]


def supports_temperature(model: str) -> bool:
    """Check if a model supports temperature control."""
    return model not in TEMPERATURE_UNSUPPORTED_MODELS


def get_model_temperature(model: str, requested_temperature: float = 0.7) -> float:
    """Get the appropriate temperature for a model."""
    if supports_temperature(model):
        return requested_temperature
    else:
        return 1.0


def get_available_models() -> Dict[str, List[str]]:
    """Get curated models for all providers."""
    return {
        "openai": OPENAI_MODELS,
        "anthropic": ANTHROPIC_MODELS,
        "gemini": GEMINI_MODELS,
        "nim": NIM_MODELS,
    }


def get_provider_models(provider: str) -> List[str]:
    """Get curated models for a specific provider."""
    models = get_available_models()
    return models.get(provider, [])


def get_supported_providers() -> List[str]:
    """Get list of supported providers."""
    return ["openai", "anthropic", "gemini", "nim"]


def get_default_model(provider: str) -> str:
    """Get the default/recommended model for a provider."""
    defaults = {
        "openai": "gpt-5",
        "anthropic": "claude-opus-4-1-20250805",
        "gemini": "gemini-2.5-pro",
        "nim": "meta/llama3-70b-instruct",
    }
    return defaults.get(provider, "gpt-5")
