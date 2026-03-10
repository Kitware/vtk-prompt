"""
Prompt Loader Module.

This module provides functions for loading and processing custom YAML prompt files
in the VTK Prompt UI application.
"""

from pathlib import Path
from typing import Any

import yaml

from .. import get_logger
from ..provider_utils import DEFAULT_MODEL, get_supported_providers, supports_temperature

logger = get_logger(__name__)


def load_custom_prompt_file(app: Any) -> None:
    """Load custom YAML prompt file and extract model parameters."""
    if not app.custom_prompt_file:
        return

    try:
        custom_file_path = Path(app.custom_prompt_file)
        if not custom_file_path.exists():
            logger.error("Custom prompt file not found: %s", app.custom_prompt_file)
            return

        with open(custom_file_path, "r") as f:
            app.custom_prompt_data = yaml.safe_load(f)

        logger.info("Loaded custom prompt file: %s", custom_file_path.name)

        # Override UI defaults with custom prompt parameters
        if app.custom_prompt_data and isinstance(app.custom_prompt_data, dict):
            _process_model_configuration(app)
            _process_rag_and_generation_settings(app)
            _process_model_parameters(app)
    except (yaml.YAMLError, ValueError) as e:
        # Log error and surface to UI as well
        logger.error("Failed to load custom prompt file %s: %s", app.custom_prompt_file, e)
        app.state.error_message = str(e)
        app.custom_prompt_data = None


def _process_model_configuration(app: Any) -> None:
    """Process model configuration from custom prompt data."""
    model_value = app.custom_prompt_data.get("model", DEFAULT_MODEL)
    if isinstance(model_value, str) and model_value:
        if "/" in model_value:
            provider_part, model_part = model_value.split("/", 1)
            # Validate provider
            supported = set(get_supported_providers() + ["local"])
            if provider_part not in supported or not model_part.strip():
                msg = (
                    "Invalid 'model' in prompt file. Expected '<provider>/<model>' "
                    "with provider in {openai, anthropic, gemini, nim, local}."
                )
                app.state.error_message = msg
                raise ValueError(msg)
            if provider_part == "local":
                # Switch to local mode
                app.state.use_cloud_models = False
                app.state.tab_index = 1
                app.state.local_model = model_part
            else:
                # Cloud provider/model
                app.state.use_cloud_models = True
                app.state.tab_index = 0
                app.state.provider = provider_part
                app.state.model = model_part
        else:
            # Enforce explicit provider/model format
            msg = (
                "Invalid 'model' format in prompt file. Expected '<provider>/<model>' "
                "(e.g., 'openai/gpt-5' or 'local/llama3')."
            )
            app.state.error_message = msg
            raise ValueError(msg)


def _process_rag_and_generation_settings(app: Any) -> None:
    """Process RAG and generation control settings from custom prompt data."""
    # RAG and generation controls
    if "rag" in app.custom_prompt_data:
        app.state.use_rag = bool(app.custom_prompt_data.get("rag"))
    if "top_k" in app.custom_prompt_data:
        _top_k = app.custom_prompt_data.get("top_k")
        if isinstance(_top_k, int):
            app.state.top_k = _top_k
        elif isinstance(_top_k, str) and _top_k.isdigit():
            app.state.top_k = int(_top_k)
        else:
            logger.warning("Invalid top_k in prompt file: %r; keeping existing", _top_k)
    if "retries" in app.custom_prompt_data:
        _retries = app.custom_prompt_data.get("retries")
        if isinstance(_retries, int):
            app.state.retry_attempts = _retries
        elif isinstance(_retries, str) and _retries.isdigit():
            app.state.retry_attempts = int(_retries)
        else:
            logger.warning("Invalid retries in prompt file: %r; keeping existing", _retries)


def _process_model_parameters(app: Any) -> None:
    """Process model parameters from custom prompt data."""
    # Extract model part for temperature support check
    model_value = app.custom_prompt_data.get("model", DEFAULT_MODEL)
    model_part = model_value.split("/")[-1] if "/" in model_value else model_value

    app.state.temperature_supported = supports_temperature(model_part)
    # Set model parameters from prompt file
    model_params = app.custom_prompt_data.get("modelParameters", {})
    if isinstance(model_params, dict):
        if "temperature" in model_params:
            if not app.state.temperature_supported:
                app.state.temperature = 1.0  # enforce
                logger.warning("Temperature not supported for model %s; forcing 1.0", model_part)
            else:
                app.state.temperature = model_params["temperature"]
        if "max_tokens" in model_params:
            app.state.max_tokens = model_params["max_tokens"]
