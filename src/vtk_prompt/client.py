"""
VTK Code Generation Client.

This module provides the core VTKPromptClient class which handles conversation management,
code generation, execution, and error handling with retry logic.

Features:
- Singleton pattern for conversation persistence
- vtk-mcp integration for context-aware code generation and validation
- Automatic code execution and error handling
- Conversation history management and file persistence
- Multiple model provider support (OpenAI, Anthropic, Gemini, NIM)
- Template-based prompt construction with VTK-specific context
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai

from . import get_logger
from .prompts import assemble_vtk_prompt
from .provider_utils import DEFAULT_MODEL
from .utils.helpers import ensure_vtk_importable

logger = get_logger(__name__)


@dataclass
class VTKPromptClient:
    """OpenAI client for VTK code generation."""

    _instance: "VTKPromptClient" | None = None
    _initialized: bool = False
    verbose: bool = False
    conversation_file: str | None = None
    conversation: list[dict[str, str]] | None = None
    mcp_url: str | None = None

    def __new__(cls, **kwargs: Any) -> "VTKPromptClient":
        """Create singleton instance of VTKPromptClient."""
        # Make sure that this is a singleton
        if cls._instance is None:
            cls._instance = super(VTKPromptClient, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance.conversation = []
        return cls._instance

    def __post_init__(self) -> None:
        """Post-init hook to prevent double initialization in singleton."""
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

    def load_conversation(self) -> list[dict[str, str]]:
        """Load conversation history from file."""
        if not self.conversation_file or not Path(self.conversation_file).exists():
            return []

        try:
            with open(self.conversation_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    logger.warning("Invalid conversation file format, no history loaded.")
                    return []
        except Exception as e:
            logger.error("Could not load conversation file: %s", e)
            return []

    def save_conversation(self) -> None:
        """Save conversation history to file."""
        if not self.conversation_file or not self.conversation:
            return

        try:
            # Ensure directory exists
            Path(self.conversation_file).parent.mkdir(parents=True, exist_ok=True)

            with open(self.conversation_file, "w") as f:
                json.dump(self.conversation, f, indent=2)
        except Exception as e:
            logger.error("Could not save conversation file: %s", e)

    def update_conversation(
        self, new_convo: list[dict[str, str]], new_convo_file: str | None = None
    ) -> None:
        """Update conversation history with new conversation."""
        if not self.conversation:
            self.conversation = []
        self.conversation.extend(new_convo)

        if new_convo_file:
            self.conversation_file = new_convo_file

    def validate_code_syntax(self, code_string: str) -> tuple[bool, str | None]:
        """Validate Python code syntax using AST."""
        try:
            ast.parse(code_string)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e.msg} at line {e.lineno}"
        except Exception as e:
            return False, f"AST parsing error: {str(e)}"

    def run_code(self, code_string: str) -> None:
        """Execute VTK code using exec() after AST validation."""
        is_valid, error_msg = self.validate_code_syntax(code_string)
        if not is_valid:
            logger.error("Code validation failed: %s", error_msg)
            if self.verbose:
                logger.debug("Generated code:\n%s", code_string)
            return

        if self.verbose:
            logger.debug("Executing code:\n%s", code_string)

        try:
            exec(code_string, globals(), {})
        except Exception as e:
            logger.error("Error executing code: %s", e)
            if not self.verbose:
                logger.debug("Failed code:\n%s", code_string)
            return

    def _validate_and_extract_model_params(
        self,
        custom_prompt_data: dict,
        default_model: str,
        default_temperature: float,
        default_max_tokens: int,
    ) -> tuple[str, float, int, list[str]]:
        """Validate and extract model parameters from custom prompt data.

        Args:
            custom_prompt_data: The loaded YAML prompt data
            default_model: Default model to use if validation fails
            default_temperature: Default temperature to use if validation fails
            default_max_tokens: Default max_tokens to use if validation fails

        Returns:
            Tuple of (model, temperature, max_tokens, warnings)
        """
        from .provider_utils import (
            get_available_models,
            get_default_model,
            get_supported_providers,
            supports_temperature,
        )

        warnings = []
        model = default_model
        temperature = default_temperature
        max_tokens = default_max_tokens

        # Extract model if present
        if "model" in custom_prompt_data:
            custom_model = custom_prompt_data["model"]

            # Validate model format: should be "provider/model"
            if "/" not in custom_model:
                warnings.append(
                    f"Invalid model format '{custom_model}'. Expected 'provider/model'. "
                    f"Using default: {default_model}"
                )
            else:
                provider, model_name = custom_model.split("/", 1)

                # Validate provider is supported
                if provider not in get_supported_providers():
                    warnings.append(
                        f"Unsupported provider '{provider}'. "
                        f"Supported providers: {', '.join(get_supported_providers())}. "
                        f"Using default: {default_model}"
                    )
                else:
                    # Validate model exists for provider
                    available_models = get_available_models().get(provider, [])
                    if model_name not in available_models:
                        warnings.append(
                            f"Model '{model_name}' not in curated list for provider '{provider}'. "
                            f"Available: {', '.join(available_models)}. "
                            f"Using default: {get_default_model(provider)}"
                        )
                        model = f"{provider}/{get_default_model(provider)}"
                    else:
                        model = custom_model

        # Extract modelParameters if present
        if "modelParameters" in custom_prompt_data:
            params = custom_prompt_data["modelParameters"]

            # Validate temperature
            if "temperature" in params:
                try:
                    custom_temp = float(params["temperature"])
                    if 0.0 <= custom_temp <= 2.0:
                        temperature = custom_temp

                        # Check if model supports temperature
                        if not supports_temperature(model):
                            warnings.append(
                                f"Model '{model}' does not support temperature control. "
                                f"Temperature will be set to 1.0."
                            )
                            temperature = 1.0
                    else:
                        warnings.append(
                            f"Temperature {custom_temp} out of range [0.0, 2.0]. "
                            f"Using default: {default_temperature}"
                        )
                except (ValueError, TypeError):
                    warnings.append(
                        f"Invalid temperature value '{params['temperature']}'. "
                        f"Using default: {default_temperature}"
                    )

            # Validate max_tokens
            if "max_tokens" in params:
                try:
                    custom_max_tokens = int(params["max_tokens"])
                    if 1 <= custom_max_tokens <= 100000:
                        max_tokens = custom_max_tokens
                    else:
                        warnings.append(
                            f"max_tokens {custom_max_tokens} out of range [1, 100000]. "
                            f"Using default: {default_max_tokens}"
                        )
                except (ValueError, TypeError):
                    warnings.append(
                        f"Invalid max_tokens value '{params['max_tokens']}'. "
                        f"Using default: {default_max_tokens}"
                    )

        return model, temperature, max_tokens, warnings

    def _format_custom_prompt(
        self, custom_prompt_data: dict, message: str, context_snippets: str | None = None
    ) -> list[dict[str, str]]:
        """Format custom prompt data into messages for LLM client.

        Args:
            custom_prompt_data: The loaded YAML prompt data
            message: The user request
            context_snippets: Optional context snippets from vtk-mcp

        Returns:
            Formatted messages ready for LLM client

        Note:
            This method does NOT extract model/modelParameters from custom_prompt_data.
            That is handled by _validate_and_extract_model_params() in the query() method.
        """
        from .prompts import PYTHON_VERSION, VTK_VERSION, YAMLPromptLoader

        # Prepare variables for substitution
        variables = {
            "VTK_VERSION": VTK_VERSION,
            "PYTHON_VERSION": PYTHON_VERSION,
            "request": message,
        }

        if context_snippets:
            variables["context_snippets"] = context_snippets

        # Process messages from custom prompt
        messages = custom_prompt_data.get("messages", [])
        formatted_messages = []

        yaml_loader = YAMLPromptLoader()
        for msg in messages:
            content = yaml_loader.substitute_yaml_variables(msg.get("content", ""), variables)
            formatted_messages.append({"role": msg.get("role", "user"), "content": content})

        return formatted_messages

    def query(
        self,
        message: str = "",
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        top_k: int = 5,
        retry_attempts: int = 1,
        provider: str | None = None,
        custom_prompt: dict | None = None,
        ui_mode: bool = False,
        execution_error: str | None = None,
    ) -> tuple[str, str, Any] | tuple[str, str, Any, list[str]] | str:
        """Generate VTK code using vtk-mcp tools when available.

        Args:
            message: The user query
            api_key: API key for the service
            model: Model name to use
            base_url: API base URL
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            top_k: Number of context snippets to retrieve from vtk-mcp
            retry_attempts: Number of times to retry if AST validation fails
            provider: LLM provider to use (overrides instance provider if provided)
            custom_prompt: Custom YAML prompt data (overrides built-in prompts)
            ui_mode: Whether the request is coming from UI (affects prompt selection)
        """
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("No API key provided. Set OPENAI_API_KEY or pass api_key parameter.")

        # Create client with current parameters
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # Load existing conversation if present
        if self.conversation_file and not self.conversation:
            self.conversation = self.load_conversation()

        if not message and not self.conversation:
            raise ValueError("No prompt or conversation file provided")

        # Set up vtk-mcp client (context retrieval, tool calling, and code validation)
        mcp_client = None
        if self.mcp_url:
            from .vtk_mcp_client import VTKMCPClient, check_mcp_available

            if check_mcp_available(self.mcp_url):
                mcp_client = VTKMCPClient(self.mcp_url)
            else:
                logger.warning("vtk-mcp server not available at %s", self.mcp_url)

        # Store validation warnings to return to caller
        validation_warnings = []

        if execution_error:
            # Retry after execution failure: append error and let LLM fix it with tools
            if not self.conversation:
                raise ValueError("No conversation to retry")
            self.conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"The generated code failed at runtime with this error:\n"
                        f"{execution_error}\n"
                        f"Please use the available tools to look up the correct VTK API "
                        f"and provide corrected code."
                    ),
                }
            )
        else:
            # Normal path: build context and prompt
            context_snippets = None
            if mcp_client:
                mcp_context = mcp_client.get_enriched_context(message, top_k=top_k)
                if mcp_context:
                    context_snippets = mcp_context

            if custom_prompt:
                validated_model, validated_temp, validated_max_tokens, warnings = (
                    self._validate_and_extract_model_params(
                        custom_prompt, model, temperature, max_tokens
                    )
                )
                validation_warnings.extend(warnings)
                model = validated_model
                temperature = validated_temp
                max_tokens = validated_max_tokens
                for warning in warnings:
                    logger.warning(warning)
                yaml_messages = self._format_custom_prompt(
                    custom_prompt, message, context_snippets
                )
                if not yaml_messages:
                    logger.warning(
                        "custom_prompt provided but defines no 'messages'; "
                        "falling back to built-in prompt assembly"
                    )
                elif self.verbose:
                    logger.debug("Using custom YAML prompt from file")
            else:
                yaml_messages = []

            # Fall back to component assembly when no usable custom messages exist.
            if not yaml_messages:
                from .prompts import PYTHON_VERSION, VTK_VERSION

                prompt_data = assemble_vtk_prompt(
                    request=message,
                    ui_mode=ui_mode,
                    context_snippets=context_snippets,
                    VTK_VERSION=VTK_VERSION,
                    PYTHON_VERSION=PYTHON_VERSION,
                )
                yaml_messages = prompt_data["messages"]
                if self.verbose:
                    mode_str = "UI" if ui_mode else "CLI"
                    mcp_str = " + MCP" if mcp_client else ""
                    logger.debug(f"Using component assembly ({mode_str}{mcp_str})")

            if not self.conversation:
                self.conversation = list(yaml_messages)
            elif message and yaml_messages:
                self.conversation.append(yaml_messages[-1])

        # Fetch vtk-mcp tools for LLM tool calling
        tools = mcp_client.list_tools() if mcp_client else []

        # Retry loop for AST validation
        for attempt in range(retry_attempts):
            if self.verbose:
                if attempt > 0:
                    logger.debug("Retry attempt %d/%d", attempt + 1, retry_attempts)
                logger.debug("Making request with model: %s, temperature: %s", model, temperature)

            # Agentic tool-calling loop: LLM calls vtk-mcp tools until it generates code
            MAX_TOOL_ROUNDS = 10
            content = None
            response = None
            for _ in range(MAX_TOOL_ROUNDS):
                response = client.chat.completions.create(
                    model=model,
                    messages=self.conversation,  # type: ignore[arg-type]
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    **(  # type: ignore[call-overload]
                        {"tools": tools, "tool_choice": "auto"} if tools else {}
                    ),
                )

                if not (hasattr(response, "choices") and response.choices):
                    break

                choice = response.choices[0]

                if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                    # Add assistant message with tool calls to conversation
                    tc_msg: dict = {"role": "assistant", "content": choice.message.content or ""}
                    tc_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ]
                    self.conversation.append(tc_msg)

                    # Execute each tool and append results
                    for tc in choice.message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except Exception:
                            args = {}
                        result = mcp_client.call_tool(tc.function.name, args)  # type: ignore
                        logger.debug("Tool %s -> %s...", tc.function.name, result[:80])
                        self.conversation.append(
                            {"role": "tool", "tool_call_id": tc.id, "content": result}
                        )
                    continue  # let LLM decide what to do next

                # LLM generated a response (not a tool call)
                content = choice.message.content or "No content in response"
                break

            if content is None:
                # Tool loop exhausted without a text response
                if attempt == retry_attempts - 1:
                    return ("No response generated", "", getattr(response, "usage", None) or {})
                continue

            finish_reason = (
                response.choices[0].finish_reason
                if response and hasattr(response, "choices") and response.choices
                else None
            )
            if finish_reason == "length":
                raise ValueError(
                    f"Output was truncated due to max_tokens limit ({max_tokens}).\n"
                    "Please increase max_tokens."
                )

            expl_matches = re.findall("<explanation>(.*?)</explanation>", content, re.DOTALL)
            code_matches = re.findall("<code>(.*?)</code>", content, re.DOTALL)

            if not expl_matches or not code_matches:
                if attempt < retry_attempts - 1:
                    self.conversation.append({"role": "assistant", "content": content})
                    self.conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Please format your response with "
                                "<explanation>...</explanation> and <code>...</code> tags."
                            ),
                        }
                    )
                    continue
                return (content, "", getattr(response, "usage", None) or {})

            generated_explanation = expl_matches[0]
            generated_code = code_matches[0]

            generated_code = ensure_vtk_importable(generated_code)

            is_valid, error_msg = self.validate_code_syntax(generated_code)
            if is_valid:
                # Run full VTK API validation via vtk-mcp if available
                if mcp_client:
                    vtk_diagnostics = mcp_client.validate_code(generated_code)
                    if vtk_diagnostics:
                        if attempt < retry_attempts - 1:
                            logger.debug("vtk-mcp validation issues found, retrying")
                            self.conversation.append({"role": "assistant", "content": content})
                            self.conversation.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "The generated code has VTK API issues:\n"
                                        f"{vtk_diagnostics}\n"
                                        "Please use the available tools to fix them and regenerate."
                                    ),
                                }
                            )
                            continue
                        else:
                            validation_warnings.append(
                                f"VTK API issues found:\n{vtk_diagnostics}"
                            )
                if message:
                    self.conversation.append({"role": "assistant", "content": content})
                    self.save_conversation()
                if validation_warnings:
                    return (
                        generated_explanation,
                        generated_code,
                        getattr(response, "usage", None),
                        validation_warnings,
                    )
                return generated_explanation, generated_code, getattr(response, "usage", None)

            elif attempt < retry_attempts - 1:
                if self.verbose:
                    logger.warning("AST validation failed: %s. Retrying...", error_msg)
                self.conversation.append({"role": "assistant", "content": content})
                self.conversation.append(
                    {
                        "role": "user",
                        "content": (
                            f"The generated code has a syntax error: {error_msg}. "
                            "Please fix the syntax and generate valid Python code."
                        ),
                    }
                )
            else:
                if self.verbose:
                    logger.error("Final attempt failed AST validation: %s", error_msg)
                if message:
                    self.conversation.append({"role": "assistant", "content": content})
                    self.save_conversation()
                if validation_warnings:
                    return (
                        generated_explanation,
                        generated_code,
                        getattr(response, "usage", None) or {},
                        validation_warnings,
                    )
                return (
                    generated_explanation,
                    generated_code,
                    getattr(response, "usage", None) or {},
                )

        return "No response generated"
