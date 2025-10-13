"""
VTK Code Generation Client.

This module provides the core VTKPromptClient class which handles conversation management,
code generation, execution, and error handling with retry logic.

Features:
- Singleton pattern for conversation persistence
- RAG (Retrieval-Augmented Generation) integration for context-aware code generation
- Automatic code execution and error handling
- Conversation history management and file persistence
- Multiple model provider support (OpenAI, Anthropic, Gemini, NIM)
- Template-based prompt construction with VTK-specific context
"""

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import openai

from . import get_logger
from .prompts import assemble_vtk_prompt

logger = get_logger(__name__)


@dataclass
class VTKPromptClient:
    """OpenAI client for VTK code generation."""

    _instance: Optional["VTKPromptClient"] = None
    _initialized: bool = False
    collection_name: str = "vtk-examples"
    database_path: str = "./db/codesage-codesage-large-v2"
    verbose: bool = False
    conversation_file: Optional[str] = None
    conversation: Optional[list[dict[str, str]]] = None

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
        self, new_convo: list[dict[str, str]], new_convo_file: Optional[str] = None
    ) -> None:
        """Update conversation history with new conversation."""
        if not self.conversation:
            self.conversation = []
        self.conversation.extend(new_convo)

        if new_convo_file:
            self.conversation_file = new_convo_file

    def validate_code_syntax(self, code_string: str) -> tuple[bool, Optional[str]]:
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

    def _extract_context_snippets(self, rag_snippets: dict) -> str:
        """Extract and format RAG context snippets.

        Args:
            rag_snippets: RAG snippets dictionary

        Returns:
            Formatted context snippets string
        """
        return "\n\n".join(rag_snippets["code_snippets"])

    def _format_custom_prompt(
        self, custom_prompt_data: dict, message: str, rag_snippets: Optional[dict] = None
    ) -> list[dict[str, str]]:
        """Format custom prompt data into messages for LLM client.

        Args:
            custom_prompt_data: The loaded YAML prompt data
            message: The user request
            rag_snippets: Optional RAG snippets for context enhancement

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

        # Add RAG context if available
        if rag_snippets:
            variables["context_snippets"] = self._extract_context_snippets(rag_snippets)

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
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        base_url: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        top_k: int = 5,
        rag: bool = False,
        retry_attempts: int = 1,
        provider: Optional[str] = None,
        custom_prompt: Optional[dict] = None,
        ui_mode: bool = False,
    ) -> Union[tuple[str, str, Any], tuple[str, str, Any, list[str]], str]:
        """Generate VTK code with optional RAG enhancement and retry logic.

        Args:
            message: The user query
            api_key: API key for the service
            model: Model name to use
            base_url: API base URL
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            top_k: Number of RAG examples to retrieve
            rag: Whether to use RAG enhancement
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

        if rag:
            from .rag_chat_wrapper import (
                check_rag_components_available,
                get_rag_snippets,
            )

            if not check_rag_components_available():
                raise ValueError("RAG components not available")
            rag_snippets = get_rag_snippets(
                message,
                collection_name=self.collection_name,
                database_path=self.database_path,
                top_k=5,  # Use default top_k value
            )

        # Store validation warnings to return to caller
        validation_warnings = []

        # Use custom prompt if provided, otherwise use component-based assembly
        if custom_prompt:
            # Validate and extract model parameters from custom prompt
            validated_model, validated_temp, validated_max_tokens, warnings = (
                self._validate_and_extract_model_params(
                    custom_prompt, model, temperature, max_tokens
                )
            )
            validation_warnings.extend(warnings)

            # Apply validated parameters
            model = validated_model
            temperature = validated_temp
            max_tokens = validated_max_tokens

            # Log warnings
            for warning in warnings:
                logger.warning(warning)

            # Process custom prompt data
            yaml_messages = self._format_custom_prompt(
                custom_prompt, message, rag_snippets if rag else None
            )
            if self.verbose:
                logger.debug("Using custom YAML prompt from file")
                logger.debug(
                    "Applied model: %s, temperature: %s, max_tokens: %s",
                    model,
                    temperature,
                    max_tokens,
                )
        else:
            # Use component-based assembly system (now the default and only option)
            from .prompts import PYTHON_VERSION, VTK_VERSION

            context_snippets = None
            if rag and rag_snippets:
                context_snippets = self._extract_context_snippets(rag_snippets)

            prompt_data = assemble_vtk_prompt(
                request=message,
                ui_mode=ui_mode,
                rag_enabled=rag,
                context_snippets=context_snippets,
                VTK_VERSION=VTK_VERSION,
                PYTHON_VERSION=PYTHON_VERSION,
            )
            yaml_messages = prompt_data["messages"]

            if self.verbose:
                mode_str = "UI" if ui_mode else "CLI"
                rag_str = " + RAG" if rag else ""
                logger.debug(f"Using component assembly ({mode_str}{rag_str})")

                if rag and rag_snippets:
                    references = rag_snippets.get("references")
                    if references:
                        logger.info("Using examples from: %s", ", ".join(references))

        # Initialize conversation with YAML messages if empty
        if not self.conversation:
            self.conversation = []
            # Add all messages from YAML prompt (system + user)
            self.conversation.extend(yaml_messages)
        else:
            # If conversation exists, just add the user message (last message from YAML)
            if message and yaml_messages:
                self.conversation.append(yaml_messages[-1])

        # Retry loop for AST validation
        for attempt in range(retry_attempts):
            if self.verbose:
                if attempt > 0:
                    logger.debug("Retry attempt %d/%d", attempt + 1, retry_attempts)
                logger.debug("Making request with model: %s, temperature: %s", model, temperature)
                for i, msg in enumerate(self.conversation):
                    logger.debug("Message %d (%s): %s...", i, msg["role"], msg["content"][:100])

            response = client.chat.completions.create(
                model=model,
                messages=self.conversation,  # type: ignore[arg-type]
                max_completion_tokens=max_tokens,
                # max_tokens=max_tokens,
                temperature=temperature,
            )

            if hasattr(response, "choices") and len(response.choices) > 0:
                content = response.choices[0].message.content or "No content in response"
                finish_reason = response.choices[0].finish_reason

                if finish_reason == "length":
                    raise ValueError(
                        f"Output was truncated due to max_tokens limit ({max_tokens}).\n"
                        "Please increase max_tokens."
                    )

                generated_explanation = re.findall(
                    "<explanation>(.*?)</explanation>", content, re.DOTALL
                )[0]
                generated_code = re.findall("<code>(.*?)</code>", content, re.DOTALL)[0]
                if "import vtk" not in generated_code:
                    generated_code = "import vtk\n" + generated_code
                else:
                    pos = generated_code.find("import vtk")
                    if pos != -1:
                        generated_code = generated_code[pos:]
                    else:
                        generated_code = generated_code

                is_valid, error_msg = self.validate_code_syntax(generated_code)
                if is_valid:
                    if message:
                        self.conversation.append({"role": "assistant", "content": content})
                        self.save_conversation()
                    # Return warnings if custom prompt was used
                    if validation_warnings:
                        return (
                            generated_explanation,
                            generated_code,
                            response.usage,
                            validation_warnings,
                        )
                    return generated_explanation, generated_code, response.usage

                elif attempt < retry_attempts - 1:  # Don't log on last attempt
                    if self.verbose:
                        logger.warning("AST validation failed: %s. Retrying...", error_msg)
                    # Add error feedback to context for retry
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
                    # Last attempt failed
                    if self.verbose:
                        logger.error("Final attempt failed AST validation: %s", error_msg)

                    if message:
                        self.conversation.append({"role": "assistant", "content": content})
                        self.save_conversation()
                    # Return warnings if custom prompt was used
                    if validation_warnings:
                        return (
                            generated_explanation,
                            generated_code,
                            response.usage or {},
                            validation_warnings,
                        )
                    return (
                        generated_explanation,
                        generated_code,
                        response.usage or {},
                    )  # Return anyway, let caller handle
            else:
                if attempt == retry_attempts - 1:
                    return ("No response generated", "", response.usage or {})

        return "No response generated"
