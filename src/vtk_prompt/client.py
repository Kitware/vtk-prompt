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
from .prompts import (
    get_no_rag_context,
    get_python_role,
    get_rag_context,
)

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
    ) -> Union[tuple[str, str, Any], str]:
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
                top_k=top_k,
            )

            if not rag_snippets:
                raise ValueError("Failed to load RAG snippets")

            context_snippets = "\n\n".join(rag_snippets["code_snippets"])
            context = get_rag_context(message, context_snippets)

            if self.verbose:
                logger.debug("RAG context: %s", context)
                references = rag_snippets.get("references")
                if references:
                    logger.info("Using examples from: %s", ", ".join(references))
        else:
            context = get_no_rag_context(message)
            if self.verbose:
                logger.debug("No-RAG context: %s", context)

        # Initialize conversation with system message if empty
        if not self.conversation:
            self.conversation = []
            self.conversation.append({"role": "system", "content": get_python_role()})

        # Add current user message
        if message:
            self.conversation.append({"role": "user", "content": context})

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
                    return (
                        generated_explanation,
                        generated_code,
                        response.usage or {},
                    )  # Return anyway, let caller handle
            else:
                if attempt == retry_attempts - 1:
                    return ("No response generated", "", response.usage or {})

        return "No response generated"
