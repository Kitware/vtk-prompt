"""
Component-based prompt assembly system for VTK Prompt.

This module provides a flexible system for assembling prompts from reusable
file-based components. Components are stored as YAML files and can be composed
programmatically to create different prompt variations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from functools import lru_cache

from .yaml_prompt_loader import YAMLPromptLoader

# Global loader instance for variable substitution
_yaml_loader = YAMLPromptLoader()


class PromptComponentLoader:
    """Load and cache prompt components from files."""

    def __init__(self, components_dir: Optional[Path] = None):
        """Initialize component loader.

        Args:
            components_dir: Directory containing component YAML files
        """
        self.components_dir = components_dir or Path(__file__).parent / "components"

        # Ensure components directory exists
        if not self.components_dir.exists():
            raise FileNotFoundError(f"Components directory not found: {self.components_dir}")

    @lru_cache(maxsize=32)
    def load_component(self, component_name: str) -> Dict[str, Any]:
        """Load a component file with caching.

        Args:
            component_name: Name of component file (without .yml extension)

        Returns:
            Component data from YAML file

        Raises:
            FileNotFoundError: If component file doesn't exist
        """
        component_file = self.components_dir / f"{component_name}.yml"
        if not component_file.exists():
            raise FileNotFoundError(f"Component not found: {component_file}")

        with open(component_file) as f:
            return yaml.safe_load(f)

    def clear_cache(self):
        """Clear component cache (useful for development)."""
        self.load_component.cache_clear()

    def list_components(self) -> List[str]:
        """List available component names."""
        return [f.stem for f in self.components_dir.glob("*.yml")]


class VTKPromptAssembler:
    """Assemble VTK prompts from file-based components."""

    def __init__(self, loader: Optional[PromptComponentLoader] = None):
        """Initialize prompt assembler.

        Args:
            loader: Component loader instance (creates default if None)
        """
        self.loader = loader or PromptComponentLoader()
        self.messages: List[Dict[str, str]] = []
        self.model_params: Dict[str, Any] = {}

    def add_component(self, component_name: str) -> "VTKPromptAssembler":
        """Add a component from file.

        Args:
            component_name: Name of component to add

        Returns:
            Self for method chaining
        """
        component = self.loader.load_component(component_name)

        if "role" in component:
            # It's a message component
            message = {"role": component["role"], "content": component["content"]}

            # Handle append/prepend logic for user messages
            if component.get("append") and self.messages and self.messages[-1]["role"] == "user":
                # Append to the last user message
                self.messages[-1]["content"] += "\n\n" + component["content"]
            elif component.get("prepend") and self.messages and self.messages[-1]["role"] == "user":
                # Prepend to the last user message
                self.messages[-1]["content"] = (
                    component["content"] + "\n\n" + self.messages[-1]["content"]
                )
            else:
                # Add as new message
                self.messages.append(message)

        # Handle model parameters
        if "model" in component or "modelParameters" in component:
            self.model_params.update(
                {k: v for k, v in component.items() if k in ["model", "modelParameters"]}
            )

        return self

    def when(self, condition: bool, component_name: str) -> "VTKPromptAssembler":
        """Conditionally add a component.

        Args:
            condition: Whether to add the component
            component_name: Name of component to add if condition is True

        Returns:
            Self for method chaining
        """
        if condition:
            self.add_component(component_name)
        return self

    def add_request(self, request: str) -> "VTKPromptAssembler":
        """Add the user request as a message.

        Args:
            request: User's request text

        Returns:
            Self for method chaining
        """
        self.messages.append({"role": "user", "content": f"Request: {request}"})
        return self

    def substitute_variables(self, **variables: Any) -> "VTKPromptAssembler":
        """Substitute variables in all message content.

        Args:
            **variables: Variables to substitute in {{variable}} format

        Returns:
            Self for method chaining
        """
        for message in self.messages:
            message["content"] = _yaml_loader.substitute_yaml_variables(
                message["content"], variables
            )

        return self

    def build(self) -> Dict[str, Any]:
        """Build the final prompt data.

        Returns:
            Dictionary with 'messages' and model parameters
        """
        result = {
            "messages": self.messages.copy(),
        }
        result.update(self.model_params)
        return result

    def reset(self) -> "VTKPromptAssembler":
        """Reset the assembler to empty state.

        Returns:
            Self for method chaining
        """
        self.messages.clear()
        self.model_params.clear()
        return self


def assemble_vtk_prompt(
    request: str,
    ui_mode: bool = False,
    rag_enabled: bool = False,
    context_snippets: Optional[str] = None,
    **variables: Any,
) -> Dict[str, Any]:
    """Assemble VTK prompt from file-based components.

    Args:
        request: User's request text
        ui_mode: Whether to include UI-specific instructions
        rag_enabled: Whether to include RAG context
        context_snippets: RAG context snippets (required if rag_enabled=True)
        **variables: Additional variables for substitution

    Returns:
        Complete prompt data ready for LLM client

    Raises:
        ValueError: If rag_enabled=True but context_snippets is empty
    """
    if rag_enabled and not context_snippets:
        raise ValueError("context_snippets required when rag_enabled=True")

    assembler = VTKPromptAssembler()

    # Always add base components in order
    assembler.add_component("model_defaults")
    assembler.add_component("base_system")
    assembler.add_component("vtk_instructions")

    # Conditional components
    assembler.when(rag_enabled, "rag_context")
    assembler.when(ui_mode, "ui_renderer")

    # Always add output format and request
    assembler.add_component("output_format")
    assembler.add_request(request)

    # Variable substitution with defaults
    default_vars = {
        "VTK_VERSION": variables.get("VTK_VERSION", "9.5.0"),
        "PYTHON_VERSION": variables.get("PYTHON_VERSION", ">=3.10"),
        "context_snippets": context_snippets or "",
    }
    default_vars.update(variables)

    assembler.substitute_variables(**default_vars)

    return assembler.build()
