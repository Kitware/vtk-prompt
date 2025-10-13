# Development Guide

## Setting up development environment

```bash
git clone https://github.com/vicentebolea/vtk-prompt.git
cd vtk-prompt
pip install -e ".[all]"
```

## Running tests

### Test Suite

The project includes a comprehensive test suite focused on the prompt assembly
system:

```bash
# Run all tests with tox
tox -e test

# Run specific test file
python -m pytest tests/test_prompt_assembly.py -v

# Run specific test methods
python -m pytest tests/test_prompt_assembly.py::TestPromptAssembly::test_default_values -v
```

### Manual Testing

```bash
# Test CLI installation and basic functionality
vtk-prompt --help
vtk-prompt-ui --help
```

## Building package

```bash
python -m build
```

## Logging

The vtk-prompt package uses structured logging throughout. You can control
logging behavior via environment variables or programmatically.

### Setting Log Level

```bash
# Set log level via environment variable
export VTK_PROMPT_LOG_LEVEL=DEBUG

# Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
export VTK_PROMPT_LOG_LEVEL=INFO  # Default
```

### Logging to File

```bash
# Via environment variable (recommended)
export VTK_PROMPT_LOG_FILE="vtk-prompt.log"

# Or programmatically
setup_logging(level="DEBUG", log_file="vtk-prompt.log")
```

## Component System Architecture

The VTK Prompt system uses a modular component-based architecture for prompt
assembly.

### Overview

The component system allows you to:

- **Compose prompts** from reusable YAML files
- **Inject variables** dynamically (`{{VAR_NAME}}`)
- **Configure model parameters** per component
- **Conditionally include** components based on context

### Component Structure

Components are YAML files stored in `src/vtk_prompt/prompts/components/`:

```yaml
# example_component.yml
role: system | user | assistant
content: |
  Your prompt content here with {{VARIABLE}} substitution.
  VTK Version: {{VTK_VERSION}}

# Optional: Merge with existing message instead of creating new one
append: true | false # Add content after existing user message (e.g., additional instructions)
prepend: true | false # Add content before existing user message (e.g., context injection)

# Optional: Model configuration
model: "openai/gpt-5"
modelParameters:
  temperature: 0.5
  max_tokens: 4000
```

### Updating Existing Components

When modifying components:

1. **Preserve backward compatibility** - existing variable names
2. **Test thoroughly** - run full test suite
3. **Document changes** - update component comments
4. **Version carefully** - consider impact on existing prompts

### Component Loading System

The system uses these key classes:

- **`PromptComponentLoader`**: Loads and caches YAML files
- **`VTKPromptAssembler`**: Chains components together
- **`YAMLPromptLoader`**: Handles variable substitution
- **`assemble_vtk_prompt()`**: High-level convenience function

### Variable Substitution

Components support these built-in variables:

- `{{VTK_VERSION}}` - Current VTK version (e.g., "9.5.0")
- `{{PYTHON_VERSION}}` - Python requirements (e.g., ">=3.10")

Custom variables can be passed via:

```python
assembler.substitute_variables(CUSTOM_VAR="value")
# or
assemble_vtk_prompt("request", CUSTOM_VAR="value")
```

## Developer Mode

The web UI includes a developer mode that enables hot reload and debug logging
for faster development cycles.

### Running in Debug Mode

```bash
# Enable debug mode
vtk-prompt-ui --debug

# With custom port and host
vtk-prompt-ui --debug --port 9090 --host 0.0.0.0

# Don't auto-open browser
vtk-prompt-ui --debug --server

# See all available server options
vtk-prompt-ui --help
```

### Developer Mode Features

- **Hot Reload**: UI changes are automatically reflected without server restart
- **Debug Logging**: All log levels displayed for better debugging

## Hot Reload for Iterative Development

For faster iterative development, you can use Trame's `@hot_reload` decorator to
automatically reload specific functions when files change:

```python
from trame_server.utils.hot_reload import hot_reload


@hot_reload
def my_function():
    # This function will reload when the file is saved
    print("Updated function content")
```

## Linting, formatting, and type checking

#### Installation and Setup

```bash
# Install development dependencies
pip install -e ".[dev]"
```

#### Available Commands

```bash
# Format code
tox -e format

# Check formatting without making changes
tox -e format-check

# Lint code
tox -e lint

# Type check
tox -e type

# Run tests
tox -e test

# Run all
tox

# Recreate environments
tox -r
```

### Pre-commit Hooks

Pre-commit hooks automatically run code quality checks before each commit.

#### Installation and Setup

```bash
# Install the hooks
pre-commit install

# Run hooks on all files
pre-commit run --all-files

# Run all hooks on staged files
pre-commit run
```

### Troubleshooting

#### Import errors in mypy

Add missing type stubs to `pyproject.toml` under `[tool.mypy]` section.

#### Skipping Hooks

```bash
# Skip all pre-commit hooks for a commit
git commit --no-verify -m "commit message"

# Skip specific hook types
SKIP=flake8,mypy git commit -m "commit message"
```

**NOTE**: Only skip hooks for local work-in-progress commits. CI will still run
all checks independently and may fail if issues exist.
