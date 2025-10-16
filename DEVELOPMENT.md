# Development Guide

## Setting up development environment

```bash
git clone https://github.com/vicentebolea/vtk-prompt.git
cd vtk-prompt
pip install -e ".[all]"
```

## Running tests

```bash
# Lint and format
black src/
flake8 src/

# Test installation
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
