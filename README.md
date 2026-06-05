# VTK Prompt

[![CI](https://github.com/vicentebolea/vtk-prompt/actions/workflows/ci.yml/badge.svg)](https://github.com/vicentebolea/vtk-prompt/actions/workflows/ci.yml)
[![Build and Publish](https://github.com/vicentebolea/vtk-prompt/actions/workflows/publish.yml/badge.svg)](https://github.com/vicentebolea/vtk-prompt/actions/workflows/publish.yml)
[![PyPI version](https://badge.fury.io/py/vtk-prompt.svg)](https://badge.fury.io/py/vtk-prompt)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/github/Kitware/vtk-prompt/graph/badge.svg?token=gg8CHNeBKR)](https://codecov.io/github/Kitware/vtk-prompt)

A command-line interface and web-based UI for generating VTK visualization code
using Large Language Models (Anthropic Claude, OpenAI GPT, NVIDIA NIM, and local
models).

![Screenshot from 2025-06-11 19-02-00](https://github.com/user-attachments/assets/2e1e85c3-4efd-43e4-810c-185b851d609d)

## Features

- Multiple LLM providers: Anthropic Claude, OpenAI GPT, NVIDIA NIM, and local
  models
- Interactive web UI with live VTK rendering
- Context-enhanced generation via [vtk-mcp](https://github.com/Kitware/vtk-mcp): semantic search over VTK examples and docs, class API lookup, and full code validation
- Real-time visualization of generated code
- Token usage tracking and cost monitoring
- CLI and Python API for integration

## Installation

### From PyPI (Stable)

```bash
# pip
pip install vtk-prompt

# uv (recommended)
uv pip install vtk-prompt
```

### From TestPyPI (Latest Development)

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vtk-prompt
```

### From Source

```bash
git clone https://github.com/vicentebolea/vtk-prompt.git
cd vtk-prompt

# pip
pip install -e .

# uv
uv pip install -e .
```

## Quick Start

### 1. Set up API keys

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"  # Optional
```

### 2. Launch Web UI (Recommended)

```bash
vtk-prompt-ui
```

Access the UI at `http://localhost:8080`

### 3. Or use CLI

```bash
# Generate VTK code
vtk-prompt "Create a red sphere" -t $ANTHROPIC_API_KEY

# With vtk-mcp for context-enhanced generation
vtk-prompt "Create a sphere with custom resolution" --mcp-url http://localhost:8000 -t $API_KEY

# Different providers
vtk-prompt "Create a blue cube" --provider openai -t $OPENAI_API_KEY
vtk-prompt "Create a cone" --provider nim --token YOUR_NIM_TOKEN
```

## Usage

### Web UI Features

The web interface provides:

- Model selection: Choose between Claude models (Haiku, Sonnet 4) and other
  providers
- Token control: Adjust maximum tokens for responses
- Usage tracking: Real-time display of input/output tokens and costs
- vtk-mcp integration: Enter a vtk-mcp server URL to enable context retrieval and code validation
- Live preview: See VTK visualizations rendered in real-time
- Code export: View, edit, and copy generated VTK code
- Local & cloud support: Both cloud APIs and local model endpoints

### Command Line Interface

```bash
# Basic usage
vtk-prompt "Create a red sphere" -t $API_KEY

# Advanced options
vtk-prompt "Create a textured cone with 32 resolution" \
  --provider anthropic \
  --model claude-opus-4-7 \
  --max-tokens 4000 \
  --mcp-url http://localhost:8000 \
  --verbose \
  -t $API_KEY

# Using different providers
vtk-prompt "Create a blue cube" --provider openai --model gpt-4.1 -t $OPENAI_API_KEY
vtk-prompt "Create a cylinder" --provider nim --model meta/llama-3.3-70b-instruct -t $NIM_KEY
```

### vtk-mcp Integration

Context-enhanced generation is powered by [vtk-mcp](https://github.com/Kitware/vtk-mcp), a local
MCP server that exposes VTK knowledge tools to the LLM.

1. **Start vtk-mcp** (see its README for setup):

```bash
docker compose up   # or: uvicorn vtk_mcp.transport.http:app --port 8000
```

2. **Use with vtk-prompt**:

```bash
vtk-prompt "Create a vtkSphereSource with texture mapping" --mcp-url http://localhost:8000 -t $API_KEY
```

When `--mcp-url` is set the LLM has access to all vtk-mcp tools during generation
(class lookup, method signatures, import validation, semantic search) and the generated code
is validated against the VTK API with `validate_vtk_code` before being returned.

### Python API

```python
from vtk_prompt import VTKPromptClient

client = VTKPromptClient()
code = client.generate_code("Create a red sphere")
print(code)
```

## Model Configuration

**Model configuration with YAML prompt files:**

```yaml
# Model and parameter configuration
model: anthropic/claude-opus-4-1-20250805
modelParameters:
  temperature: 0.2
  max_tokens: 6000
```

**Using custom prompt files:**

```bash
# CLI: Use your custom prompt file
vtk-prompt "Create a sphere" --prompt-file custom_vtk_prompt.yml

# CLI: Or with additional CLI overrides
vtk-prompt "Create a complex scene" --prompt-file custom_vtk_prompt.yml --retry-attempts 3

# UI: Use your custom prompt file
vtk-prompt-ui --server --prompt-file custom_vtk_prompt.yml
```

### Model Parameters Guide

**Temperature Settings:**

- `0.1-0.3`: More focused, deterministic code generation
- `0.4-0.7`: Balanced creativity and consistency (recommended)
- `0.8-1.0`: More creative but potentially less reliable

**Token Limits:** Token usage can vary significantly between models and
providers. These are general guidelines:

- `1000-2000`: Simple visualizations and basic VTK objects
- `3000-4000`: Complex scenes with multiple objects
- `5000+`: Detailed implementations with extensive documentation

_Note: Different models have different token limits and costs. Check your
provider's documentation for specific model capabilities._

## Testing

```bash
# Run all tests with tox
tox -e test

# Or directly with pytest (pip or uv)
pip install -e ".[test]" && pytest
uv pip install -e ".[test]" && pytest
```

## Configuration

### Environment Variables

- `ANTHROPIC_API_KEY` - Anthropic Claude API key
- `OPENAI_API_KEY` - OpenAI API key (also used for NVIDIA NIM)

### Supported Providers & Models

| Provider      | Default Model                  | Base URL                            |
| ------------- | ------------------------------ | ----------------------------------- |
| **anthropic** | claude-sonnet-4-6              | https://api.anthropic.com/v1        |
| **openai**    | gpt-4.1                        | https://api.openai.com/v1           |
| **gemini**    | gemini-2.5-pro                 | https://generativelanguage.googleapis.com/v1beta |
| **nim**       | meta/llama-3.3-70b-instruct    | https://integrate.api.nvidia.com/v1 |
| **custom**    | User-defined                   | User-defined (for local models)     |

### Custom/Local Models

You can use local models via OpenAI-compatible APIs:

```bash
# Using Ollama
vtk-prompt "Create a sphere" \
  --provider custom \
  --base-url http://localhost:11434/v1 \
  --model llama2

# Using LM Studio
vtk-prompt "Create a cube" \
  --provider custom \
  --base-url http://localhost:1234/v1 \
  --model local-model
```

## CLI Reference

```
Usage: vtk-prompt [OPTIONS] INPUT_STRING

  Generate and execute VTK code using LLMs.

Options:
  --provider [openai|anthropic|gemini|nim]
                                  LLM provider to use
  -m, --model TEXT                Model name to use
  -k, --max-tokens INTEGER        Max # of tokens to generate
  --temperature FLOAT             Temperature for generation (0.0-2.0)
  -t, --token TEXT                API token for the selected provider [required]
  --base-url TEXT                 Base URL for API (auto-detected or custom)
  -v, --verbose                   Show generated source code
  --mcp-url TEXT                  vtk-mcp server URL (enables context retrieval and code validation)
  --top-k INTEGER                 Number of examples to retrieve from vtk-mcp
  --retry-attempts INTEGER        Number of times to retry if validation fails
  --conversation TEXT             Path to conversation file for chat history
  --prompt-file TEXT              Path to custom YAML prompt file
  --help                          Show this message and exit.
```

## Available Commands

- `vtk-prompt` - Main CLI for code generation
- `vtk-prompt-ui` - Launch web interface
- `gen-vtk-file` - Generate VTK XML files

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For detailed development instructions, see [DEVELOPMENT.md](DEVELOPMENT.md)
which covers:

- Setting up the development environment
- Running tests and linting
- Developer mode for the web UI
- Code formatting and type checking
- Pre-commit hooks

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

The project includes a comprehensive test suite with code coverage reporting.

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test categories
pytest tests/test_cli.py                    # CLI functionality tests
pytest tests/test_providers_smoke.py        # Provider smoke tests
pytest -m "not smoke"                       # Exclude API-dependent tests

# Generate coverage report
pytest --cov=src/vtk_prompt --cov-report=html
```

### Test Categories

- **CLI Tests**: Argument parsing, provider integration, error handling
- **Smoke Tests**: Real API connectivity testing (requires API keys)
- **Client Tests**: Core VTKPromptClient functionality
- **Integration Tests**: End-to-end workflow testing

Coverage reports are generated in `htmlcov/` directory. Current coverage:
**11.0%** (improving with ongoing test development).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Architecture

- **Core**: Python package with CLI and API
- **UI**: Trame-based web interface with VTK rendering
- **vtk-mcp**: External MCP server providing VTK knowledge, semantic search, and code validation
- **Providers**: Unified interface for multiple LLM APIs

## Links

- [PyPI Package](https://pypi.org/project/vtk-prompt/)
- [Documentation](https://github.com/vicentebolea/vtk-prompt)
- [Issues](https://github.com/vicentebolea/vtk-prompt/issues)
- [VTK Documentation](https://vtk.org/documentation/)

---

Made with care for the VTK and scientific visualization community.
