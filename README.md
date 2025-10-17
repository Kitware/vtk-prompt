# VTK Prompt

[![CI](https://github.com/vicentebolea/vtk-prompt/actions/workflows/ci.yml/badge.svg)](https://github.com/vicentebolea/vtk-prompt/actions/workflows/ci.yml)
[![Build and Publish](https://github.com/vicentebolea/vtk-prompt/actions/workflows/publish.yml/badge.svg)](https://github.com/vicentebolea/vtk-prompt/actions/workflows/publish.yml)
[![PyPI version](https://badge.fury.io/py/vtk-prompt.svg)](https://badge.fury.io/py/vtk-prompt)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A command-line interface and web-based UI for generating VTK visualization code
using Large Language Models (Anthropic Claude, OpenAI GPT, NVIDIA NIM, and local
models).

![Screenshot from 2025-06-11 19-02-00](https://github.com/user-attachments/assets/2e1e85c3-4efd-43e4-810c-185b851d609d)

## Features

- Multiple LLM providers: Anthropic Claude, OpenAI GPT, NVIDIA NIM, and local
  models
- Interactive web UI with live VTK rendering
- Retrieval-Augmented Generation (RAG) with VTK examples database
- Real-time visualization of generated code
- Token usage tracking and cost monitoring
- CLI and Python API for integration

## Installation

### From PyPI (Stable)

```bash
pip install vtk-prompt
```

### From TestPyPI (Latest Development)

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vtk-prompt
```

### From Source

```bash
git clone https://github.com/vicentebolea/vtk-prompt.git
cd vtk-prompt
pip install -e .
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
vtk-prompt "Create a red sphere"

# With RAG enhancement
vtk-prompt "Create a sphere with custom resolution" --rag

# Different providers
vtk-prompt "Create a blue cube" --provider openai
vtk-prompt "Create a cone" --provider nim --token YOUR_NIM_TOKEN
```

## Usage

### Web UI Features

The web interface provides:

- Model selection: Choose between Claude models (Haiku, Sonnet 4) and other
  providers
- Token control: Adjust maximum tokens for responses
- Usage tracking: Real-time display of input/output tokens and costs
- RAG integration: Toggle retrieval-augmented generation for better results
- Live preview: See VTK visualizations rendered in real-time
- Code export: View, edit, and copy generated VTK code
- Local & cloud support: Both cloud APIs and local model endpoints

### Command Line Interface

```bash
# Basic usage
vtk-prompt "Create a red sphere"

# Advanced options
vtk-prompt "Create a textured cone with 32 resolution" \
  --provider anthropic \
  --model claude-opus-4-1-20250805 \
  --max-tokens 4000 \
  --rag \
  --verbose

# Using different providers
vtk-prompt "Create a blue cube" --provider openai --model gpt-5
vtk-prompt "Create a cylinder" --provider nim --model meta/llama3-70b-instruct
```

### RAG (Retrieval Augmented Generation)

Enhance code generation with relevant VTK examples:

1. **Build RAG database** (one-time setup):

```bash
vtk-build-rag
```

2. **Test RAG system** (optional):

```bash
vtk-test-rag "How to create a cube in VTK"
```

3. **Use RAG in queries**:

```bash
vtk-prompt "Create a vtkSphereSource with texture mapping" --rag
```

### Python API

```python
from vtk_prompt import VTKPromptClient

client = VTKPromptClient()
code = client.generate_code("Create a red sphere")
print(code)
```

## Configuration

### Environment Variables

- `ANTHROPIC_API_KEY` - Anthropic Claude API key
- `OPENAI_API_KEY` - OpenAI API key (also used for NVIDIA NIM)

### Supported Providers & Models

| Provider      | Default Model            | Base URL                            |
| ------------- | ------------------------ | ----------------------------------- |
| **anthropic** | claude-opus-4-1-20250805 | https://api.anthropic.com/v1        |
| **openai**    | gpt-5                    | https://api.openai.com/v1           |
| **nim**       | meta/llama3-70b-instruct | https://integrate.api.nvidia.com/v1 |
| **custom**    | User-defined             | User-defined (for local models)     |

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
usage: vtk-prompt [-h] [--provider {anthropic,openai,nim,custom}]
                  [-m MODEL] [-k MAX_TOKENS] [-t TOKEN] [--base-url BASE_URL]
                  [-r] [-v] [--collection COLLECTION] [--database DATABASE]
                  [--top-k TOP_K] input_string

Generate VTK visualization code using Large Language Models

positional arguments:
  input_string          Description of the VTK visualization to generate

options:
  -h, --help            Show this help message and exit
  -m MODEL, --model MODEL
                        Model name to use
  -k MAX_TOKENS, --max-tokens MAX_TOKENS
                        Maximum number of tokens to generate
  -t TOKEN, --token TOKEN
                        API token (defaults to environment variable)
  --base-url BASE_URL   Base URL for API (for custom/local models)
  -r, --rag             Use Retrieval Augmented Generation
  -v, --verbose         Show generated source code
  --provider {anthropic,openai,nim,custom}
                        LLM provider to use

RAG Options:
  --collection COLLECTION
                        Collection name for RAG (default: vtk-examples)
  --database DATABASE   Database path for RAG (default: ./db/codesage-codesage-large-v2)
  --top-k TOP_K         Number of examples to retrieve (default: 5)
```

## Available Commands

- `vtk-prompt` - Main CLI for code generation
- `vtk-prompt-ui` - Launch web interface
- `vtk-build-rag` - Build RAG database from VTK examples
- `vtk-test-rag` - Test RAG functionality
- `gen-vtk-file` - Generate VTK XML files
- `rag-chat` - Interactive RAG chat interface

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Architecture

- **Core**: Python package with CLI and API
- **UI**: Trame-based web interface with VTK rendering
- **RAG**: ChromaDB + Llama Index for code example retrieval
- **Providers**: Unified interface for multiple LLM APIs

## Links

- [PyPI Package](https://pypi.org/project/vtk-prompt/)
- [Documentation](https://github.com/vicentebolea/vtk-prompt)
- [Issues](https://github.com/vicentebolea/vtk-prompt/issues)
- [VTK Documentation](https://vtk.org/documentation/)

---

Made with care for the VTK and scientific visualization community.
