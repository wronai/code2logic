# LOLM - Lightweight Orchestrated LLM Manager

[![PyPI version](https://badge.fury.io/py/lolm.svg)](https://badge.fury.io/py/lolm)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Reusable LLM configuration and management package for Python projects.**

Multi-provider support with automatic fallback, priority routing, and unified configuration.

## ✨ Features

- 🔄 **Multi-provider support** - OpenRouter, Ollama, Groq, Together, LiteLLM
- ⚡ **Automatic fallback** - Seamless provider switching on failure
- 🎯 **Priority routing** - Configure provider order by priority
- 🔧 **Unified config** - `.env`, `litellm_config.yaml`, `~/.lolm/`
- 🖥️ **CLI interface** - Similar to `reclapp llm`

## 🚀 Installation

```bash
pip install lolm
```

Or with optional dependencies:

```bash
pip install lolm[full]      # All providers
pip install lolm[ollama]    # Ollama support
pip install lolm[litellm]   # LiteLLM support
```

## 📖 Quick Start

### CLI

```bash
# Show provider status
lolm status

# Set default provider
lolm set-provider openrouter

# Set model
lolm set-model openrouter nvidia/nemotron-3-nano-30b-a3b:free

# Manage API keys
lolm key set openrouter YOUR_API_KEY

# Test generation
lolm test
```

### Python API

```python
from lolm import get_client, LLMManager

# Simple usage
client = get_client()
response = client.generate("Explain this code")

# With specific provider
client = get_client(provider='openrouter')
response = client.generate("Hello!", system="You are helpful")

# With manager for fallback
manager = LLMManager()
manager.initialize()

response = manager.generate_with_fallback(
    "Generate code",
    providers=['openrouter', 'groq', 'ollama']
)
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# API Keys
OPENROUTER_API_KEY=sk-or-v1-...
GROQ_API_KEY=gsk_...
TOGETHER_API_KEY=...

# Default provider
LLM_PROVIDER=auto

# Model overrides
OPENROUTER_MODEL=nvidia/nemotron-3-nano-30b-a3b:free
OLLAMA_MODEL=qwen2.5-coder:14b
```

### litellm_config.yaml

```yaml
model_list:
  - model_name: code-analyzer
    litellm_params:
      model: ollama/qwen2.5-coder:7b
      api_base: http://localhost:11434
    priority: 10

router_settings:
  routing_strategy: simple-shuffle
  num_retries: 3
```

## 🖥️ CLI Reference

| Command | Description |
| ------- | ----------- |
| `lolm status` | Show provider status |
| `lolm set-provider PROVIDER` | Set default provider |
| `lolm set-model PROVIDER MODEL` | Set model for provider |
| `lolm key set PROVIDER KEY` | Set API key |
| `lolm key show` | Show configured keys |
| `lolm models [PROVIDER]` | List recommended models |
| `lolm test [--provider P]` | Test LLM generation |
| `lolm config show` | Show configuration |
| `lolm priority set-provider P N` | Set provider priority |
| `lolm priority set-mode MODE` | Set priority mode |

## 🔌 Supported Providers

| Provider | Type | Free Tier | Default Model |
| -------- | ---- | --------- | ------------- |
| OpenRouter | Cloud | ✓ | nvidia/nemotron-3-nano-30b-a3b:free |
| Ollama | Local | ✓ | qwen2.5-coder:14b |
| Groq | Cloud | ✓ | llama-3.1-70b-versatile |
| Together | Cloud | - | Qwen/Qwen2.5-Coder-32B-Instruct |
| LiteLLM | Universal | - | gpt-4 |

## 🧰 Monorepo (code2logic) workflow

If you use `lolm` inside the `code2logic` monorepo, you can manage all packages from the repository root:

```bash
make test-all
make build-subpackages
make publish-all
```

See: `docs/19-monorepo-workflow.md`.

## 🧪 Development

```bash
# Install dev dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Lint
make lint

# Build package
make build

# Publish to PyPI
make publish
```

## 📄 License

Apache 2.0 License - see [LICENSE](../LICENSE) for details.

## 🔗 Links

- [Documentation](https://code2logic.readthedocs.io/en/latest/lolm/)
- [PyPI](https://pypi.org/project/lolm/)
- [GitHub](https://github.com/wronai/code2logic/tree/main/lolm)
- [Issues](https://github.com/wronai/code2logic/issues)
