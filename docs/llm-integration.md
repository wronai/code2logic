# LLM Integration Guide

> Using Code2Logic with Large Language Models

[← Output Formats](output-formats.md) | [Examples →](examples.md)

## Overview

Code2Logic generates LLM-optimized representations of codebases. This guide covers integration with various LLM providers.

## Supported Providers

| Provider | Type | Setup |
|----------|------|-------|
| [OpenRouter](#openrouter) | Cloud | API key |
| [Ollama](#ollama) | Local | Install |
| [OpenAI](#openai) | Cloud | API key |
| [Anthropic](#anthropic) | Cloud | API key |
| [LiteLLM](#litellm) | Universal | pip install |

## OpenRouter

Best for accessing multiple models through one API.

### Setup

```bash
# Get API key from https://openrouter.ai/keys
export OPENROUTER_API_KEY="sk-or-v1-your-key"
export OPENROUTER_MODEL="qwen/qwen-2.5-coder-32b-instruct"
```

### Recommended Models

| Model | Best For | Cost |
|-------|----------|------|
| `qwen/qwen-2.5-coder-32b-instruct` | Code generation | $$ |
| `deepseek/deepseek-coder-33b-instruct` | Code analysis | $$ |
| `anthropic/claude-3-haiku` | Fast responses | $ |
| `meta-llama/llama-3.1-70b-instruct` | General | $$ |

### Example

```python
from code2logic import analyze_project
from code2logic.gherkin import GherkinGenerator
import httpx

# Analyze code
project = analyze_project("./my_project")
gherkin = GherkinGenerator().generate(project, detail='standard')

# Send to OpenRouter
response = httpx.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json"
    },
    json={
        "model": "qwen/qwen-2.5-coder-32b-instruct",
        "messages": [
            {"role": "system", "content": "You are a code expert."},
            {"role": "user", "content": f"Analyze this code:\n{gherkin}"}
        ]
    }
)
print(response.json()['choices'][0]['message']['content'])
```

## Ollama

Best for local, private, and free LLM usage.

### Setup

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull code-optimized model
ollama pull qwen2.5-coder:14b

# Start server
ollama serve
```

### Configuration

```bash
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="qwen2.5-coder:14b"
```

### Example

```python
from code2logic import analyze_project
from code2logic.llm import OllamaClient

# Analyze code
project = analyze_project("./my_project")

# Use Ollama
client = OllamaClient(model="qwen2.5-coder:14b")
response = client.generate(
    prompt=f"Analyze this project:\n{project.total_files} files",
    system="You are a code reviewer."
)
print(response)
```

### CLI Integration

```bash
# Generate analysis and pipe to Ollama
code2logic ./my_project -f gherkin -d minimal | \
  ollama run qwen2.5-coder:7b "Generate unit tests for this code"
```

## OpenAI

### Setup

```bash
export OPENAI_API_KEY="sk-your-key"
export OPENAI_MODEL="gpt-4-turbo"
```

### Example

```python
from openai import OpenAI
from code2logic import analyze_project
from code2logic.generators import CSVGenerator

client = OpenAI()
project = analyze_project("./my_project")
csv_output = CSVGenerator().generate(project, detail='standard')

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are a code analyst."},
        {"role": "user", "content": f"Review this code:\n{csv_output}"}
    ]
)
print(response.choices[0].message.content)
```

## Anthropic

### Setup

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key"
export ANTHROPIC_MODEL="claude-3-sonnet-20240229"
```

### Example

```python
from anthropic import Anthropic
from code2logic import analyze_project
from code2logic.gherkin import GherkinGenerator

client = Anthropic()
project = analyze_project("./my_project")
gherkin = GherkinGenerator().generate(project)

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=4000,
    messages=[
        {"role": "user", "content": f"Analyze this code:\n{gherkin}"}
    ]
)
print(response.content[0].text)
```

## LiteLLM

Universal interface for all LLM providers.

### Setup

```bash
pip install litellm
```

### Example

```python
import litellm
from code2logic import analyze_project
from code2logic.gherkin import GherkinGenerator

project = analyze_project("./my_project")
gherkin = GherkinGenerator().generate(project, detail='minimal')

# Works with any provider
response = litellm.completion(
    model="ollama/qwen2.5-coder:7b",  # or "gpt-4", "claude-3-sonnet", etc.
    messages=[
        {"role": "system", "content": "You are a code expert."},
        {"role": "user", "content": f"Review:\n{gherkin}"}
    ]
)
print(response.choices[0].message.content)
```

## Code Reproduction Workflow

Complete workflow: Code → Gherkin → LLM → Code

```bash
# Run the OpenRouter reproduction example
python examples/openrouter_code_reproduction.py --source code2logic/models.py

# Or with Ollama (free, local)
python examples/mcp_litellm_refactor.py ./my_project
```

### Workflow Steps

1. **Analyze** - Convert code to Gherkin specification
2. **Generate** - LLM generates code from Gherkin
3. **Compare** - Measure reproduction quality

```python
# Step 1: Analyze
from code2logic import analyze_project
from code2logic.gherkin import GherkinGenerator

project = analyze_project("./my_project")
gherkin = GherkinGenerator().generate(project, detail='full')

# Step 2: Generate with LLM
prompt = f"""Generate Python code for this specification:

{gherkin}

Output complete, working code."""

response = llm_client.generate(prompt)

# Step 3: Compare
# See examples/openrouter_code_reproduction.py for full comparison
```

## Best Practices

### Token Efficiency

| Format | Tokens | Accuracy | When to Use |
|--------|--------|----------|-------------|
| Gherkin minimal | ~3K | 95% | Code generation |
| CSV minimal | ~8K | 70% | Analysis |
| Compact | ~200 | 50% | Quick overview |

### Prompt Engineering

```python
# Good: Specific task with context
prompt = f"""Given this code structure:
{gherkin}

Task: Generate unit tests for the UserService class.
Requirements:
- Use pytest
- Mock external dependencies
- Cover edge cases"""

# Bad: Vague request
prompt = f"Write tests for {gherkin}"
```

### Cost Optimization

1. Use **Gherkin minimal** for code generation
2. Use **Compact** for quick queries
3. Use **local Ollama** for development
4. Use **cloud LLMs** for production

## MCP Server Integration

For Claude Desktop / Windsurf:

```json
// ~/.config/claude/claude_desktop_config.json
{
  "mcpServers": {
    "code2logic": {
      "command": "python",
      "args": ["-m", "code2logic.mcp_server"]
    }
  }
}
```

See [Examples](examples.md) for more integration patterns.

---

[← Output Formats](output-formats.md) | [Examples →](examples.md)
