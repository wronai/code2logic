# Examples

> Usage examples and workflows for Code2Logic

[← LLM Integration](llm-integration.md) | [Architecture →](architecture.md)

## Available Examples

All examples are in the `examples/` folder:

| Example | Description |
|---------|-------------|
| [quick_start.py](../examples/quick_start.py) | Basic usage guide |
| [bdd_workflow.py](../examples/bdd_workflow.py) | Complete BDD workflow |
| [token_efficiency.py](../examples/token_efficiency.py) | Format comparison |
| [duplicate_detection.py](../examples/duplicate_detection.py) | Find duplicates |
| [refactor_suggestions.py](../examples/refactor_suggestions.py) | LLM refactoring |
| [generate_code.py](../examples/generate_code.py) | Code generation |
| [code_review.py](../examples/code_review.py) | Automated review |
| [api_documentation.py](../examples/api_documentation.py) | Generate docs |
| [openrouter_code_reproduction.py](../examples/openrouter_code_reproduction.py) | Code reproduction |
| [mcp_litellm_refactor.py](../examples/mcp_litellm_refactor.py) | MCP/LiteLLM |
| [windsurf_mcp_integration.py](../examples/windsurf_mcp_integration.py) | Windsurf setup |

## Quick Start

```python
from code2logic import analyze_project

# Analyze a project
project = analyze_project("/path/to/project")

print(f"Files: {project.total_files}")
print(f"Lines: {project.total_lines}")
print(f"Languages: {list(project.languages.keys())}")
```

## BDD Workflow

Generate Gherkin specs and step definitions:

```python
from code2logic import analyze_project
from code2logic.gherkin import GherkinGenerator, StepDefinitionGenerator

# Analyze
project = analyze_project("./my_project")

# Generate Gherkin
gherkin_gen = GherkinGenerator()
gherkin = gherkin_gen.generate(project, detail='standard')

# Generate step definitions
step_gen = StepDefinitionGenerator()
steps = step_gen.generate(project)

# Save files
with open("features/project.feature", "w") as f:
    f.write(gherkin)

with open("tests/steps/test_steps.py", "w") as f:
    f.write(steps)
```

## Code Reproduction with OpenRouter

```bash
# Configure API key
export OPENROUTER_API_KEY="sk-or-v1-your-key"

# Run reproduction test
python examples/openrouter_code_reproduction.py \
  --source code2logic/models.py \
  --model qwen/qwen-2.5-coder-32b-instruct \
  --output results/
```

Output:
```
Step 1: Reading source code...
  Read 5,234 chars from models.py

Step 2: Generating Gherkin specification...
  Generated 2,104 chars of Gherkin

Step 3: Generating code with LLM...
  Generated 4,892 chars of code

Step 4: Comparing original vs generated...
  Similarity: 72.5%
  Structural score: 85.0%

SUMMARY
  Average Score: 78.8%
  ✓ Good reproduction quality!
```

## Duplicate Detection

```python
from code2logic import analyze_project

project = analyze_project("./my_project")

# Find duplicates by hash
from collections import defaultdict
import hashlib

duplicates = defaultdict(list)
for module in project.modules:
    for func in module.functions:
        sig = f"{func.name}({','.join(func.params)})"
        h = hashlib.md5(sig.encode()).hexdigest()[:8]
        duplicates[h].append(f"{module.path}::{func.name}")

# Report
for h, funcs in duplicates.items():
    if len(funcs) > 1:
        print(f"Duplicate group {h}:")
        for f in funcs:
            print(f"  - {f}")
```

## LLM Refactoring Suggestions

```python
from code2logic import analyze_project
from code2logic.generators import CSVGenerator
import httpx

# Analyze
project = analyze_project("./my_project")
csv = CSVGenerator().generate(project, detail='standard')

# Get LLM suggestions
response = httpx.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5-coder:7b",
        "prompt": f"""Analyze this code and suggest refactoring:

{csv[:4000]}

List top 5 refactoring opportunities with effort estimates.""",
        "stream": False
    }
)
print(response.json()['response'])
```

## Token Efficiency Analysis

```python
from code2logic import analyze_project
from code2logic.generators import (
    MarkdownGenerator, CompactGenerator, JSONGenerator,
    YAMLGenerator, CSVGenerator
)
from code2logic.gherkin import GherkinGenerator

project = analyze_project("./my_project")

# Compare formats
formats = {
    'markdown': MarkdownGenerator().generate(project),
    'compact': CompactGenerator().generate(project),
    'json': JSONGenerator().generate(project),
    'yaml': YAMLGenerator().generate(project),
    'csv': CSVGenerator().generate(project),
    'gherkin': GherkinGenerator().generate(project),
}

for name, output in formats.items():
    tokens = len(output) // 4  # Approximate
    print(f"{name:12} {tokens:>6,} tokens")
```

## API Documentation Generation

```bash
python examples/api_documentation.py ./my_project --format all --output docs/api/
```

Generates:
- `API.md` - Markdown documentation
- `openapi.json` - OpenAPI/Swagger spec
- `types.d.ts` - TypeScript definitions
- `project.pyi` - Python stubs

## Code Review

```bash
# Run automated code review
python examples/code_review.py ./my_project --focus all --output review.md

# Security-focused review
python examples/code_review.py ./my_project --focus security

# Without LLM (fast, local analysis only)
python examples/code_review.py ./my_project --no-llm
```

## MCP Server Integration

Start the MCP server for Claude Desktop:

```bash
# Start server
python -m code2logic.mcp_server

# Or via make
make mcp-server
```

Configure Claude Desktop:
```json
{
  "mcpServers": {
    "code2logic": {
      "command": "python",
      "args": ["-m", "code2logic.mcp_server"]
    }
  }
}
```

## Running Examples

```bash
# Quick start
python examples/quick_start.py

# BDD workflow (uses current directory)
python examples/bdd_workflow.py ./my_project

# Token efficiency comparison
python examples/token_efficiency.py ./my_project

# Duplicate detection
python examples/duplicate_detection.py ./my_project

# Code review (with LLM)
python examples/code_review.py ./my_project

# Code review (without LLM)
python examples/code_review.py ./my_project --no-llm

# OpenRouter reproduction (requires API key)
python examples/openrouter_code_reproduction.py --source code2logic/models.py
```

## See Also

- [CLI Reference](cli-reference.md) - Command-line options
- [Python API](python-api.md) - Programmatic interface
- [LLM Integration](llm-integration.md) - LLM provider setup

---

[← LLM Integration](llm-integration.md) | [Architecture →](architecture.md)
