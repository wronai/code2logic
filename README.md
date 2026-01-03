# Code2Logic

[![PyPI version](https://badge.fury.io/py/code2logic.svg)](https://badge.fury.io/py/code2logic)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Convert source code to logical representation for LLM analysis.**

Code2Logic analyzes codebases and generates compact, LLM-friendly representations with semantic understanding. Perfect for feeding project context to AI assistants, building code documentation, or analyzing code structure.

## ✨ Features

- 🌳 **Multi-language support** - Python, JavaScript, TypeScript, Java, Go, Rust, and more
- 🎯 **Tree-sitter AST parsing** - 99% accuracy with graceful fallback
- 📊 **NetworkX dependency graphs** - PageRank, hub detection, cycle analysis
- 🔍 **Rapidfuzz similarity** - Find duplicate and similar functions
- 🧠 **NLP intent extraction** - Human-readable function descriptions
- 📦 **Zero dependencies** - Core works without any external libs

## 🚀 Installation

### Basic (no dependencies)
```bash
pip install code2logic
```

### Full (all features)
```bash
pip install code2logic[full]
```

### Selective features
```bash
pip install code2logic[treesitter]  # High-accuracy AST parsing
pip install code2logic[graph]       # Dependency analysis
pip install code2logic[similarity]  # Similar function detection
pip install code2logic[nlp]         # Enhanced intents
```

## 📖 Quick Start

### Command Line

```bash
# Standard Markdown output
code2logic /path/to/project

# Compact format (10-15x smaller)
code2logic /path/to/project -f compact

# JSON for RAG systems
code2logic /path/to/project -f json -o project.json

# With detailed analysis
code2logic /path/to/project -d detailed
```

### Python API

```python
from code2logic import analyze_project, MarkdownGenerator

# Analyze a project
project = analyze_project("/path/to/project")

# Generate output
generator = MarkdownGenerator()
output = generator.generate(project, detail_level='standard')
print(output)

# Access analysis results
print(f"Files: {project.total_files}")
print(f"Lines: {project.total_lines}")
print(f"Languages: {project.languages}")

# Get hub modules (most important)
hubs = [p for p, n in project.dependency_metrics.items() if n.is_hub]
print(f"Key modules: {hubs}")
```

## 📋 Output Formats

### Markdown (default)
Human-readable documentation with:
- Project structure tree with hub markers (★)
- Dependency graphs with PageRank scores
- Classes with methods and intents
- Functions with signatures and descriptions

### Compact
Ultra-compact format optimized for LLM context:
```
# myproject | 102f 31875L | typescript:79/python:23
ENTRY: index.ts main.py
HUBS: evolution-manager llm-orchestrator

[core/evolution]
  evolution-manager.ts (3719L) C:EvolutionManager | F:createEvolutionManager
  task-queue.ts (139L) C:TaskQueue,Task
```

### JSON
Machine-readable format for:
- RAG (Retrieval-Augmented Generation)
- Database storage
- Further analysis

## 🔧 Configuration

### Library Status
Check which features are available:
```bash
code2logic --status
```
```
Library Status:
  tree_sitter: ✓
  networkx: ✓
  rapidfuzz: ✓
  nltk: ✗
  spacy: ✗
```

### Python API
```python
from code2logic import get_library_status

status = get_library_status()
# {'tree_sitter': True, 'networkx': True, ...}
```

## 📊 Analysis Features

### Dependency Analysis
- **PageRank** - Identifies most important modules
- **Hub detection** - Central modules marked with ★
- **Cycle detection** - Find circular dependencies
- **Clustering** - Group related modules

### Intent Generation
Functions get human-readable descriptions:
```yaml
methods:
  async findById(id:string) -> Promise<User>  # retrieves user by id
  async createUser(data:UserDTO) -> Promise<User>  # creates user
  validateEmail(email:string) -> boolean  # validates email
```

### Similarity Detection
Find duplicate and similar functions:
```yaml
Similar Functions:
  core/auth.ts::validateToken:
    - python/auth.py::validate_token (92%)
    - services/jwt.ts::verifyToken (85%)
```

## 🏗️ Architecture

```
code2logic/
├── analyzer.py      # Main orchestrator
├── parsers.py       # Tree-sitter + fallback parser
├── dependency.py    # NetworkX dependency analysis
├── similarity.py    # Rapidfuzz similar detection
├── intent.py        # NLP intent generation
├── generators.py    # Output generators (MD/Compact/JSON)
├── models.py        # Data structures
└── cli.py           # Command-line interface
```

## 🔌 Integration Examples

### With Claude/ChatGPT
```python
from code2logic import analyze_project, CompactGenerator

project = analyze_project("./my-project")
context = CompactGenerator().generate(project)

# Use in your LLM prompt
prompt = f"""
Analyze this codebase and suggest improvements:

{context}
"""
```

### With RAG Systems
```python
import json
from code2logic import analyze_project, JSONGenerator

project = analyze_project("./my-project")
data = json.loads(JSONGenerator().generate(project))

# Index in vector DB
for module in data['modules']:
    for func in module['functions']:
        embed_and_store(
            text=f"{func['name']}: {func['intent']}",
            metadata={'path': module['path'], 'type': 'function'}
        )
```

## 🧪 Development

### Setup
```bash
git clone https://github.com/softreck/code2logic
cd code2logic
pip install -e ".[dev]"
pre-commit install
```

### Tests
```bash
pytest
pytest --cov=code2logic --cov-report=html
```

### Type Checking
```bash
mypy code2logic
```

### Linting
```bash
ruff check code2logic
black code2logic
```

## 📈 Performance

| Codebase Size | Files | Lines | Time | Output Size |
|--------------|-------|-------|------|-------------|
| Small        | 10    | 1K    | <1s  | ~5KB        |
| Medium       | 100   | 30K   | ~2s  | ~50KB       |
| Large        | 500   | 150K  | ~10s | ~200KB      |

Compact format is ~10-15x smaller than Markdown.

## 🤝 Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- [Documentation](https://code2logic.readthedocs.io)
- [PyPI](https://pypi.org/project/code2logic/)
- [GitHub](https://github.com/softreck/code2logic)
- [Issues](https://github.com/softreck/code2logic/issues)
