# Logic2Code - Code Generation from Logic Files

[![PyPI version](https://badge.fury.io/py/logic2code.svg)](https://badge.fury.io/py/logic2code)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Generate source code from Code2Logic output files.**

Reads YAML, Hybrid YAML, or TOON format files and produces working code scaffolds.

## ✨ Features

- 🏗️ **Code scaffolds** - Complete class and function structures
- 📝 **Type hints** - Full type annotation support
- 📚 **Docstrings** - Auto-generated documentation
- 🤖 **LLM integration** - Optional AI-powered implementations
- 🔄 **Refactoring support** - Logic → Code → Logic workflow

## 🚀 Installation

```bash
pip install logic2code
```

Or with LLM support:

```bash
pip install logic2code[llm]  # With lolm for LLM generation
```

## 📖 Quick Start

### CLI

```bash
# Show what can be generated
logic2code out/code2logic/project.c2l.yaml --summary

# Generate Python code
logic2code out/code2logic/project.c2l.yaml -o out/logic2code/generated_code/

# Generate stubs only
logic2code out/code2logic/project.c2l.yaml -o out/logic2code/generated_code/ --stubs-only

# Generate specific modules
logic2code out/code2logic/project.c2l.yaml -o out/logic2code/generated_code/ --modules "analyzer.py,parsers.py"
```

### Python API

```python
from logic2code import CodeGenerator

# Create generator
generator = CodeGenerator('out/code2logic/project.c2l.yaml')

# Get summary
summary = generator.summary()
print(f"Modules: {summary['total_modules']}")
print(f"Classes: {summary['total_classes']}")

# Generate code
result = generator.generate('out/logic2code/generated_code/')
print(f"Generated {result.files_generated} files")

# Generate single module
code = generator.generate_module('analyzer.py')
print(code)
```

## 📋 Generated Code Example

```python
@dataclass
class ProjectInfo:
    """Information about analyzed project."""
    name: str
    total_files: int = 0
    total_lines: int = 0
    modules: List[ModuleInfo] = field(default_factory=list)


async def analyze_project(
    path: str,
    use_treesitter: bool = True,
    verbose: bool = False
) -> ProjectInfo:
    """Analyzes project and returns ProjectInfo."""
    raise NotImplementedError("TODO: Implement analyze_project")
```

## 🤖 LLM-Enhanced Generation

```python
from logic2code import CodeGenerator, GeneratorConfig

config = GeneratorConfig(
    use_llm=True,
    llm_provider='openrouter'
)

generator = CodeGenerator('out/code2logic/project.c2l.yaml', config)
result = generator.generate('out/logic2code/generated_code/')
```

## 🖥️ CLI Reference

| Option | Description |
| ------ | ----------- |
| `-o, --output DIR` | Output directory |
| `-l, --language LANG` | Target language (default: python) |
| `--stubs-only` | Generate stubs only |
| `--no-docstrings` | Skip docstring generation |
| `--no-type-hints` | Skip type hints |
| `--no-init` | Skip `__init__.py` generation |
| `--flat` | Flat output structure |
| `--modules LIST` | Comma-separated modules to generate |
| `--summary` | Show summary without generating |
| `-v, --verbose` | Verbose output |

## ⚙️ Configuration

```python
from logic2code import CodeGenerator, GeneratorConfig

config = GeneratorConfig(
    language='python',
    stubs_only=False,
    include_docstrings=True,
    include_type_hints=True,
    generate_init=True,
    preserve_structure=True,
    use_llm=False,
    llm_provider=None,
)

generator = CodeGenerator('project.c2l.yaml', config)
```

## 🔄 Refactoring Workflow

```bash
# 1. Analyze original code
code2logic src/ -f yaml -o out/code2logic/project.c2l.yaml

# 2. Modify logic file (add/change functions)
# Edit out/code2logic/project.c2l.yaml

# 3. Generate new code
logic2code out/code2logic/project.c2l.yaml -o out/logic2code/new_src/

# 4. Compare and merge
diff -r src/ out/logic2code/new_src/
```

## 🧰 Monorepo (code2logic) workflow

If you use `logic2code` inside the `code2logic` monorepo, you can manage all packages from the repository root:

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

- [Documentation](https://code2logic.readthedocs.io/en/latest/logic2code/)
- [PyPI](https://pypi.org/project/logic2code/)
- [GitHub](https://github.com/wronai/code2logic/tree/main/logic2code)
- [Issues](https://github.com/wronai/code2logic/issues)
