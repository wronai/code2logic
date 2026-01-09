# Logic2Test - Test Generation from Logic Files

[![PyPI version](https://badge.fury.io/py/logic2test.svg)](https://badge.fury.io/py/logic2test)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Generate test scaffolds from Code2Logic output files.**

Reads YAML, Hybrid YAML, or TOON format files and produces pytest-compatible test suites.

## ✨ Features

- 🧪 **Unit tests** - Test scaffolds for classes and functions
- 🔗 **Integration tests** - Cross-module interaction tests
- 📊 **Property tests** - Hypothesis-based property testing
- 🎯 **Smart mocking** - Auto-generated mocks based on type hints
- 📝 **Pytest style** - Ready-to-run pytest test files

## 🚀 Installation

```bash
pip install logic2test
```

Or with optional dependencies:

```bash
pip install logic2test[hypothesis]  # Property testing support
```

## 📖 Quick Start

### CLI

```bash
# Show what can be generated
logic2test out/code2logic/project.c2l.yaml --summary

# Generate unit tests
logic2test out/code2logic/project.c2l.yaml -o out/logic2test/tests/

# Generate all test types
logic2test out/code2logic/project.c2l.yaml -o out/logic2test/tests/ --type all

# Include private methods
logic2test out/code2logic/project.c2l.yaml -o out/logic2test/tests/ --include-private
```

### Python API

```python
from logic2test import TestGenerator

# Create generator
generator = TestGenerator('out/code2logic/project.c2l.yaml')

# Get summary
summary = generator.summary()
print(f"Classes: {summary['testable_classes']}")
print(f"Functions: {summary['testable_functions']}")

# Generate unit tests
result = generator.generate_unit_tests('out/logic2test/tests/')
print(f"Generated {result.tests_generated} tests")

# Generate all test types
result = generator.generate_all('out/logic2test/tests/')
```

## 📋 Generated Test Structure

```text
tests/
├── unit/
│   ├── test_analyzer.py
│   ├── test_parsers.py
│   └── test_generators.py
├── integration/
│   └── test_integration.py
└── property/
    └── test_properties.py
```

### Example Generated Test

```python
def test_analyze_project():
    """Test analyze_project function."""
    # Arrange
    path = "/tmp/test_path"
    use_treesitter = Mock()
    verbose = False

    # Act
    # result = analyze_project(path, use_treesitter, verbose)

    # Assert
    assert result is not None  # TODO: Add specific assertion
```

## 🖥️ CLI Reference

| Option | Description |
| ------ | ----------- |
| `-o, --output DIR` | Output directory for tests |
| `-t, --type TYPE` | Test type: unit, integration, property, all |
| `--framework FW` | Test framework: pytest, unittest |
| `--include-private` | Include private methods |
| `--include-dunder` | Include dunder methods |
| `--summary` | Show summary without generating |
| `-v, --verbose` | Verbose output |

## ⚙️ Configuration

```python
from logic2test import TestGenerator, GeneratorConfig

config = GeneratorConfig(
    framework='pytest',
    include_private=False,
    include_dunder=False,
    max_tests_per_file=50,
    output_prefix='test_',
    generate_class_tests=True,
    generate_function_tests=True,
    generate_dataclass_tests=True,
)

generator = TestGenerator('project.c2l.yaml', config)
```

## 📥 Supported Input Formats

| Format | Extension | Description |
| ------ | --------- | ----------- |
| YAML | `.yaml` | Standard Code2Logic output |
| Hybrid | `.hybrid.yaml` | Compact YAML with metadata |
| TOON | `.toon` | Token-Oriented Object Notation |

## 🧰 Monorepo (code2logic) workflow

If you use `logic2test` inside the `code2logic` monorepo, you can manage all packages from the repository root:

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

- [Documentation](https://code2logic.readthedocs.io/en/latest/logic2test/)
- [PyPI](https://pypi.org/project/logic2test/)
- [GitHub](https://github.com/wronai/code2logic/tree/main/logic2test)
- [Issues](https://github.com/wronai/code2logic/issues)
