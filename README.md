# code2logic

Convert codebase structure to logical representations with AI-powered insights.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/wronai/code2logic)
[![Coverage](https://img.shields.io/badge/coverage-80%25-green.svg)](https://codecov.io)

code2logic is a powerful Python package that analyzes code projects and generates various logical representations, including dependency graphs, architectural insights, and AI-powered refactoring suggestions.

## ✨ Features

### 🔍 **Code Analysis**
- **Multi-language support**: Python, JavaScript, Java, C/C++
- **AST parsing** with Tree-sitter (primary) and fallback parsers
- **Dependency graph analysis** using NetworkX
- **Code similarity detection** across modules, functions, and classes
- **Complexity metrics** and code quality assessment

### 🤖 **AI-Powered Insights**
- **LLM integration** with Ollama and LiteLLM
- **Intent analysis** from natural language queries
- **Automated refactoring suggestions**
- **Code generation** and improvement recommendations
- **Documentation generation** with AI assistance

### 📊 **Multiple Output Formats**
- **JSON**: Machine-readable structured data
- **YAML**: Human-readable configuration format
- **CSV**: Tabular data for spreadsheet analysis
- **Markdown**: Documentation-friendly reports
- **Compact**: Minimal text representation

### 🔗 **Integration Support**
- **MCP Server** for Claude Desktop integration
- **CLI** with auto-dependency installation
- **Python API** for programmatic use
- **Docker** support for containerized deployment

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install code2logic

# Install with optional dependencies
pip install code2logic[all]  # All features
pip install code2logic[mcp]   # MCP server support
pip install code2logic[dev]   # Development tools
```

### Basic Usage

```bash
# Analyze a project
code2logic /path/to/your/project

# Generate specific format
code2logic /path/to/project --format json --output analysis.json

# Generate all formats
code2logic /path/to/project --format all --output project_analysis
```

### Python API

```python
from code2logic import ProjectAnalyzer, JSONGenerator

# Analyze a project
analyzer = ProjectAnalyzer("/path/to/project")
project = analyzer.analyze()

# Generate JSON output
generator = JSONGenerator()
generator.generate(project, "analysis.json")

# Access project data
print(f"Modules: {len(project.modules)}")
print(f"Functions: {sum(len(m.functions) for m in project.modules)}")
print(f"Dependencies: {len(project.dependencies)}")
```

### LLM Integration

```python
from code2logic import ProjectAnalyzer, LLMInterface, LLMConfig

# Setup LLM
config = LLMConfig(provider="ollama", model="codellama")
llm = LLMInterface(config)

# Analyze project
analyzer = ProjectAnalyzer("/path/to/project")
project = analyzer.analyze()

# Get refactoring suggestions
for module in project.modules:
    suggestions = llm.suggest_refactoring(module, project)
    print(f"Suggestions for {module.name}: {suggestions}")
```

## 📖 Examples

### Basic Project Analysis

```bash
# Analyze current directory
code2logic . --format json --output analysis.json

# Generate comprehensive report
code2logic . --format all --output comprehensive_analysis
```

### AI-Powered Refactoring

```python
from code2logic import ProjectAnalyzer, IntentAnalyzer

# Analyze project
analyzer = ProjectAnalyzer("/path/to/project")
project = analyzer.analyze()

# Analyze user intent
intent_analyzer = IntentAnalyzer()
intents = intent_analyzer.analyze_intent(
    "I want to refactor the main module to improve performance",
    project
)

for intent in intents:
    print(f"Intent: {intent.type.value} (confidence: {intent.confidence})")
    print(f"Suggestions: {intent.suggestions}")
```

### Dependency Analysis

```python
from code2logic.dependency import DependencyAnalyzer

# Analyze dependencies
dep_analyzer = DependencyAnalyzer()
dependencies = dep_analyzer.analyze_dependencies(project.modules)

# Get circular dependencies
circular_deps = dep_analyzer.get_circular_dependencies()
print(f"Circular dependencies: {circular_deps}")

# Get dependency layers
layers = dep_analyzer.get_dependency_layers()
print(f"Dependency layers: {layers}")
```

### MCP Server for Claude Desktop

```bash
# Start MCP server
code2logic --mcp --mcp-port 8080

# Use with Claude Desktop
# Configure Claude Desktop to connect to localhost:8080
```

## 🔧 Configuration

### LLM Configuration

```python
from code2logic import LLMConfig

# Ollama configuration
config = LLMConfig(
    provider="ollama",
    model="codellama",
    temperature=0.7,
    max_tokens=2000
)

# LiteLLM configuration
config = LLMConfig(
    provider="litellm",
    model="gpt-4",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"
)
```

### Analysis Configuration

```python
from code2logic import ProjectAnalyzer

# Custom configuration
config = {
    "include_tests": False,
    "max_depth": 10,
    "ignore_patterns": ["__pycache__", "node_modules"],
    "complexity_threshold": 10
}

analyzer = ProjectAnalyzer("/path/to/project", config)
```

## 📊 Output Formats

### JSON Structure

```json
{
  "project": {
    "name": "my_project",
    "path": "/path/to/project",
    "statistics": {
      "total_modules": 10,
      "total_functions": 50,
      "total_classes": 20,
      "total_dependencies": 35,
      "total_lines_of_code": 5000
    },
    "modules": [...],
    "dependencies": [...],
    "similarities": [...]
  }
}
```

### Markdown Report

```markdown
# My Project

## Statistics

| Metric | Value |
|--------|-------|
| Modules | 10 |
| Functions | 50 |
| Classes | 20 |
| Dependencies | 35 |
| Lines of Code | 5000 |

## Modules

### main.py

**Path:** `/project/main.py`
**Lines of Code:** 150

**Functions:**
- `calculate_sum()` (15 LOC, complexity: 2) 📝
- `fibonacci()` (8 LOC, complexity: 5) 📝

**Classes:**
- `Calculator` (4 methods)
```

## 🐳 Docker Support

### Using Pre-built Images

```bash
# Pull the image
docker pull wronai/code2logic:latest

# Run analysis
docker run -v /path/to/project:/workspace wronai/code2logic:latest code2logic /workspace
```

### Building from Source

```bash
# Build the image
docker build -t code2logic .

# Run with Ollama
docker-compose up
```

### Docker Compose

```yaml
version: '3.8'
services:
  code2logic:
    build: .
    volumes:
      - ./project:/workspace
    command: code2logic /workspace --format json
    
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
      
  litellm:
    image: ghcr.io/berriai/litellm:main
    ports:
      - "4000:4000"
    volumes:
      - ./litellm_config.yaml:/app/config.yaml
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=code2logic --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

## 📝 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/wronai/code2logic.git
cd code2logic

# Install development dependencies
pip install -e .[dev]

# Setup pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Format code
black code2logic tests examples
isort code2logic tests examples

# Lint code
flake8 code2logic tests examples
mypy code2logic

# Run security checks
bandit -r code2logic
```

### Project Structure

```
code2logic/
├── code2logic/                    # Main package
│   ├── __init__.py               # API exports
│   ├── analyzer.py               # Core analyzer
│   ├── cli.py                    # CLI interface
│   ├── dependency.py             # Dependency analysis
│   ├── generators.py             # Output generators
│   ├── intent.py                 # Intent analysis
│   ├── llm.py                    # LLM integration
│   ├── mcp_server.py             # MCP server
│   ├── models.py                 # Data models
│   ├── parsers.py                # Code parsers
│   ├── similarity.py             # Similarity detection
│   └── py.typed                  # Type hints marker
├── tests/                        # Test suite
├── examples/                     # Usage examples
├── docs/                         # Documentation
├── dist/                         # Built packages
└── docker/                       # Docker files
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Contribution Areas

- **Core analysis features**: Improve parsing, dependency analysis
- **LLM integration**: Add new providers, improve prompts
- **Output formats**: Support new formats, improve existing ones
- **Documentation**: Improve docs, add examples
- **Testing**: Increase test coverage, add integration tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Tree-sitter](https://tree-sitter.github.io/) for robust AST parsing
- [NetworkX](https://networkx.org/) for graph analysis
- [LiteLLM](https://github.com/BerriAI/litellm) for LLM integration
- [Ollama](https://ollama.ai/) for local LLM support
- [MCP](https://modelcontextprotocol.io/) for Claude Desktop integration

## 📞 Support

- **Documentation**: [https://code2logic.readthedocs.io](https://code2logic.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/wronai/code2logic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/wronai/code2logic/discussions)
- **Email**: team@code2logic.dev

## 🗺️ Roadmap

- [ ] Support for more programming languages (Rust, Go, TypeScript)
- [ ] Web-based visualization interface
- [ ] Advanced refactoring patterns
- [ ] Integration with popular IDEs
- [ ] Performance optimizations for large codebases
- [ ] Cloud-based analysis service
- [ ] Team collaboration features

---

**code2logic** - Transform code into insights 🚀