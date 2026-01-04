# Code2Logic - API Documentation

## Overview

Code2Logic is a Python library for analyzing source code and generating compact, LLM-friendly representations with semantic understanding.

## Table of Contents

- [Installation](#installation)
- [CLI Usage](#cli-usage)
- [Python API](#python-api)
- [Output Formats](#output-formats)
- [Models](#models)
- [Examples](#examples)

---

## Installation

```bash
# Basic
pip install code2logic

# Full features
pip install code2logic[full]

# Development
pip install -e ".[dev]"
```

---

## CLI Usage

```bash
# Run as module
python -m code2logic /path/to/project

# Available formats
python -m code2logic /path/to/project -f markdown   # Default
python -m code2logic /path/to/project -f compact    # Ultra-compact
python -m code2logic /path/to/project -f json       # Nested JSON
python -m code2logic /path/to/project -f yaml       # YAML
python -m code2logic /path/to/project -f csv        # CSV table
python -m code2logic /path/to/project -f gherkin    # BDD Gherkin

# Detail levels
python -m code2logic /path/to/project -d minimal    # 4 columns
python -m code2logic /path/to/project -d standard   # 8 columns (default)
python -m code2logic /path/to/project -d full       # 16 columns

# Options
python -m code2logic /path/to/project -o output.md  # Save to file
python -m code2logic /path/to/project --flat        # Flat structure (json/yaml)
python -m code2logic /path/to/project --no-install  # Skip auto-install
python -m code2logic /path/to/project --no-treesitter  # Use fallback parser
python -m code2logic /path/to/project -v           # Verbose output
python -m code2logic --status                       # Show library status
python -m code2logic --version                      # Show version
```

---

## Python API

### Main Functions

#### `analyze_project(path, use_treesitter=True, verbose=False, include_private=False)`

Analyze a project directory and return a `ProjectInfo` object.

```python
from code2logic import analyze_project

project = analyze_project("/path/to/project")
print(f"Files: {project.total_files}")
print(f"Lines: {project.total_lines}")
print(f"Languages: {list(project.languages.keys())}")
```

**Parameters:**
- `path` (str): Path to the project directory
- `use_treesitter` (bool): Use Tree-sitter for parsing (default: True)
- `verbose` (bool): Enable verbose logging (default: False)
- `include_private` (bool): Include private functions (default: False)

**Returns:** `ProjectInfo` object

---

### Generators

#### `MarkdownGenerator`

Generate Markdown documentation.

```python
from code2logic import analyze_project, MarkdownGenerator

project = analyze_project(".")
generator = MarkdownGenerator()
output = generator.generate(project, detail='standard')
```

**Methods:**
- `generate(project, detail='standard')` → str

---

#### `CompactGenerator`

Generate ultra-compact output (~10-15x smaller).

```python
from code2logic import analyze_project, CompactGenerator

project = analyze_project(".")
generator = CompactGenerator()
output = generator.generate(project)
```

**Methods:**
- `generate(project)` → str

---

#### `JSONGenerator`

Generate JSON output for RAG/embeddings.

```python
from code2logic import analyze_project, JSONGenerator

project = analyze_project(".")
generator = JSONGenerator()
output = generator.generate(project, flat=False, detail='standard')
```

**Methods:**
- `generate(project, flat=False, detail='standard')` → str

---

#### `YAMLGenerator`

Generate YAML output.

```python
from code2logic import analyze_project, YAMLGenerator

project = analyze_project(".")
generator = YAMLGenerator()
output = generator.generate(project, flat=False, detail='standard')
```

**Methods:**
- `generate(project, flat=False, detail='standard')` → str

---

#### `CSVGenerator`

Generate CSV table output.

```python
from code2logic import analyze_project, CSVGenerator

project = analyze_project(".")
generator = CSVGenerator()
output = generator.generate(project, detail='standard')
```

**Methods:**
- `generate(project, detail='standard')` → str

---

#### `GherkinGenerator`

Generate BDD Gherkin feature files (50x compression vs CSV).

```python
from code2logic import analyze_project, GherkinGenerator

project = analyze_project(".")
generator = GherkinGenerator(language='en')
output = generator.generate(project, detail='standard')

# Generate test scenarios
features = generator.generate_test_scenarios(project)
for feature in features:
    print(f"Feature: {feature.name} ({len(feature.scenarios)} scenarios)")
```

**Methods:**
- `generate(project, detail='standard')` → str
- `generate_test_scenarios(project)` → List[Feature]

---

#### `StepDefinitionGenerator`

Generate step definitions for BDD frameworks.

```python
from code2logic import GherkinGenerator, StepDefinitionGenerator

gherkin_gen = GherkinGenerator()
features = gherkin_gen.generate_test_scenarios(project)

step_gen = StepDefinitionGenerator()

# pytest-bdd
pytest_steps = step_gen.generate_pytest_bdd(features)

# behave
behave_steps = step_gen.generate_behave(features)

# cucumber-js
cucumber_steps = step_gen.generate_cucumber_js(features)
```

**Methods:**
- `generate_pytest_bdd(features)` → str
- `generate_behave(features)` → str
- `generate_cucumber_js(features)` → str

---

#### `CucumberYAMLGenerator`

Generate Cucumber YAML for CI/CD integration.

```python
from code2logic import analyze_project, CucumberYAMLGenerator

project = analyze_project(".")
generator = CucumberYAMLGenerator()
output = generator.generate(project, detail='standard')
```

**Methods:**
- `generate(project, detail='standard')` → str

---

### Utility Functions

#### `csv_to_gherkin(csv_content)`

Convert CSV analysis to Gherkin format.

```python
from code2logic import csv_to_gherkin

gherkin = csv_to_gherkin(csv_content)
```

---

#### `gherkin_to_test_data(gherkin_content)`

Extract test data from Gherkin for LLM consumption.

```python
from code2logic import gherkin_to_test_data

test_data = gherkin_to_test_data(gherkin_content)
print(f"Features: {len(test_data['features'])}")
print(f"Scenarios: {test_data['total_scenarios']}")
```

---

### Components

#### `ProjectAnalyzer`

Low-level project analyzer class.

```python
from code2logic import ProjectAnalyzer

analyzer = ProjectAnalyzer(
    root_path="/path/to/project",
    use_treesitter=True,
    verbose=False,
    include_private=False
)
project = analyzer.analyze()
stats = analyzer.get_statistics()
```

**Methods:**
- `analyze()` → ProjectInfo
- `get_statistics()` → dict

---

#### `TreeSitterParser`

High-accuracy AST parser using Tree-sitter.

```python
from code2logic import TreeSitterParser

parser = TreeSitterParser(verbose=False)
module_info = parser.parse_file("/path/to/file.py")
```

---

#### `UniversalParser`

Fallback parser using regex patterns.

```python
from code2logic import UniversalParser

parser = UniversalParser(verbose=False)
module_info = parser.parse_file("/path/to/file.py")
```

---

#### `DependencyAnalyzer`

Analyze dependencies using NetworkX.

```python
from code2logic import DependencyAnalyzer

analyzer = DependencyAnalyzer(verbose=False)
graph, metrics = analyzer.analyze(modules)
```

---

#### `SimilarityDetector`

Find similar/duplicate functions using Rapidfuzz.

```python
from code2logic import SimilarityDetector

detector = SimilarityDetector(threshold=0.8)
similar = detector.find_similar(functions)
```

---

#### `EnhancedIntentGenerator`

Generate human-readable function intents using NLP.

```python
from code2logic import EnhancedIntentGenerator

generator = EnhancedIntentGenerator()
intent = generator.generate_intent(function_name, docstring)
```

---

## Models

### `ProjectInfo`

```python
@dataclass
class ProjectInfo:
    name: str                           # Project name
    root_path: str                      # Root directory path
    languages: Dict[str, int]           # Language → file count
    modules: List[ModuleInfo]           # All analyzed modules
    dependency_graph: Dict[str, List]   # Module dependencies
    dependency_metrics: Dict            # PageRank, hub detection
    entrypoints: List[str]              # Entry point files
    similar_functions: Dict             # Similar function groups
    total_files: int                    # Total file count
    total_lines: int                    # Total line count
    generated_at: str                   # Timestamp
```

### `ModuleInfo`

```python
@dataclass
class ModuleInfo:
    path: str                           # File path
    language: str                       # Programming language
    imports: List[str]                  # Import statements
    exports: List[str]                  # Exported symbols
    classes: List[ClassInfo]            # Classes in module
    functions: List[FunctionInfo]       # Functions in module
    types: List[TypeInfo]               # Type definitions
    constants: List[str]                # Constants
    docstring: Optional[str]            # Module docstring
    lines_total: int                    # Total lines
    lines_code: int                     # Code lines (no comments)
```

### `FunctionInfo`

```python
@dataclass
class FunctionInfo:
    name: str                           # Function name
    params: List[str]                   # Parameters
    return_type: Optional[str]          # Return type annotation
    docstring: Optional[str]            # Function docstring
    calls: List[str]                    # Called functions
    raises: List[str]                   # Raised exceptions
    complexity: int                     # Cyclomatic complexity
    lines: int                          # Line count
    decorators: List[str]               # Applied decorators
    is_async: bool                      # Async function flag
    is_static: bool                     # Static method flag
    is_private: bool                    # Private function flag
    intent: str                         # Generated intent description
    start_line: int                     # Start line number
    end_line: int                       # End line number
```

### `ClassInfo`

```python
@dataclass
class ClassInfo:
    name: str                           # Class name
    bases: List[str]                    # Base classes
    docstring: Optional[str]            # Class docstring
    methods: List[FunctionInfo]         # Class methods
    properties: List[str]               # Properties
    is_interface: bool                  # Interface/protocol flag
    is_abstract: bool                   # Abstract class flag
    generic_params: List[str]           # Generic type parameters
```

---

## Output Formats Comparison

| Format | Tokens/100 funcs | LLM Accuracy | Best For |
|--------|-----------------|--------------|----------|
| Gherkin | ~300 | 95% | LLM context, BDD tests |
| Compact | ~500 | 50% | Quick overview |
| CSV | ~16K | 70% | Spreadsheets, analysis |
| YAML | ~20K | 90% | Human-readable config |
| JSON | ~25K | 75% | RAG, embeddings |
| Markdown | ~35K | 60% | Documentation |

---

## Examples

All examples are in the `examples/` directory:

| Example | Description |
|---------|-------------|
| `01_quick_start.py` | Basic usage guide |
| `02_refactoring.py` | Duplicate + quality analysis |
| `03_reproduction.py` | Reproduce code from specs |
| `04_project.py` | Project-level reproduction |
| `05_llm_integration.py` | LLM integration demo |
| `06_metrics.py` | Detailed reproduction metrics |
| `08_format_benchmark.py` | Benchmark formats across files |
| `09_async_benchmark.py` | Parallel benchmark |
| `10_function_reproduction.py` | Function-level reproduction |
| `11_token_benchmark.py` | Token-aware benchmark |
| `12_comprehensive_analysis.py` | Comprehensive format analysis |
| `13_project_benchmark.py` | Whole-project benchmark |
| `14_repeatability_test.py` | Repeatability testing |
| `15_unified_benchmark.py` | Unified benchmark runner |
| `16_terminal_demo.py` | Terminal markdown rendering demo |

Run examples:

```bash
python examples/01_quick_start.py
python examples/02_refactoring.py
python examples/03_reproduction.py --show-spec
python examples/04_project.py tests/samples/ --no-llm
python examples/11_token_benchmark.py --folder tests/samples/ --no-llm
python examples/15_unified_benchmark.py --type format --folder tests/samples/ --no-llm
python examples/16_terminal_demo.py --folder tests/samples/
```

---

## MCP Server (Claude Desktop)

Add to `claude_desktop_config.json`:

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

Then in Claude Desktop:
- "Analyze my project at /path/to/project"
- "Find duplicates in my codebase"
- "Generate BDD tests for my API"

---

## LLM Integration

### With Ollama

```python
from code2logic import analyze_project, CSVGenerator
from code2logic.llm import CodeAnalyzer

# Analyze project
project = analyze_project("/path/to/project")

# Use with Ollama
analyzer = CodeAnalyzer(model="qwen2.5-coder:7b")
suggestions = analyzer.suggest_refactoring(project)
duplicates = analyzer.find_semantic_duplicates(project)
```

### With LiteLLM

```python
from code2logic.llm import LiteLLMClient

client = LiteLLMClient(model="gpt-4")
response = client.chat([
    {"role": "user", "content": f"Analyze this code:\n{csv_output}"}
])
```
