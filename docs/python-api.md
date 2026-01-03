# Python API Reference

> Programmatic usage of Code2Logic

[← README](../README.md) | [← CLI Reference](cli-reference.md) | [Output Formats →](output-formats.md)

## Core Functions

### analyze_project

Main entry point for project analysis.

```python
from code2logic import analyze_project

project = analyze_project(
    path="/path/to/project",
    use_treesitter=True,   # Use Tree-sitter parser
    verbose=False,          # Print progress
    include_private=True    # Include private functions
)
```

**Returns:** `ProjectInfo` object

### ProjectInfo

```python
@dataclass
class ProjectInfo:
    name: str                    # Project name
    root_path: str              # Root directory
    total_files: int            # Number of files
    total_lines: int            # Total lines of code
    languages: Dict[str, int]   # Language breakdown
    modules: List[ModuleInfo]   # Module list
    entrypoints: List[str]      # Entry point files
    dependencies: Dict          # Dependencies
```

### ModuleInfo

```python
@dataclass
class ModuleInfo:
    path: str                   # File path
    language: str               # Programming language
    lines_code: int             # Lines of code
    lines_comment: int          # Comment lines
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    imports: List[str]
    docstring: str
```

### FunctionInfo

```python
@dataclass
class FunctionInfo:
    name: str                   # Function name
    params: List[str]           # Parameters
    return_type: str            # Return type
    docstring: str              # Docstring
    lines: int                  # Line count
    complexity: int             # Cyclomatic complexity
    calls: List[str]            # Function calls made
    is_async: bool              # Async function
    is_generator: bool          # Generator function
    decorators: List[str]       # Decorators
    intent: str                 # Inferred intent
    category: str               # Category (lifecycle, validate, etc.)
    raises: List[str]           # Exceptions raised
```

### ClassInfo

```python
@dataclass
class ClassInfo:
    name: str                   # Class name
    bases: List[str]            # Base classes
    docstring: str              # Docstring
    methods: List[FunctionInfo] # Methods
    properties: List[str]       # Properties
    decorators: List[str]       # Class decorators
```

## Generators

### MarkdownGenerator

```python
from code2logic.generators import MarkdownGenerator

gen = MarkdownGenerator()
output = gen.generate(project, detail='standard')
# detail: 'minimal', 'standard', 'full'
```

### JSONGenerator

```python
from code2logic.generators import JSONGenerator

gen = JSONGenerator()
output = gen.generate(project, flat=False, detail='standard')
# flat=True for RAG-friendly format
```

### YAMLGenerator

```python
from code2logic.generators import YAMLGenerator

gen = YAMLGenerator()
output = gen.generate(project, flat=False, detail='standard')
```

### CSVGenerator

```python
from code2logic.generators import CSVGenerator

gen = CSVGenerator()
output = gen.generate(project, detail='standard')
```

### CompactGenerator

```python
from code2logic.generators import CompactGenerator

gen = CompactGenerator()
output = gen.generate(project)
# No detail parameter - always ultra-compact
```

### GherkinGenerator

```python
from code2logic.gherkin import GherkinGenerator

gen = GherkinGenerator()
output = gen.generate(project, detail='standard')
```

### StepDefinitionGenerator

```python
from code2logic.gherkin import StepDefinitionGenerator

gen = StepDefinitionGenerator()
output = gen.generate(project)
# Generates pytest-bdd step definitions
```

## LLM Clients

### OllamaClient

```python
from code2logic.llm import OllamaClient

client = OllamaClient(
    model="qwen2.5-coder:7b",
    host="http://localhost:11434"
)

# Check availability
if client.is_available():
    # Generate completion
    response = client.generate(
        prompt="Explain this code",
        system="You are a code expert"
    )
    
    # Chat completion
    response = client.chat([
        {"role": "user", "content": "Hello"}
    ])
    
    # List models
    models = client.list_models()
```

### LiteLLMClient

```python
from code2logic.llm import LiteLLMClient

client = LiteLLMClient(model="gpt-4")

response = client.generate(
    prompt="Explain this code",
    system="You are a code expert"
)
```

### CodeAnalyzer

High-level LLM analysis:

```python
from code2logic.llm import CodeAnalyzer

analyzer = CodeAnalyzer(model="qwen2.5-coder:7b")

# Analyze code
result = analyzer.analyze_code(code_string)

# Get refactoring suggestions
suggestions = analyzer.suggest_refactoring(code_string)

# Generate documentation
docs = analyzer.generate_documentation(code_string)
```

## Configuration

### Config Class

```python
from code2logic.config import Config

config = Config()

# Get API key
api_key = config.get_api_key('openrouter')

# Get model
model = config.get_model('ollama')

# List providers
providers = config.list_configured_providers()
# {'openrouter': True, 'openai': False, 'ollama': True, ...}

# Export config
config_dict = config.to_dict()
```

### Environment Loading

```python
from code2logic.config import load_env, get_api_key, get_model

# Load .env file
load_env()

# Get specific values
key = get_api_key('openrouter')
model = get_model('ollama')
```

## Parsers

### TreeSitterParser

```python
from code2logic.parsers import TreeSitterParser

parser = TreeSitterParser(verbose=False)

# Parse file
result = parser.parse_file("/path/to/file.py")
# Returns: ModuleInfo
```

### UniversalParser

Fallback parser when Tree-sitter unavailable:

```python
from code2logic.parsers import UniversalParser

parser = UniversalParser(verbose=False)
result = parser.parse_file("/path/to/file.py")
```

## Analyzers

### ProjectAnalyzer

```python
from code2logic.analyzer import ProjectAnalyzer

analyzer = ProjectAnalyzer(
    root_path="/path/to/project",
    use_treesitter=True,
    verbose=False,
    include_private=True
)

project = analyzer.analyze()
stats = analyzer.get_statistics()
```

### DependencyAnalyzer

```python
from code2logic.dependency import DependencyAnalyzer

analyzer = DependencyAnalyzer(verbose=False)
deps = analyzer.analyze(project)
# Returns dependency graph
```

### SimilarityDetector

```python
from code2logic.similarity import SimilarityDetector

detector = SimilarityDetector()
duplicates = detector.find_duplicates(project)
similar = detector.find_similar(func1, func2)
```

## Complete Example

```python
from code2logic import analyze_project
from code2logic.generators import CSVGenerator
from code2logic.gherkin import GherkinGenerator
from code2logic.llm import OllamaClient

# 1. Analyze project
project = analyze_project("./my_project")
print(f"Analyzed {project.total_files} files")

# 2. Generate outputs
csv = CSVGenerator().generate(project, detail='standard')
gherkin = GherkinGenerator().generate(project, detail='minimal')

# 3. Save outputs
with open("analysis.csv", "w") as f:
    f.write(csv)

with open("analysis.feature", "w") as f:
    f.write(gherkin)

# 4. Use with LLM
client = OllamaClient(model="qwen2.5-coder:7b")
if client.is_available():
    response = client.generate(
        prompt=f"Review this code structure:\n{gherkin[:2000]}",
        system="You are a senior developer."
    )
    print(response)
```

---

[← CLI Reference](cli-reference.md) | [Output Formats →](output-formats.md)
