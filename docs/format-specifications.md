# Format Specifications

Complete specifications for Code2Logic output formats.

## Overview

| Format | Score | Syntax OK | Compression | Best For |
|--------|-------|-----------|-------------|----------|
| **YAML** | 70.5% | 100% | 0.6x | Overall quality |
| **LogicML** | 63.9% | 100% | 0.5x | Token efficiency |
| **Markdown** | ~60% | ~80% | 0.5x | Documentation |
| **Gherkin** | ~50% | ~70% | 0.6x | BDD scenarios |

---

## YAML Format

### Schema

```yaml
project: <project_name>
statistics:
  files: <int>
  lines: <int>
  languages:
    python: <count>
    typescript: <count>

modules:
  - path: <relative_path>
    language: python|typescript|javascript
    lines: <int>
    imports:
      - <import_path>
    exports:
      - <export_name>
    classes:
      - name: <ClassName>
        bases: [BaseClass1, BaseClass2]
        docstring: "<description>"
        methods:
          - name: <method_name>
            signature: (self, param1: Type1, param2: Type2) -> ReturnType
            intent: "<what it does>"
            lines: <int>
            is_async: true|false
        properties:
          - <attr_name>: <Type>
    functions:
      - name: <function_name>
        signature: (param1: Type1) -> ReturnType
        intent: "<description>"
        is_async: true|false
```

### Validation

```python
from code2logic import validate_yaml

spec = """
project: myproject
modules:
  - path: main.py
    classes:
      - name: MyClass
        methods:
          - name: __init__
            signature: (self)
"""

is_valid, errors = validate_yaml(spec)
if not is_valid:
    print("Errors:", errors)
```

### Best Practices

1. Always include `signature` with full type hints
2. Use `intent` for clear docstrings
3. Include `is_async: true` for async methods
4. List all imports for complete reproduction

---

## LogicML Format

### Schema

```yaml
# filename.py | ClassName, Class2 | N lines

imports:
  stdlib: [typing.List, datetime.datetime]
  third_party: [pydantic.BaseModel, pydantic.Field]
  local: [.submodule.Helper]

ClassName:
  doc: "Class description"
  bases: [BaseModel]
  # Pydantic model - use Field() for attributes
  attrs:
    attr_name: Type
    other_attr: Optional[str]
  methods:
    __init__:
      sig: (self, param: Type) -> None
      does: "Initialize instance"
    async_method:
      sig: async (self, data: dict) -> Result
      does: "Process data asynchronously"
      side: "Modifies internal state"
    property_method:
      sig: @property (self) -> str
      does: "Get computed value"
    error_handler:
      sig: (self, value: int) -> Optional[int]
      does: "Handle value"
      edge: "value < 0 → return None"

functions:
  helper_func:
    sig: (items: List[T]) -> List[T]
    does: "Process items"

# Re-export module (for __init__.py files)
type: re-export
exports:
  - ClassName
  - helper_func
```

### Key Features

| Feature | Syntax | Example |
|---------|--------|---------|
| **Signature** | `sig:` | `(self, x: int) -> str` |
| **Async** | `async` prefix | `async (self) -> None` |
| **Property** | `@property` prefix | `@property (self) -> Type` |
| **Docstring** | `does:` | `"What it does"` |
| **Side effect** | `side:` | `"Modifies list"` |
| **Edge case** | `edge:` | `"x < 0 → return None"` |
| **Pydantic** | `bases: [BaseModel]` | Auto-generates Field() |
| **Re-export** | `type: re-export` | For `__init__.py` |

### Validation

```python
from code2logic import validate_logicml

spec = """
# main.py | Calculator | 50 lines

Calculator:
  doc: "Simple calculator"
  methods:
    add:
      sig: (self, a: float, b: float) -> float
      does: "Add two numbers"
"""

is_valid, errors = validate_logicml(spec)
```

### Best Practices

1. Always start with header comment: `# filename | classes | lines`
2. Use `bases: [BaseModel]` for Pydantic models
3. Include `side:` for methods that modify state
4. Use `edge:` for important edge cases
5. For re-export modules, use `type: re-export`

---

## Markdown Hybrid Format

### Schema

```markdown
# Module: filename.py

## Metadata
- Language: python
- Lines: 150
- Imports: typing, dataclasses

## Classes

### ClassName

Base classes: BaseClass

**Attributes:**
- attr1: str
- attr2: Optional[int]

**Methods:**

#### method_name

```yaml
signature: (self, param: Type) -> ReturnType
async: false
```

```gherkin
Scenario: Method behavior
  Given a valid input
  When method is called
  Then returns expected result
```

## Functions

### helper_function

```yaml
signature: (data: List[str]) -> Dict[str, int]
```
```

### Validation

```python
from code2logic import validate_markdown

spec = """
# Module: utils.py

## Classes

### Helper
**Methods:**
#### process
```yaml
signature: (self, data: str) -> str
```
"""

is_valid, errors = validate_markdown(spec)
```

---

## Format Selection Guide

### By Use Case

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| Production code | YAML | Highest score (70.5%) |
| Token-limited LLMs | LogicML | Best compression (0.5x) |
| Documentation | Markdown | Human readable |
| Async Python | LogicML | Best async handling |
| Pydantic models | LogicML | Built-in support |
| TypeScript | YAML | Better re-export handling |
| Re-export modules | LogicML | `type: re-export` |

### By Metric

| Metric | Winner | Value |
|--------|--------|-------|
| Score | YAML | 70.5% |
| Syntax OK | Both | 100% |
| Compression | LogicML | 0.5x |
| Token Efficiency | YAML | 44.2 |

---

## API Usage

### Generate and Validate

```python
from code2logic import (
    analyze_project,
    YAMLGenerator,
    LogicMLGenerator,
    validate_yaml,
    validate_logicml,
)

# Analyze project
project = analyze_project('/path/to/project')

# Generate YAML
yaml_gen = YAMLGenerator()
yaml_spec = yaml_gen.generate(project, detail='full')
is_valid, errors = validate_yaml(yaml_spec)

# Generate LogicML
logicml_gen = LogicMLGenerator()
logicml_spec = logicml_gen.generate(project)
is_valid, errors = validate_logicml(logicml_spec.content)
```

### Benchmarking

```bash
# Compare formats
python examples/11_token_benchmark.py \
  --folder /path/to/code \
  --formats yaml logicml markdown

# Project-level benchmark
python examples/13_project_benchmark.py \
  --project /path/to/project \
  --formats yaml logicml
```
