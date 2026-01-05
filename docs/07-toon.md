# TOON Format

> Token-Oriented Object Notation - Ultra-compact format for LLM consumption

[← README](../README.md) | [← Index](00-index.md) | [Format Specs →](06-format-specifications.md)

## Overview

TOON (Token-Oriented Object Notation) is a compact data format designed specifically for LLM input. It combines:
- **YAML-like** indentation for nesting
- **CSV-style** tabular arrays for uniform data
- **Minimal quoting** to reduce token usage

### Token Efficiency

| Format | Size | Tokens (~) | vs JSON |
|--------|------|------------|---------|
| **TOON** | 9,113 | ~2,278 | **17.0%** |
| Gherkin | 13,047 | ~3,261 | 24.4% |
| Markdown | 17,450 | ~4,362 | 32.6% |
| LogicML | 18,733 | ~4,683 | 35.0% |
| YAML | 23,537 | ~5,884 | 44.0% |
| JSON | 53,478 | ~13,369 | 100% |

**TOON is 6x smaller than JSON** - the most token-efficient format.

## Installation

TOON is included in Code2Logic by default:

```python
from code2logic import TOONGenerator, TOONParser, parse_toon
```

## Usage

### Generate TOON

```python
from code2logic import analyze_project, TOONGenerator

project = analyze_project('/path/to/project')

# Standard TOON output
gen = TOONGenerator()
toon_output = gen.generate(project, detail='full')

# Ultra-compact TOON (71% smaller)
ultra_compact = gen.generate_ultra_compact(project)

# Generate JSON Schema
schema = gen.generate_schema('ultra_compact')  # or 'standard'

# With tabs (better for some LLMs)
gen_tabs = TOONGenerator(use_tabs=True)
toon_tabs = gen_tabs.generate(project)

print(f"Standard: {len(toon_output)} chars")
print(f"Ultra-compact: {len(ultra_compact)} chars")
```

### Ultra-Compact Format

New ultra-compact format with single-letter keys for maximum token efficiency:

```python
# Ultra-compact TOON
ultra = gen.generate_ultra_compact(project)
print(ultra)
```

**Sample Ultra-Compact Output:**
```
# myproject | 15f 1700L | python:15
# Keys: M=modules, D=details, i=imports, c=classes, f=functions, m=methods
M[2]:
  utils.py,50
  main.py,100

D:
  utils.py:
    i: typing.{Dict,List},os.path
    c: Helper: get_version(0), validate_input(1)
    f: format_data()->str, load_config(path:str)->dict
  main.py:
    i: utils,click
    f: main()->None, cli()->None
```

**Schema Generation:**
```python
# Generate JSON Schema for validation
schema = gen.generate_schema('ultra_compact')
with open('toon-schema.json', 'w') as f:
    f.write(schema)
```

### Parse TOON

```python
from code2logic import parse_toon

toon_data = """
project: my_project
modules[2]{path,lang,lines}:
  utils.py,python,50
  main.py,python,100
"""

parsed = parse_toon(toon_data)
print(parsed['modules'])
# [{'path': 'utils.py', 'lang': 'python', 'lines': '50'}, ...]
```

### CLI Usage

```bash
# Generate TOON output
code2logic /path/to/project -f toon -o analysis.toon

# With full detail
code2logic /path/to/project -f toon -d full
```

## Format Syntax

### Basic Key-Value

```toon
project: my_project
version: 1.0.0
enabled: true
```

### Nested Objects

```toon
stats:
  files: 15
  lines: 1700
  languages:
    python: 10
    javascript: 5
```

### Tabular Arrays

Most powerful feature - CSV-style arrays with schema:

```toon
modules[3]{path,language,lines}:
  utils.py,python,50
  main.py,python,100
  helpers.js,javascript,30
```

Equivalent JSON:
```json
{
  "modules": [
    {"path": "utils.py", "language": "python", "lines": "50"},
    {"path": "main.py", "language": "python", "lines": "100"},
    {"path": "helpers.js", "language": "javascript", "lines": "30"}
  ]
}
```

### Primitive Arrays

Simple comma-separated values:

```toon
tags[3]: api,utils,core
imports[5]: typing,os,sys,json,pathlib
```

### Quoting Rules

Quote only when necessary:

```toon
# No quotes needed
name: simple_name
path: src/main.py

# Quotes required (special chars)
signature: "(self, data: str) -> bool"
description: "Process data, validate, and return"
```

## Complete Example

```toon
project: my_project
root: /home/user/project
stats:
  files: 2
  lines: 150
  languages[1]: python:2

modules[2]{path,lang,lines}:
  utils.py,python,50
  main.py,python,100

module_details:
  main.py:
    imports[3]: typing,dataclasses,pathlib
    classes[1]{name,bases,methods}:
      Application,BaseClass,5
    class_details:
      Application:
        doc: "Main application class"
        properties[2]: name:str,config:dict
        methods[3]{name,sig,async,lines}:
          __init__,"(self;name:str)",false,5
          run,(self)->int,false,10
          start,"(self;port:int)->None",true,8
    functions[1]{name,sig,intent}:
      main,()->None,"Entry point"
```

## Key Features

| Feature | Syntax | Description |
|---------|--------|-------------|
| **Tabular arrays** | `items[N]{f1,f2}:` | CSV-style for uniform data |
| **Primitive arrays** | `tags[N]: a,b,c` | Simple value lists |
| **Minimal quoting** | Only when needed | Reduces tokens |
| **Indentation** | 2 spaces | YAML-style nesting |
| **Explicit lengths** | `[N]` suffix | Helps LLM parsing |
| **Schema hints** | `{field1,field2}` | Clear structure |

## Comparison with Other Formats

### vs JSON
- **6x smaller** token usage
- Same data structure support
- Human readable

### vs YAML
- **2.5x smaller** for arrays
- Tabular syntax for uniform data
- Less verbose

### vs LogicML
- Similar compression
- Better for general data
- Simpler syntax

## API Reference

### TOONGenerator

```python
class TOONGenerator:
    def __init__(self, use_tabs: bool = False):
        """
        Args:
            use_tabs: Use tabs instead of spaces for indentation
        """
    
    def generate(self, project: ProjectInfo, detail: str = 'standard') -> str:
        """
        Generate TOON output.
        
        Args:
            project: Analyzed project info
            detail: 'minimal', 'standard', or 'full'
        
        Returns:
            TOON formatted string
        """
```

### TOONParser

```python
class TOONParser:
    def parse(self, content: str) -> dict:
        """
        Parse TOON content to dictionary.
        
        Args:
            content: TOON formatted string
        
        Returns:
            Parsed dictionary
        """
```

### Convenience Functions

```python
from code2logic import generate_toon, parse_toon

# Generate
toon_str = generate_toon(project)

# Parse
data = parse_toon(toon_str)
```

## Benchmarks

Run TOON benchmarks:

```bash
# Token benchmark with TOON
python examples/11_token_benchmark.py --formats json yaml toon

# Compare all formats
python examples/12_comprehensive_analysis.py
```

## See Also

- [Output Formats](05-output-formats.md) - All format comparison
- [Format Specifications](06-format-specifications.md) - Detailed specs
- [Benchmarking](10-benchmark.md) - Benchmark results
- [TOON GitHub](https://github.com/toon-format/toon) - Original spec

---

[← Format Specs](06-format-specifications.md) | [LLM Integration →](08-llm-integration.md)
