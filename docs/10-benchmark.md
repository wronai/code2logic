# Code2Logic Benchmark Guide

[← README](../README.md) | [Docs Index](00-index.md)

## Overview

This document provides comprehensive benchmark results comparing different specification formats for code reproduction using LLMs.

## Format Comparison

### Benchmark Results (tests/samples/)

| Format | Score | Syntax OK | Compression | Token Eff | Avg Lines |
|--------|-------|-----------|-------------|-----------|-----------|
| **YAML** | **70.3%** | **100%** | 0.6x | 36.4 | 257 |
| **LogicML** | 63.6% | **100%** | **0.42x** | 35.9 | 204 |
| JSON | 62.3% | 50% | 1.1x | 21.7 | 301 |
| Markdown | 61.2% | 75% | 0.5x | 43.9 | 206 |
| Gherkin | 44.1% | 50% | 0.6x | 12.5 | 339 |

### Key Metrics Explained

- **Score**: Overall reproduction quality (text + structural + semantic similarity)
- **Syntax OK**: Percentage of generated code that compiles without errors
- **Compression**: Ratio of spec size to original code size (lower = better)
- **Token Efficiency**: Score per 100 tokens (higher = better)
- **Avg Lines**: Average lines of generated code

## Format Recommendations

| Use Case | Recommended Format | Reason |
|----------|-------------------|--------|
| **Production code** | YAML | Highest score (70.3%), 100% syntax OK |
| **Token-limited LLMs** | LogicML | Best compression (0.42x) |
| **Documentation** | Markdown | Good balance, readable |
| **Avoid** | Gherkin | Over-engineering, lowest score |

## Running Benchmarks

### Token Benchmark

Compare formats with token usage tracking:

```bash
# All formats
python examples/11_token_benchmark.py \
  --folder tests/samples/ \
  --formats yaml logicml markdown gherkin json

# Best formats only
python examples/11_token_benchmark.py \
  --folder tests/samples/ \
  --formats yaml logicml

# Limit files
python examples/11_token_benchmark.py \
  --folder tests/samples/ \
  --formats yaml logicml \
  --limit 5
```

### Project Benchmark

Test entire project structure:

```bash
# Analyze project
python examples/13_project_benchmark.py \
  --project /path/to/project \
  --formats yaml logicml markdown

# With verbose output
python examples/13_project_benchmark.py \
  --project /path/to/project \
  --formats yaml logicml \
  --verbose
```

### Comprehensive Analysis

Analyze generated code quality:

```bash
python examples/12_comprehensive_analysis.py
```

## Sample Output

### YAML Format (95 lines)
```yaml
# sample_class.py | Calculator | 74 lines

imports:
  stdlib: [typing.List, typing.Optional]

Calculator:
  doc: "Simple calculator with history."
  attrs:
    precision: int
    history: List[str]
  methods:
    add:
      sig: (a: float, b: float) -> float
      intent: "Add two numbers"
```

Generated Python:
```python
class Calculator:
    """Simple calculator with history."""
    
    def __init__(self, precision: int) -> None:
        self.precision = precision
        self.history: List[str] = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"add({a}, {b}) = {result}")
        return result
```

### LogicML Format (best compression)
```yaml
# sample_class.py | Calculator | 74 lines

imports:
  stdlib: [typing.List, typing.Optional]

Calculator:
  doc: "Simple calculator with history."
  attrs:
    precision: int
    history: List[str]
  methods:
    add:
      sig: (a: float, b: float) -> float
      does: "Add two numbers"
      side: "Modifies list"
    divide:
      sig: (a: float, b: float) -> Optional[float]
      does: "Divide a by b"
      edge: "b == 0 → return None"
```

### Gherkin Format (not recommended)
```gherkin
Feature: Calculator
  Simple calculator with history.

  Scenario: Add two numbers
    Given a calculator with precision 2
    When I add 2.5 and 3.5
    Then the result should be 6.0
    And the history should contain "add(2.5, 3.5) = 6.0"
```

**Problem**: Generates 2x more code with unnecessary error classes and over-engineering.

## Benchmark Scripts

| Script | Purpose |
|--------|---------|
| `11_token_benchmark.py` | Token-aware format comparison |
| `12_comprehensive_analysis.py` | Generated code quality analysis |
| `13_project_benchmark.py` | Entire project structure testing |
| `14_repeatability_test.py` | Code generation repeatability |

## API Usage

### Using LogicML (best compression)

```python
from code2logic import analyze_project, LogicMLGenerator

project = analyze_project('/path/to/project')
gen = LogicMLGenerator()
spec = gen.generate(project)

print(f"Tokens: ~{spec.token_estimate}")
print(spec.content)
```

### Using YAML (best quality)

```python
from code2logic import analyze_project
from code2logic.generators import YAMLGenerator

project = analyze_project('/path/to/project')
gen = YAMLGenerator()
spec = gen.generate(project, detail='full')

print(spec)
```

### Comparing Formats

```python
from code2logic import analyze_project
from code2logic.generators import YAMLGenerator
from code2logic.logicml import LogicMLGenerator
from code2logic.gherkin import GherkinGenerator

project = analyze_project('/path/to/project')

formats = {
    'yaml': YAMLGenerator().generate(project, detail='full'),
    'logicml': LogicMLGenerator().generate(project).content,
    'gherkin': GherkinGenerator().generate(project),
}

for name, spec in formats.items():
    tokens = len(spec) // 4
    print(f"{name}: {tokens} tokens, {len(spec)} chars")
```

## Conclusions

### Latest Benchmark Results (January 2026)

| Format | Score | Success | Text Sim | Structural | Semantic |
|--------|-------|---------|----------|------------|----------|
| **YAML** | **74.5%** | 100% | **91.8%** | **80.0%** | **83.0%** |
| **LogicML** | **65.9%** | 100% | 89.7% | 66.7% | 74.7% |
| Gherkin | 50.2% | 33% | 64.0% | 6.7% | 78.1% |

### Key Findings

1. **YAML is the best for quality** - 74.5% score, 100% success rate
2. **LogicML is the best for compression** - 0.51x compression ratio
3. **LogicML has 100% success rate** - reliable code generation
4. **Gherkin over-engineers** - only 33% success rate, creates extra classes
5. **LogicML handles async code well** - 76.2% score on async files

### Format Strengths

| Format | Best For |
|--------|----------|
| **YAML** | Overall quality, text similarity |
| **LogicML** | Compression, token efficiency, async code |
| **Gherkin** | Semantic descriptions (but low success) |

### Improvements Made

- LogicML prompt improved for async code handling
- Added explicit instructions for `sig: async (...)` pattern
- Better dataclass handling with `abstract: true`
- Improved docstring extraction for semantic reproduction

## Environment Requirements

```bash
pip install python-dotenv httpx pyyaml
```

For LLM integration:
```bash
# OpenRouter (recommended)
export OPENROUTER_API_KEY=your_key

# Or Ollama (local)
ollama serve
```

## Repeatability Analysis

Testing how consistent code generation is across multiple invocations.

### Running Repeatability Test

```bash
# Test with 3 runs (default)
python examples/14_repeatability_test.py \
  --file tests/samples/sample_class.py

# Test with 5 runs
python examples/14_repeatability_test.py \
  --file tests/samples/sample_class.py \
  --runs 5 \
  --formats yaml logicml gherkin
```

### Repeatability Results

| Format | Avg Similarity | Min Sim | Max Sim | Line Variance | Syntax OK |
|--------|---------------|---------|---------|---------------|-----------|
| **LogicML** | **56.9%** | 44.5% | **80.9%** | 94.9 | 100% |
| YAML | 41.0% | 29.3% | 61.8% | **8.7** | 100% |
| Gherkin | 14.1% | 2.0% | 24.9% | 374.9 | 100% |

### Key Findings

1. **LogicML is most consistent** - 56.9% average similarity between runs
2. **YAML has lowest variance** - 8.7 line variance (most stable output size)
3. **Gherkin is highly variable** - only 14.1% similarity, 375 line variance

### Code Differences Between Runs

| Format | Changed Lines | Stability |
|--------|--------------|-----------|
| YAML | 57 | Good |
| LogicML | 98 | Medium |
| Gherkin | 118 | Poor |

### Recommendations for Repeatability

1. **Use YAML for stable output** - lowest line variance
2. **Use LogicML for consistent logic** - highest similarity between runs
3. **Avoid Gherkin** - highly variable, unpredictable output

## Output Files

Benchmark results are saved to:
- `examples/output/token_benchmark.json`
- `examples/output/project_benchmark.json`
- `examples/output/comprehensive_analysis.json`
- `examples/output/repeatability_test.json`
- `examples/output/generated/{format}/` - Generated code files



