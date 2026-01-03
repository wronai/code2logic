# Code2Logic - Refactoring Plan

## Overview

This document outlines the refactoring tasks identified during code analysis. Tasks are prioritized by impact and effort.

---

## 🔴 High Priority (Immediate)

### 1. Extract Duplicate Functions
**Effort:** Low | **Impact:** High

Multiple `main()` functions with identical signatures across files:
- `code2logic/cli.py::main`
- `examples/refactor_suggestions.py::main`
- `examples/token_efficiency.py::main`
- `examples/quick_start.py::main`
- `examples/generate_code.py::main`
- `examples/duplicate_detection.py::main`

**Action:** This is acceptable for CLI entry points - no action needed.

---

### 2. Consolidate LLM Client Classes
**Effort:** Medium | **Impact:** High

`OllamaClient` and `LiteLLMClient` have duplicate methods:
- `__init__`
- `generate`
- `chat`
- `is_available`

**Action:** Create abstract `BaseLLMClient` class:

```python
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    def __init__(self, model: str, host: str = None):
        self.model = model
        self.host = host
    
    @abstractmethod
    def generate(self, prompt: str, system: str = None) -> str:
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict]) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
```

**Files:** `code2logic/llm.py`

---

### 3. Consolidate Parser Init Methods
**Effort:** Low | **Impact:** Medium

`TreeSitterParser`, `UniversalParser`, and `DependencyAnalyzer` have similar `__init__`:

```python
def __init__(self, verbose: bool = False):
    self.verbose = verbose
```

**Action:** Create mixin or base class:

```python
class VerboseMixin:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def log(self, msg: str):
        if self.verbose:
            print(f"[{self.__class__.__name__}] {msg}")
```

**Files:** `code2logic/parsers.py`, `code2logic/dependency.py`

---

## 🟡 Medium Priority (Short-term)

### 4. Split Long Files
**Effort:** Medium | **Impact:** Medium

Files exceeding 500 lines:
- `code2logic/generators.py` (1017 lines) - Split into separate generator modules
- `code2logic/gherkin.py` (975 lines) - Extract step definitions
- `code2logic/parsers.py` (908 lines) - Split by language

**Action:** Create subpackages:

```
code2logic/
├── generators/
│   ├── __init__.py
│   ├── base.py
│   ├── markdown.py
│   ├── json.py
│   ├── yaml.py
│   ├── csv.py
│   └── compact.py
├── parsers/
│   ├── __init__.py
│   ├── base.py
│   ├── treesitter.py
│   └── fallback.py
└── gherkin/
    ├── __init__.py
    ├── generator.py
    ├── steps.py
    └── cucumber.py
```

---

### 5. Refactor Long Functions
**Effort:** Medium | **Impact:** Medium

Functions exceeding 50 lines:
- `generators.py::MarkdownGenerator._generate_module_section` (85 lines)
- `gherkin.py::GherkinGenerator._categorize_functions` (72 lines)
- `parsers.py::TreeSitterParser._parse_python_function` (68 lines)

**Action:** Extract helper methods for each logical block.

---

### 6. Add Type Hints
**Effort:** Low | **Impact:** Medium

Several functions lack complete type annotations. Run mypy and fix:

```bash
mypy code2logic --strict
```

---

### 7. Improve Test Coverage
**Effort:** Medium | **Impact:** High

Current coverage: ~31%

Priority areas:
- `code2logic/generators.py` (40% coverage)
- `code2logic/gherkin.py` (10% coverage)
- `code2logic/parsers.py` (18% coverage)
- `code2logic/llm.py` (0% coverage)
- `code2logic/cli.py` (0% coverage)

**Target:** 80% coverage

---

## 🟢 Low Priority (Long-term)

### 8. Large Classes (SRP Violation)
**Effort:** High | **Impact:** Medium

Classes with >20 methods:
- `GherkinGenerator` (24 methods)
- `MarkdownGenerator` (18 methods)

**Action:** Consider splitting by responsibility:
- `GherkinGenerator` → `GherkinParser`, `GherkinFormatter`, `GherkinWriter`

---

### 9. Signature Duplicates
**Effort:** Low | **Impact:** Low

17 groups of functions with identical signatures but different names.

**Action:** Review for potential generic implementations.

---

### 10. Documentation Improvements
**Effort:** Low | **Impact:** Medium

- [ ] Add docstrings to all public functions
- [ ] Add usage examples to README
- [ ] Create API reference docs
- [ ] Add architecture diagrams

---

### 11. Performance Optimization
**Effort:** Medium | **Impact:** Low

- [ ] Profile large codebases (>500 files)
- [ ] Consider parallel file parsing
- [ ] Add caching for repeated analyses
- [ ] Optimize regex patterns in fallback parser

---

### 12. Error Handling
**Effort:** Medium | **Impact:** Medium

- [ ] Add custom exception classes
- [ ] Improve error messages
- [ ] Add recovery mechanisms for parse failures
- [ ] Log warnings for unsupported syntax

---

## 📋 Task Checklist

### Immediate (Week 1)
- [ ] Create `BaseLLMClient` abstract class
- [ ] Create `VerboseMixin` for common init
- [ ] Fix remaining test failures
- [ ] Update CHANGELOG

### Short-term (Week 2-3)
- [ ] Split `generators.py` into subpackage
- [ ] Split `parsers.py` into subpackage
- [ ] Refactor long functions
- [ ] Add missing type hints

### Long-term (Month 1-2)
- [ ] Increase test coverage to 80%
- [ ] Split `gherkin.py` into subpackage
- [ ] Add comprehensive documentation
- [ ] Performance optimization

---

## 📊 Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| Test Coverage | 31% | 80% |
| Duplicate Groups | 17 | 5 |
| Long Files (>500) | 3 | 0 |
| Long Functions (>50) | 21 | 5 |
| Large Classes (>20 methods) | 2 | 0 |
| mypy Errors | ? | 0 |

---

## 🔧 Development Commands

```bash
# Run tests with coverage
make test-cov

# Run linter
make lint

# Run type checker
mypy code2logic --strict

# Find duplicates
python examples/duplicate_detection.py .

# Analyze refactoring needs
python examples/refactor_suggestions.py . --no-llm
```

---

## Notes

- All changes should maintain backward compatibility
- Each refactoring should have corresponding tests
- Update documentation after each change
- Use feature branches for large refactorings
