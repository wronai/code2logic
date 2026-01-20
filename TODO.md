# Code2Logic - Refactoring Plan

## Active Task List (Execution Order)

- [x] Add `toon` support to CLI (`code2logic ... -f toon`)
- [x] Fix TOON tabular headers (`{path,lang,lines}`) and delimiter/quoting in TOON parser
- [x] Fix public API collisions in `code2logic/__init__.py` (`analyze_quality`, `reproduce_project`, `ReproductionResult`)
- [x] Update CLI docs to include TOON and clarify `python -m code2logic` vs `code2logic` entrypoint
- [x] **Fix critical parser bug**: TreeSitterParser._text() byte vs char offset mismatch
- [x] **Fix truncated identifiers**: Function/class names now extracted correctly
- [x] **Fix corrupted signatures**: Parameter parsing uses proper byte slicing
- [x] **Fix import deduplication**: Remove `module.module` duplicates
- [x] **Add docstring truncation**: First sentence or 80 chars for efficiency
- [x] **Add parser integrity tests**: 15 new test cases in `tests/test_parser_integrity.py`
- [x] **YAML compact format**: Short keys (p, l, i, e, c, f, n, d, m) with header legend
- [x] **YAML self removal**: Method signatures no longer include 'self' parameter
- [x] **YAML import dedup**: `typing.Dict, typing.List` → `typing.{Dict,List}`
- [x] **YAML schema update**: Validation supports both full and compact keys
- [x] **Add YAML compact tests**: 9 new test cases in `tests/test_yaml_compact.py`
- [x] **Create shared_utils.py**: Common utility functions (compact_imports, abbreviate_type, build_signature, truncate_docstring)
- [x] **Add shared_utils tests**: 47 test cases in `tests/test_shared_utils.py`
- [x] **TOON remove self**: Method signatures no longer include 'self' parameter
- [x] **TOON ultra-compact**: New `generate_ultra_compact()` method - 71% size reduction (78KB → 22KB)
- [x] **LogicML optimization**: Truncated docstrings, grouped imports, removed self from signatures
- [ ] Add a TOON round-trip sanity check (generate TOON -> parse -> validate key structure)
- [ ] Review TOON reproduction prompt quality (benchmark `toon` vs `yaml/json` after spec fixes)
- [x] Run full smoke-test (279 tests pass):
  - [x] `python -m code2logic code2logic/ -f toon`
  - [x] `python examples/02_refactoring.py`
  - [x] `python examples/04_project.py tests/samples/ --no-llm`
  - [x] `python -m pytest`

---

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
- [x] Create `BaseLLMClient` abstract class (Done: llm_clients.py)
- [x] Create `VerboseMixin` for common init (Done: metrics.py)
- [x] Consolidate examples (19 → 6 files)
- [x] Add advanced metrics system (metrics.py)
- [x] Add refactoring utilities (refactor.py)
- [x] Add universal reproduction (universal.py)
- [x] Add project reproduction (project_reproducer.py)
- [x] Add format benchmarks (08_format_benchmark.py)
- [x] Add async multi-provider LLM (09_async_benchmark.py)
- [x] Add function-level reproduction (10_function_reproduction.py)
- [x] Add token-aware benchmarking (11_token_benchmark.py)
- [x] Add JSON generator for format comparison
- [x] Fix dataclass detection in parser
- [x] Add class properties to YAML output
- [ ] Fix remaining test failures
- [x] Implement `code2logic llm` management commands (provider/model/key/priority)
- [x] Update CHANGELOG

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

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Test Coverage | 31% | 80% | 🟡 |
| Duplicate Groups | 17 → 18 | 5 | 🟡 |
| Long Files (>500) | 3 | 0 | 🟡 |
| Long Functions (>50) | 21 → 19 | 5 | 🟡 |
| Large Classes (>20 methods) | 2 | 0 | 🟡 |
| Example Files | 19 → 11 | 6 | 🟡 |
| New Modules Added | 0 → 12 | - | ✅ |
| Reproduction Score (YAML) | 74.5% | 80% | 🟡 |
| Reproduction Score (LogicML) | 65.9% | 80% | 🟡 |
| Token Efficiency (Markdown) | 43.9 | 50 | 🟡 |
| LogicML Compression | 0.42x | - | ✅ |
| LogicML Repeatability | 56.9% | 70% | 🟡 |
| YAML Repeatability | 41.0% | 70% | 🟡 |
| YAML vs JSON Token Savings | 44.1% | - | ✅ |
| Syntax OK (YAML/LogicML) | 100% | 100% | ✅ |
| Total Tests | 40 | 50 | 🟡 |
| LogicML Success Rate | 100% | 100% | ✅ |

### New Modules Added
- `llm_clients.py` - Unified LLM client interface
- `reproduction.py` - Code reproduction utilities
- `code_review.py` - Code review functions
- `benchmark.py` - Reproduction benchmarking
- `metrics.py` - Advanced quality metrics
- `refactor.py` - Refactoring utilities
- `logicml.py` - ⭐ LogicML format generator (best compression 0.42x)
- `prompts.py` - Optimized prompt templates for reproduction
- `universal.py` - Universal code representation
- `project_reproducer.py` - Multi-file reproduction
- `adaptive.py` - Adaptive format selection
- `file_formats.py` - File-specific format generators
- `markdown_format.py` - Hybrid Markdown generator

### New Example Scripts
- `08_format_benchmark.py` - Format comparison benchmark
- `09_async_benchmark.py` - Async multi-provider benchmark
- `10_function_reproduction.py` - Function-level reproduction
- `11_token_benchmark.py` - Token-aware benchmarking

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




celem projektu jest możliwośc wygenerowania pośredniego formatu, ktory będzie przechowywał logikę w taki sposob, by reprodukcja
pozwalała na zachowanie bazowych właściwości kodu, w taki sposob, aby działałpoprawnie, aby możliwe było odtworzenie tysięcy plików z zzachowaniem funkcjonalności, miarą efektywnosci jest ilość bajtów potrzebnych  do rpzechowania logiki w stosunku do bajtów zajmowanego kodu, czym mniejszy jest pliki reprezntuający logikę w stosunku do wielkości pliku wygenerowanego na jego podstawie tym bardziej efektywne  jest przechowywanie  logiki, 
trzeba jednak sprawdzać czy model LLM jest w stanie sobie z tym poradzić, dlatego trzeba też w oparciu o używany LLM do preprodukcji uużyć innej formy, stworz rozwiązanie, ktore będzie badało możliwości LLM i dopasowywało format i typ przechowania logiki aby unikknąć problemów przy niespojnosci regeneracji kodu z logiki
dodatkowo sprawdz jakie optymalizacje można zastosować, aby przy mniejszych modelach generować mniejsze fragmenty  logiki zamiast wszystko na raz z uwagi na ograniczenia LLM, ale rob to automatycznie na bazie danych LLM i logiki  rpzechowywanej w pliku

kontynuuj, dodaj więcej przykladow  kodu, do testow reprodukcji, aby reprodukowac z dowolnego języka do dowolnego języka, rowniez z językow DSL jak SQL, przetestuj i popraw jakość reprodukcji code2logic, sparwdz czy są inne  dodatkowe biblitoeki, ktore mogą pomóc w  lepszej regenracji, czy są jakieś lepsze biblitoeki, ktore ułatwiają reprodukcje kodu niezaleznie od jezyka programowania

prztetsuj z benchmarkiem wszystkie testowe przypadki i kontynuuj improvement, do uzysklania lepszych rezultatow dla kazdego przykladu
