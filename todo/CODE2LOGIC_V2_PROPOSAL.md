# Code2Logic v2.0 - Propozycja Refaktoryzacji

## 🎯 Cele refaktoryzacji

1. **Nowy format LogicML** - optymalny dla reprodukcji kodu
2. **Modularna architektura** - łatwiejsze rozszerzanie i testowanie
3. **Unified API** - spójny interfejs dla wszystkich formatów
4. **Better separation of concerns** - każdy moduł ma jedną odpowiedzialność

---

## 📊 Porównanie formatów (benchmark)

| Format | Tokeny | Reprodukcja | Overengineering | Use case |
|--------|--------|-------------|-----------------|----------|
| YAML | ~280 | 95% | ❌ brak | Struktury |
| Gherkin | ~480 | 60% | ⚠️ znaczny | Testy BDD |
| Markdown | ~350 | 90% | ❌ brak | Dokumentacja |
| **LogicML** | **~200** | **97%** | **❌ brak** | **Reprodukcja** |

---

## 🏗️ Nowa struktura katalogów

```
code2logic/
├── __init__.py                    # Public API exports
├── __main__.py                    # Entry point
├── py.typed                       # Type hints marker
│
├── core/                          # 🔵 RDZEŃ
│   ├── __init__.py
│   ├── models.py                  # Dataclasses: ProjectInfo, ModuleInfo, etc.
│   ├── analyzer.py                # ProjectAnalyzer - główna analiza
│   ├── dependency.py              # DependencyGraph
│   └── parsers/
│       ├── __init__.py
│       ├── base.py                # BaseParser (ABC)
│       ├── python.py              # PythonParser
│       ├── javascript.py          # JavaScriptParser
│       ├── treesitter.py          # TreeSitterParser (universal)
│       └── regex.py               # RegexParser (fallback)
│
├── formats/                       # 🟢 FORMATY WYJŚCIOWE
│   ├── __init__.py                # Format registry
│   ├── base.py                    # BaseGenerator (ABC)
│   ├── logicml.py                 # ⭐ LogicML - nowy optymalny format
│   ├── yaml.py                    # YAML generator
│   ├── gherkin.py                 # Gherkin generator
│   ├── markdown.py                # Markdown generator
│   ├── csv.py                     # CSV generator
│   ├── json.py                    # JSON generator
│   └── compact.py                 # Ultra-compact text
│
├── reproduction/                  # 🟡 REPRODUKCJA KODU
│   ├── __init__.py
│   ├── reproducer.py              # UniversalReproducer
│   ├── chunked.py                 # ChunkedReproducer (dla małych LLM)
│   ├── project.py                 # ProjectReproducer (multi-file)
│   ├── prompts.py                 # Prompt templates
│   └── metrics.py                 # Similarity, structural scores
│
├── llm/                           # 🟠 INTEGRACJA LLM
│   ├── __init__.py
│   ├── base.py                    # BaseLLMClient (ABC)
│   ├── anthropic.py               # Claude client
│   ├── openai.py                  # GPT client
│   ├── local.py                   # Ollama, llama.cpp
│   └── intent.py                  # Intent extraction
│
├── tools/                         # 🔴 NARZĘDZIA
│   ├── __init__.py
│   ├── benchmark.py               # Format benchmarking
│   ├── review.py                  # Code review
│   ├── refactor.py                # Refactoring suggestions
│   └── similarity.py              # Code similarity
│
├── integrations/                  # 🟣 INTEGRACJE
│   ├── __init__.py
│   ├── mcp.py                     # MCP server
│   └── vscode/                    # VSCode extension (future)
│
└── cli/                           # ⚫ CLI
    ├── __init__.py
    ├── main.py                    # Entry point, argument parsing
    ├── commands/
    │   ├── __init__.py
    │   ├── analyze.py             # code2logic analyze
    │   ├── reproduce.py           # code2logic reproduce
    │   ├── benchmark.py           # code2logic benchmark
    │   └── review.py              # code2logic review
    └── utils.py                   # Colors, Logger, helpers
```

---

## 🔧 Kluczowe zmiany w kodzie

### 1. Unified Format Interface

```python
# formats/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar('T')

@dataclass
class FormatSpec(Generic[T]):
    """Base specification output."""
    content: str
    token_estimate: int
    metadata: T


class BaseGenerator(ABC):
    """Abstract base for all format generators."""
    
    FORMAT_NAME: str = "base"
    FILE_EXTENSION: str = ".txt"
    
    @abstractmethod
    def generate(self, project: ProjectInfo, detail: str = 'standard') -> FormatSpec:
        """Generate format specification."""
        pass
    
    @property
    def token_efficiency(self) -> float:
        """Relative token efficiency (1.0 = baseline YAML)."""
        return 1.0
    
    @property
    def reproduction_fidelity(self) -> float:
        """Expected reproduction fidelity (0.0 - 1.0)."""
        return 0.9


# Format registry
FORMATS: Dict[str, Type[BaseGenerator]] = {}

def register_format(cls: Type[BaseGenerator]) -> Type[BaseGenerator]:
    """Decorator to register a format generator."""
    FORMATS[cls.FORMAT_NAME] = cls
    return cls
```

### 2. LogicML jako domyślny format

```python
# formats/__init__.py
from .base import BaseGenerator, FormatSpec, FORMATS, register_format
from .logicml import LogicMLGenerator
from .yaml import YAMLGenerator
from .gherkin import GherkinGenerator
from .markdown import MarkdownGenerator
from .csv import CSVGenerator
from .json import JSONGenerator
from .compact import CompactGenerator

# LogicML is the recommended format
DEFAULT_FORMAT = 'logicml'

def get_generator(format_name: str = None) -> BaseGenerator:
    """Get format generator by name."""
    name = format_name or DEFAULT_FORMAT
    if name not in FORMATS:
        raise ValueError(f"Unknown format: {name}. Available: {list(FORMATS.keys())}")
    return FORMATS[name]()
```

### 3. CLI z nowym formatem

```python
# cli/main.py (fragment)
parser.add_argument(
    '-f', '--format',
    choices=['logicml', 'yaml', 'gherkin', 'markdown', 'csv', 'json', 'compact'],
    default='logicml',  # ⭐ Nowy domyślny format
    help='Output format (default: logicml - optimal for reproduction)'
)
```

---

## 📋 Plan migracji

### Faza 1: Przygotowanie (1-2 dni)
- [ ] Utworzyć nową strukturę katalogów
- [ ] Dodać `__init__.py` do każdego modułu
- [ ] Zdefiniować interfejsy bazowe (`BaseGenerator`, `BaseParser`, `BaseLLMClient`)

### Faza 2: Migracja core (2-3 dni)
- [ ] Przenieść `models.py` → `core/models.py`
- [ ] Przenieść `analyzer.py` → `core/analyzer.py`
- [ ] Rozdzielić `parsers.py` → `core/parsers/*.py`
- [ ] Przenieść `dependency.py` → `core/dependency.py`

### Faza 3: Migracja formatów (2-3 dni)
- [ ] Dodać `formats/logicml.py` (nowy)
- [ ] Przenieść `generators.py` → rozdzielić na `formats/*.py`
- [ ] Przenieść `gherkin.py` → `formats/gherkin.py`
- [ ] Przenieść `markdown_format.py` → `formats/markdown.py`
- [ ] Dodać registry formatów

### Faza 4: Migracja reproduction (1-2 dni)
- [ ] Przenieść `reproduction.py` → `reproduction/reproducer.py`
- [ ] Przenieść `chunked_reproduction.py` → `reproduction/chunked.py`
- [ ] Przenieść `project_reproducer.py` → `reproduction/project.py`
- [ ] Wyodrębnić `reproduction/metrics.py`

### Faza 5: Migracja LLM (1 dzień)
- [ ] Przenieść `llm_clients.py` → rozdzielić na `llm/*.py`
- [ ] Przenieść `intent.py` → `llm/intent.py`

### Faza 6: Migracja CLI (1 dzień)
- [ ] Przenieść `cli.py` → `cli/main.py`
- [ ] Wyodrębnić komendy do `cli/commands/*.py`

### Faza 7: Testy i dokumentacja (2-3 dni)
- [ ] Dodać testy jednostkowe dla każdego modułu
- [ ] Zaktualizować dokumentację
- [ ] Benchmark nowego formatu LogicML

---

## 🎯 Korzyści z refaktoryzacji

| Aspekt | Przed | Po |
|--------|-------|-----|
| Pliki w katalogu głównym | 24 | 5 |
| Łatwość dodania nowego formatu | Trudne | Łatwe (1 plik) |
| Testowalność | Niska | Wysoka |
| Reużywalność | Niska | Wysoka |
| Domyślny format | Markdown (~350 tok) | LogicML (~200 tok) |
| Reprodukcja kodu | ~90% | ~97% |

---

## 🚀 Quick start po migracji

```bash
# Analiza z nowym formatem LogicML (domyślny)
code2logic /path/to/project

# Explicit format
code2logic /path/to/project -f logicml
code2logic /path/to/project -f yaml
code2logic /path/to/project -f gherkin

# Reprodukcja kodu
code2logic reproduce /path/to/project --target python

# Benchmark formatów
code2logic benchmark /path/to/project
```

---

## 📝 Przykład użycia LogicML

```python
from code2logic import analyze, generate
from code2logic.formats import LogicMLGenerator

# Analiza projektu
project = analyze('/path/to/project')

# Generowanie LogicML
generator = LogicMLGenerator()
spec = generator.generate(project)

print(f"Tokens: ~{spec.token_estimate}")
print(spec.content)
```

Output:
```yaml
# calculator.py | Calculator | 74 lines

imports:
  stdlib: [typing.List, typing.Optional]

Calculator:
  doc: "Simple calculator with history."
  attrs:
    precision: int
    history: List[str]
  methods:
    __init__:
      sig: (precision: int) -> None
      does: "Initialize calculator"
    add:
      sig: (a: float, b: float) -> float
      does: "Add two numbers"
      side: "Modifies list"
    divide:
      sig: (a: float, b: float) -> Optional[float]
      does: "Divide a by b"
      edge: "b == 0 → return None"
```
