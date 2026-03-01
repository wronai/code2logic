# Plan Refaktoryzacji code2flow

## Podsumowanie Zmian

Refaktoryzacja monolitycznego `flow.py` (1145 linii) w modularną paczkę Python.

## Nowa Struktura

```
debug/
├── code2flow/                  # Główna paczka
│   ├── __init__.py            # Eksportuje publiczne API
│   ├── __main__.py            # Entry point: python -m code2flow
│   ├── cli.py                 # CLI: code2flow komenda
│   ├── core/                  # Klasy bazowe i konfiguracja
│   │   ├── __init__.py
│   │   ├── config.py          # Config, ANALYSIS_MODES, NODE_COLORS
│   │   ├── models.py          # FlowNode, FlowEdge, DataFlow, AnalysisResult
│   │   └── analyzer.py        # ProjectAnalyzer - główny orchestrator
│   ├── extractors/            # Ekstraktory AST
│   │   ├── __init__.py
│   │   ├── cfg_extractor.py   # CFGExtractor - Control Flow Graph
│   │   ├── dfg_extractor.py   # DFGExtractor - Data Flow Graph
│   │   └── call_graph.py      # CallGraphExtractor
│   ├── exporters/             # Eksport do formatów
│   │   ├── __init__.py
│   │   └── base.py            # YAMLExporter, JSONExporter, MermaidExporter, LLMPromptExporter
│   ├── visualizers/           # Wizualizacja
│   │   ├── __init__.py
│   │   └── graph.py           # GraphVisualizer (NetworkX + matplotlib)
│   └── patterns/              # Detekcja wzorców
│       ├── __init__.py
│       └── detector.py        # PatternDetector (rekurencja, singleton, factory, state machine, strategy)
├── setup.py                   # Setup konfiguracja
├── pyproject.toml            # Nowoczesna konfiguracja pyproject
├── Makefile                  # Zaktualizowany Makefile
├── requirements.txt          # Zależności (networkx, matplotlib, pyyaml, numpy)
└── README.md                 # Zaktualizowana dokumentacja
```

## Kluczowe Decyzje Architektoniczne

### 1. Separacja Odpowiedzialności
- **core/**: Modele danych i główny analyzer
- **extractors/**: Logika parsowania AST (CFG, DFG, Call Graph)
- **exporters/**: Formaty wyjściowe (YAML, JSON, Mermaid)
- **visualizers/**: Renderowanie grafów
- **patterns/**: Detekcja wzorców behawioralnych

### 2. API Publiczne
```python
from code2flow import ProjectAnalyzer, Config
from code2flow.core.models import AnalysisResult
```

### 3. CLI
```bash
code2flow /path/to/project -m hybrid -o ./output --format yaml,json,mermaid,png
```

### 4. Konfiguracja
- `Config` dataclass z opcjami analizy
- `ANALYSIS_MODES` - dostępne tryby
- `NODE_COLORS` - kolory dla wizualizacji

## Porównanie z Narzędziami Referencyjnymi

| Cecha | code2flow | PyCG | Pyan | Angr | Code2Logic |
|-------|-----------|------|------|------|------------|
| CFG | ✓ | ✓ | ✗ | ✓ | ✓ |
| DFG | ✓ | ✗ | ✗ | ✓ | ✓ |
| Call Graph | ✓ | ✓ | ✓ | ✓ | ✓ |
| Wzorce | ✓ | ✗ | ✗ | ✗ | ✓ |
| LLM Output | ✓ | ✗ | ✗ | ✗ | ✓ |
| Modularność | ✓ | ✓ | ✓ | ✗ | ? |

## Przyszłe Rozszerzenia

### Priorytet Wysoki
1. [ ] Testy jednostkowe (pytest)
2. [ ] CI/CD pipeline (GitHub Actions)
3. [ ] Type hints (mypy compliant)
4. [ ] Obsługa dynamicznej analizy (sys.settrace)

### Priorytet Średni
5. [ ] Więcej formatów wyjściowych (Graphviz DOT, PlantUML)
6. [ ] Interaktywna wizualizacja (D3.js/Plotly)
7. [ ] Plugin system dla custom extractors
8. [ ] Cache analizy (pickle/JSON)

### Priorytet Niski
9. [ ] Wsparcie dla Cython
10. [ ] Analiza bytecode (dis)
11. [ ] Integracja z IDE (VS Code extension)
12. [ ] Web UI (Flask/FastAPI)

## Komendy Makefile

```bash
make install       # pip install -e .
make dev-install   # pip install -e ".[dev]"
make test          # pytest tests/
make lint          # flake8 + black --check
make format        # black code2flow/
make typecheck     # mypy code2flow/
make run           # code2flow ../python/stts_core
make build         # python setup.py sdist bdist_wheel
make clean         # rm -rf build/ dist/
make check         # lint + typecheck + test
```

## Instalacja

```bash
cd debug/
pip install -e .
code2flow /path/to/project -v
```

## Użycie Programowe

```python
from code2flow import ProjectAnalyzer, Config
from code2flow.exporters.base import YAMLExporter

config = Config(mode='hybrid', max_depth_enumeration=10)
analyzer = ProjectAnalyzer(config)
result = analyzer.analyze_project('/path/to/project')

exporter = YAMLExporter()
exporter.export(result, 'output.yaml')  # Default: skip empty values
exporter.export(result, 'output_full.yaml', include_defaults=True)  # Full output
```

## Eksport Danych (Compact by Default)

Wszystkie eksporty YAML/JSON domyślnie **ukrywają puste wartości**:
- `column: null` - pomijane
- `conditions: []` - pomijane  
- `data_flow: []` - pomijane
- `metadata: {}` - pomijane
- `returns: null` - pomijane

Aby pokazać wszystkie pola (np. dla debugowania):
```bash
code2flow /path/to/project --full
```

Programowo:
```python
result.to_dict()  # Default: False - skip empty values
result.to_dict(include_defaults=True)  # Include all fields
```

## Znane Problemy

1. **Dynamic analysis**: Wymaga implementacji `DynamicTracer` w pełni
2. **Cross-file resolution**: Może nie rozwiązać wszystkich importów
3. **Complex control flow**: Np. async/await, generators - uproszczona obsługa
4. **Performance**: Duże projekty (>10k LOC) mogą być wolne

## Konwencje Kodu

- **PEP 8** z line-length=100
- **Type hints** dla wszystkich funkcji publicznych
- **Docstrings** Google style
- **Black** do formatowania
- **isort** do importów (opcjonalnie)

## Status: ✅ Ukończone

- [x] Struktura katalogów
- [x] Moduły core/
- [x] Moduły extractors/
- [x] Moduły exporters/
- [x] Moduły visualizers/
- [x] Moduły patterns/
- [x] CLI
- [x] setup.py
- [x] pyproject.toml
- [x] Makefile
- [x] requirements.txt
- [ ] Tests (do zrobienia)
- [ ] Dokumentacja API (do zrobienia)
