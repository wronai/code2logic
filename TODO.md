# Code2Logic — Plan refaktoryzacji

> Podstawa: Benchmarki 2026-02-25, 20 plików, model: `arcee-ai/trinity-large-preview`

## Aktualne wyniki benchmarków

| Format | Wynik | Tokeny | Efektywność (p/kT) |
|--------|------:|-------:|---------:|
| **toon** | **63,8%** | 17 875 | **3,57** |
| json | 62,9% | 104 914 | 0,60 |
| markdown | 62,5% | 36 851 | 1,70 |
| yaml | 62,4% | 68 651 | 0,91 |
| logicml | 60,4% | ~30 000 | ~2,01 |
| csv | 53,0% | 80 779 | 0,66 |
| function.toon | 49,3% | 29 271 | 1,68 |
| gherkin | 38,6% | ~25 000 | ~1,54 |

Behavioral benchmark: **85,7%** (6/7 funkcji).

---

## Zrealizowane (P0 + P1.4)

- [x] **P0.1** function.toon — kontekst strukturalny (`--function-logic-context none|minimal|full`)
- [x] **P0.2** LogicML — typowane sygnatury (`level=compact|typed|full`, domyślnie: `typed`)
- [x] **P0.3** AST-based scoring w `metrics.py` (Python `ast` + regex fallback, ratio-based)
- [x] **P1.4** TOON-Hybrid (`generate_hybrid()` + `--hybrid` w CLI)
- [x] Naprawiono agregację benchmarków (zero scores, merge-score, failure_rate)
- [x] Przepisano prompty reprodukcji (gherkin, function.toon, csv, markdown, logicml)
- [x] Usunięto martwy kod (`llm_clients_new.py`)
- [x] Ujednolicono sygnatury `.generate()` we wszystkich generatorach

## Pozostałe do zrobienia

### P1 — Wysokie

- [ ] **P1.5** Ujednolicenie interfejsu generatorów — `base.py` z adapter pattern dla backward compatibility
- [ ] **P1.6** Auto-chunking — automatyczne dzielenie przy przekroczeniu limitu kontekstu modelu (`--auto-chunk`)

### P2 — Średnie

- [ ] **P2.7** YAML ultra-compact — cel 40-50% redukcji przez skróty z `meta.legend`
- [ ] **P2.8** Metryki efektywności p/kT (punkty/1000 tokenów) w raportach
- [ ] **P2.9** Benchmarki na silniejszych modelach (Claude 3.5 Sonnet, GPT-4o)

### P3 — Długoterminowe

- [ ] AST scoring dla JavaScript (tree-sitter lub esprima)
- [ ] Zwiększenie test coverage do 60%+
- [ ] TOON round-trip sanity check (generate → parse → validate)

---

## Oczekiwane wyniki po refaktoryzacji

| Format | Obecny | Docelowy | Główna zmiana |
|--------|-------:|--------:|---------------|
| toon (Hybrid) | 63,8% | 68-72% | TOON-Hybrid format |
| function.toon | 49,3% | 57-61% | Kontekst strukturalny |
| logicml | 60,4% | 63-65% | Typy w sygnaturach |
| yaml | 62,4% | 64-66% | Ultra-compact opcja |
| json | 62,9% | 62-63% | Bez zmian (baseline) |

**Cel:** TOON-Hybrid 70%+ przy ~25k tokenów.

---

## Co NIE wymaga zmiany

- **Cztery moduły reprodukcji** (`SpecReproducer`, `ProjectReproducer`, `ChunkedReproducer`, `UniversalReproducer`) — różna odpowiedzialność, scalenie złamie SRP
- **`file_formats.py`** i **`prompts.py`** — TOON widzi tylko sygnatury, ukrywając 350+ linii logiki
- **`llm.py` ↔ `llm_clients.py`** — brak circular dependencies (LLM błędnie diagnozował)

---

## Komendy deweloperskie

```bash
make test          # 286 testów
make test-cov      # Z pokryciem
make benchmark     # Pełne benchmarki (wymaga OPENROUTER_API_KEY)
make lint          # Linting
```
