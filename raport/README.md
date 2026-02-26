---
title: "LLM i limit kontekstu: Dlaczego JSON to ślepa uliczka i jak Code2Logic zmienia zasady gry"
date: 2026-02-25
author: Tom Sapletta
tags: [LLM, AI, Code2Logic, TOON, benchmark, context-window, optymalizacja]
categories: [AI, Narzędzia, Open Source]
excerpt: "Przeprowadziłem serię benchmarków porównujących formaty serializacji kodu dla modeli językowych. Wyniki? JSON zużywa 5x więcej tokenów niż TOON, a jakość rekonstrukcji nie jest proporcjonalna do rozmiaru. Oto dane i wnioski."
---

# LLM i limit kontekstu: Dlaczego JSON to ślepa uliczka i jak Code2Logic zmienia zasady gry

**Autor: Tom Sapletta**

<script type="module" src="./mermaid-init.js"></script>

Jeśli kiedykolwiek próbowałeś „nakarmić" model językowy całym repozytorium kodu — by poprosić o refaktoryzację, znalezienie błędu lub wygenerowanie dokumentacji — na pewno zderzyłeś się ze ścianą. Tą ścianą jest **limit okna kontekstowego** oraz zjawisko *lost in the middle*: model zapomina lub ignoruje informacje w środku długiego promptu.

Cześć, jestem Tom Sapletta. Od dłuższego czasu pracuję nad tym, jak zoptymalizować komunikację między kodem źródłowym a sztuczną inteligencją. Tak właśnie narodził się projekt **Code2Logic**.

> **Ważna nota metodyczna:** Wyniki benchmarków mierzą jakość *rekonstrukcji* kodu na podstawie specyfikacji — głównie zgodność struktury, sygnatur i semantyki tekstowej. **Wysoki wynik nie jest dowodem pełnej równoważności behawioralnej (runtime).** Pełną poprawność potwierdza dopiero uruchomienie testów.

---
## Dlaczego powstał Code2Logic?

Kiedy LLM analizuje kod, nie potrzebuje wszystkich średników, nawiasów, wcięć ani nadmiarowej struktury. Tradycyjne podejście polega na serializacji projektu do formatu JSON. Problem: JSON jest dla modeli językowych **„głośny"** — większość tokenów (za które płacimy i które marnują uwagę modelu) to nawiasy klamrowe, cudzysłowy i powtarzające się klucze.

Diagram przepływu: od kodu do specyfikacji i z powrotem (LLM):

Wizualna różnica:

```
┌─────────────────────────────────────┐   ┌───────────────────────────┐
│ JSON (~104 914 tokenów dla projektu)│   │  TOON (~17 875 tokenów)   │
├─────────────────────────────────────┤   ├───────────────────────────┤
│ {                                   │   │ classes:                  │
│   "User": {                         │   │   User                    │
│     "methods": [                    │   │     - get_email()         │
│       {                             │   │     - set_email(e)        │
│         "name": "get_email",        │   │                           │
│         "type": "string"            │   │                           │
│       }                             │   │                           │
│     ]                               │   │                           │
│   }                                 │   │                           │
│ }                                   │   │                           │
└─────────────────────────────────────┘   └───────────────────────────┘
```

Code2Logic wyekstrahuje **czystą logikę** z kodu i przekaże ją do modelu w maksymalnie skompresowanych formatach: autorskim **TOON**, **LogicML** lub zwięzłym **YAML compact**.

Diagram „jak Code2Logic zmienia architekturę przepływu danych”:

![Schemat przepływu Code2Logic](img_4.png)

---
## Rzeczywiste wyniki benchmarków (20 plików, model: arcee-ai/trinity-large-preview)

### Rozmiar plików dla tego samego projektu

| Format | Rozmiar | ~Tokenów | vs JSON |
|--------|--------:|--------:|--------|
| **TOON** (project) | 70 KB | 17 875 | **5,9x mniejszy** |
| YAML compact | 268 KB | 68 651 | 1,5x mniejszy |
| JSON | 410 KB | 104 914 | baseline |
| CSV | 316 KB | 80 779 | większy od YAML |
| Markdown | 144 KB | 36 851 | 2,8x mniejszy |
| function.toon | 114 KB | 29 271 | 3,6x mniejszy |

> Zredukowaliśmy objętość **prawie 6-krotnie**. Do kontekstu modelu możemy zmieścić 6x większy projekt, płacąc ułamek ceny.

---
### Project Benchmark — jakość reprodukcji kodu z całego projektu

Test ocenia, jak dobrze LLM odtwarza kod na podstawie specyfikacji w danym formacie (syntaktyka, struktura, semantyka):

Wykres porównawczy (wizualizacja):

![Project Benchmark – wykres](img.png)

| Format | Wynik | Syntax OK | Runs OK |
|--------|------:|----------:|--------:|
| **toon** | **63,8%** | 100% | 60% |
| json | 62,9% | 100% | 60% |
| markdown | 62,5% | 100% | 55% |
| yaml | 62,4% | 100% | 55% |
| logicml | 60,4% | 100% | 55% |
| csv | 53,0% | 100% | 40% |
| function.toon | 49,3% | 95% | 35% |
| gherkin | 38,6% | 95% | 30% |

**Kluczowe obserwacje:**

1. **TOON wygrywa project benchmark** mimo że zajmuje prawie 6x mniej tokenów niż JSON (który osiąga 62,9%). Mniejszy format = lepsza jakość.

2. **function.toon (49,3%) wypada gorzej niż project.toon (63,8%)** — paradoks wyjaśniony: format skupiony wyłącznie na logice funkcji traci kontekst klas i modułów, co utrudnia LLM rekonstrukcję pełnej struktury. Większy plik (114 KB vs 70 KB) nie pomaga, jeśli brakuje kontekstu strukturalnego.

3. **CSV (53%) i gherkin (38,6%)** — formaty nieoptymalne dla opisu kodu źródłowego. Ich składnia narzuca ramy koncepcyjne, które nie mapują się dobrze na struktury programistyczne.

4. **Syntax OK = 100% dla wszystkich głównych formatów** — LLM zawsze generuje syntaktycznie poprawny kod. Problem leży w semantyce i kompletności, nie w składni.

---
### Behavioral Benchmark — prawdziwa miara

Osobny test: **85,7% (6/7 funkcji zaliczonych, 1 pominięta)** przy testowaniu behawioralnej równoważności odtworzonego kodu. To jedyny test, który mierzy „czy kod działa tak samo", a nie tylko „czy wygląda podobnie".

---
## Jak działa Code2Logic w praktyce

### Instalacja i użycie

```bash
pip install code2logic

# Wygeneruj projekt w formacie TOON (najefektywniejszy)
code2logic ./ -f toon --compact --name project -o ./

# TOON z logiką funkcji (szczegółowy)
code2logic ./ -f toon --compact --no-repeat-module \
  --function-logic function.toon --with-schema --name project -o ./

# YAML compact (czytelny dla człowieka, dobry kompromis)
code2logic ./ -f yaml --compact --name project -o ./
```

### Integracja z Claude Code

Wszystko w jednym kroku:

```bash
# Krok 1: Wygeneruj manifest
code2logic ./ -f toon --compact --no-repeat-module --function-logic -o ./

# Krok 2: Przekaż do Claude
claude --dangerously-skip-permissions -p \
  "Zrób refaktoryzację projektu, użyj pliku project.functions.toon jako źródła prawdy"
```

Lub jako jeden pipeline:

```bash
printf '%s\n\n' "Zrób refaktoryzację projektu. Poniżej masz manifest w formacie TOON." \
  > /tmp/prompt.txt
code2logic ./ -f toon --compact --no-repeat-module --function-logic -o ./
cat ./project.functions.toon >> /tmp/prompt.txt
claude --dangerously-skip-permissions --file /tmp/prompt.txt
```

Start komendy (zrzut):

![Start komendy w Claude](img_1.png)

Wnioski wygenerowane przez Claude (zrzut):

![Wnioski Claude](img_2.png)

Szacunki / koszty (zrzut):

![Szacunki](img_3.png)

---
## Wnioski z eksperymentu refaktoryzacji

Przeprowadziłem pełną refaktoryzację projektu na podstawie manifestu TOON i zweryfikowałem twierdzenia modelu w praktyce. Co LLM trafnie wyłapał?

**✅ Trafne wskazania:**
- Martwy kod (`llm_clients_new.py`) — faktycznie usunięty
- Śmieciowe katalogi z wygenerowanym kodem — wyczyszczone
- Niespójne interfejsy generatorów (różne argumenty `.generate()`) — ujednolicone

**❌ Błędne wskazania:**
- „Scal moduły z małą liczbą funkcji" — `file_formats.py` ma 350 linii złożonej logiki, `prompts.py` 150 linii szablonów. TOON widzi tylko sygnatury, ukrywając rozmiar logiki.
- „Zmerguj cztery moduły reprodukcji" — każdy robi coś innego (`SpecReproducer`, `ProjectReproducer`, `ChunkedReproducer`, `UniversalReproducer`). Scalenie złamałoby zasadę SRP.
- „Circular dependencies w `llm.py`" — nieprawdziwe, wynikało z niepełnego kontekstu importów.

**Wniosek:** Manifest TOON świetnie wykrywa martwy kod i niespójne interfejsy. Gorzej radzi sobie z oceną faktycznej złożoności modułów (widzi tylko sygnatury) i relacji importów. Zawsze weryfikuj sugestie LLM przed wdrożeniem.

---
## Kiedy używać jakiego formatu?

| Scenariusz | Rekomendowany format | Dlaczego |
|-----------|---------------------|---------|
| Analiza całego projektu przez LLM | **TOON compact** | Najlepszy wynik/token |
| Refaktoryzacja konkretnych funkcji | **function.toon** | Pełna logika funkcji |
| Debugging + human review | **YAML compact** | Czytelny dla człowieka |
| RAG / baza wektorowa | **JSON** | Łatwy w parsowaniu |
| Duży projekt, mały kontekst | **TOON + chunking** | Chunked reproduction |

---
## Co dalej?

Obszary do natychmiastowej poprawy (dane z benchmarków):

1. **AST zamiast Regex** — obecny benchmark traci skuteczność przy JavaScript, Java i Rust (często wynik 0%). Przejście na AST-based scoring rozwiąże problem wielojęzyczności.

2. **Lepszy kontekst w function.toon** — dodanie minimalnego kontekstu klas/modułów podniesie wynik z 49% bliżej 63% przy zachowaniu kompaktowości.

3. **Hybrydowy format** — kombinacja project-level TOON (struktura) + selektywny function.toon (dla kluczowych modułów) powinna dać najlepszy stosunek jakości do tokenów.

4. **Benchmark na silniejszych modelach** — testy na `arcee-ai/trinity-large-preview` (darmowy model) zaniżają wyniki. Benchmark na Claude 3.5 Sonnet lub GPT-4o powinien pokazać 80%+ dla TOON.

---
## Podsumowanie

| Co zyskujesz z Code2Logic? | Wartość |
|---------------------------|---------|
| Redukcja tokenów (TOON vs JSON) | ~5,9x |
| Jakość reprodukcji (TOON) | 63,8% |
| Syntax OK | 100% |
| Behavioral equivalence | 85,7% |
| Czas analizy (20 plików) | ~37 min |

Code2Logic to nie tylko narzędzie do kompresji kodu. To zmiana architektury komunikacji między programistą, kodem i modelem językowym. **Format ma znaczenie** — a dane to potwierdzają.

---
## Linki i artefakty (dane źródłowe)

- **Raport końcowy benchmarków**
  - `../examples/output/BENCHMARK_REPORT.md`
- **Wyniki (JSON)**
  - `../examples/output/benchmark_format.json`
  - `../examples/output/benchmark_project.json`
  - `../examples/output/benchmark_token.json`
  - `../examples/output/benchmark_function.json`
  - `../examples/output/benchmark_behavioral.json`
- **Przykładowe specyfikacje projektu (te same dane w różnych formatach)**
  - `../examples/output/project.toon`
  - `../examples/output/project.yaml`
  - `../examples/output/project.json`
  - `../examples/output/project.md`
  - `../examples/output/project.csv`
  - `../examples/output/project.functions.toon`

**Projekt:** [github.com/wronai/code2logic](https://github.com/wronai/code2logic)  
**Autor:** Tom Sapletta  
**Licencja:** Open Source