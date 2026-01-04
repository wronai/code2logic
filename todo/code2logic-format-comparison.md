# Code2Logic Format Comparison: TOON vs YAML

**Analysis Date:** 2026-01-04  
**Project:** code2logic (50 files, 18,989 lines)

---

## File Size Comparison

| Metric | TOON | YAML | Difference |
|--------|------|------|------------|
| **Bytes** | 77,776 | 86,321 | TOON 10% smaller |
| **Lines** | 1,216 | 2,639 | TOON 54% fewer lines |
| **Tokens (est.)** | ~19,400 | ~21,600 | TOON ~10% fewer |

**Winner:** TOON (ale różnica mniejsza niż oczekiwana)

---

## Szczegółowa Analiza Formatów

### TOON Format - Mocne Strony

1. **Kompaktowa struktura nagłówka:**
```
modules[50]{path,lang,lines}:
  llm_profiler.py,python,491
  config.py,python,168
```

2. **Tabularyczna definicja metod:**
```
methods[9]{name,sig,decorators,async,lines}:
  __init__,"(self;client;verbose:bool)->None",-,false,12
  run_profile,"(self;quick:bool)->LLMProfile",-,false,53
```

3. **Separacja szczegółów od indeksu:**
```
module_details:
  llm_profiler.py:
    imports[20]: json,os,time...
```

### TOON Format - Problemy do Poprawy

| Problem | Przykład | Rekomendacja |
|---------|----------|--------------|
| **Redundantne `self` w sygnaturach** | `(self;client;verbose:bool)` | Usunąć `self` - domyślne dla metod |
| **Pełne sygnatury typów** | `->Dict[str, Any]` | Skrócić: `->Dict` lub `->D[s,A]` |
| **Powtórzone info o liczbie linii** | `lines: 491` + `lines` w metodach | Usunąć z metod lub z nagłówka |
| **Zbędne `-` dla pustych wartości** | `decorators: -` | Pominąć pole jeśli puste |
| **Pełne docstringi** | Cały pierwszy akapit | Limit 60 znaków |

### YAML Format - Mocne Strony

1. **Czytelność dla człowieka:**
```yaml
- name: __init__
  signature: (self,client,verbose:bool)
  intent: Initialize profiler.
```

2. **Standardowy format - łatwy do parsowania**

3. **Hierarchiczna struktura klas/metod**

### YAML Format - Problemy do Poprawy

| Problem | Przykład | Rekomendacja |
|---------|----------|--------------|
| **Nadmiarowe importy** | `typing` + `typing.Dict` + `typing.List` | Tylko konkretne: `typing.{Dict,List}` |
| **Powtórzenia `language: python`** | Każdy moduł ma to pole | Zdefiniować raz globalnie |
| **Rozwlekłe definicje** | 6 linii na metodę | Tryb inline dla prostych |
| **Brak liczby linii metod** | Nie wiadomo co jest duże | Dodać `lines` |
| **Puste `bases: []`** | Każda klasa | Pominąć jeśli puste |

---

## Proponowane Optymalizacje

### 1. TOON v2 - Ultra-Compact

```toon
# code2logic | 50f 18989L | py:50
# Klucze: p=path l=lines i=imports e=exports c=classes f=functions m=methods

M[50]:  # modules
llm_profiler.py,491
config.py,168
file_formats.py,279

D:  # details
llm_profiler.py:
  i: json,os,time,hashlib,dataclasses,datetime,typing
  e: load_profiles,save_profile,get_profile,LLMProfiler,AdaptiveChunker
  c:
    LLMProfiler: "Profile LLM capabilities"
      m: __init__(client,verbose=F)->None|12
         run_profile(quick=F)->LLMProfile|53
         _test_reproduction(name:s,code:s)->ProfileTestResult|48
    AdaptiveChunker: "Adaptive chunking based on LLM profile"
      m: __init__(profile?)->None|11
         get_optimal_settings()->Dict|8
  f:
    _get_profiles_path()->Path|5  # path to profiles
    load_profiles()->Dict|14  # Load all saved profiles
```

**Oszczędności:**
- Usunięcie `self` z sygnatur: ~500 bajtów
- Skrócenie typów (`str`→`s`, `bool`→`b`, `False`→`F`): ~1000 bajtów
- Usunięcie pustych pól: ~800 bajtów
- Inline docstringi jako komentarze: ~2000 bajtów
- **Szacowana redukcja: 30-40%**

### 2. YAML v2 - Optimized

```yaml
# code2logic | 50 files | 18989 lines | python
defaults:
  language: python
  bases: []

modules:
- p: llm_profiler.py  # 491 lines
  i: [json, os, time, hashlib, dataclasses, datetime, typing]
  e: [load_profiles, save_profile, get_profile, LLMProfiler, AdaptiveChunker]
  c:
  - n: LLMProfiler
    d: "Profile LLM capabilities"
    m:
    - __init__(client, verbose=False) -> None  # 12L
    - run_profile(quick=False) -> LLMProfile  # 53L | Run full profiling
    - _test_reproduction(name, code) -> ProfileTestResult  # 48L
  - n: AdaptiveChunker
    d: "Adaptive chunking"
    m:
    - __init__(profile=None)  # 11L
    - get_optimal_settings() -> Dict  # 8L
  f:
  - _get_profiles_path() -> Path  # 5L | path to profiles
  - load_profiles() -> Dict  # 14L | Load all saved profiles
```

**Oszczędności:**
- Krótkie klucze (`path`→`p`, `name`→`n`): ~3000 bajtów
- Inline sygnatury: ~5000 bajtów
- Usunięcie domyślnych wartości: ~2000 bajtów
- **Szacowana redukcja: 45-55%**

---

## Konkretne Zmiany w Kodzie

### Dla TOON (`toon_format.py`)

```python
# PRZED
def _generate_methods(self, methods, detail, indent):
    lines.append(f'{ind}methods[{len(methods)}]{{name,sig,decorators,async,lines}}:')
    for m in methods:
        sig = self._build_signature(m)  # Zawiera self
        lines.append(f'{ind}  {m.name},"{sig}",{deco},{async_},{m.lines}')

# PO
def _generate_methods(self, methods, detail, indent):
    lines.append(f'{ind}m[{len(methods)}]:')  # Krótszy nagłówek
    for m in methods:
        sig = self._build_signature_compact(m)  # Bez self, skrócone typy
        intent_short = (m.intent or "")[:40]
        lines.append(f'{ind}  {m.name}({sig})->{m.return_type_short}|{m.lines}  # {intent_short}')

def _build_signature_compact(self, func):
    """Build signature without self, with type abbreviations."""
    params = []
    for p in func.params:
        if p.name == 'self':
            continue
        ptype = self._abbreviate_type(p.type) if p.type else ''
        default = f'={self._abbreviate_default(p.default)}' if p.default else ''
        params.append(f'{p.name}{":"+ptype if ptype else ""}{default}')
    return ','.join(params)

def _abbreviate_type(self, t):
    """Abbreviate common types."""
    abbrevs = {
        'str': 's', 'int': 'i', 'bool': 'b', 'float': 'f',
        'None': 'N', 'Any': 'A', 'List': 'L', 'Dict': 'D',
        'Optional': '?', 'Tuple': 'T', 'Set': 'S',
    }
    for full, short in abbrevs.items():
        t = t.replace(full, short)
    return t
```

### Dla YAML (`generators.py`)

```python
# PRZED
def generate(self, project, detail_level='standard'):
    for module in project.modules:
        mod_data = {
            'path': module.path,
            'language': module.language,
            'lines': module.lines,
            'imports': module.imports,
            # ... pełna struktura
        }

# PO  
def generate(self, project, detail_level='standard'):
    # Globalny header
    output = {
        'project': project.name,
        'defaults': {'language': 'python', 'bases': []},
        'modules': []
    }
    
    for module in project.modules:
        mod_data = {
            'p': module.path,  # Krótkie klucze
            'i': self._compact_imports(module.imports),
            'e': module.exports,
        }
        # Pomijamy language jeśli == default
        if module.language != 'python':
            mod_data['lang'] = module.language
        
        # Klasy w formacie kompaktowym
        if module.classes:
            mod_data['c'] = [self._compact_class(c) for c in module.classes]
        
        output['modules'].append(mod_data)

def _compact_imports(self, imports):
    """Deduplicate and compact imports."""
    # typing.Dict, typing.List -> typing.{Dict,List}
    modules = {}
    for imp in imports:
        if '.' in imp:
            mod, name = imp.rsplit('.', 1)
            modules.setdefault(mod, []).append(name)
        else:
            modules[imp] = []
    
    result = []
    for mod, names in modules.items():
        if names:
            result.append(f"{mod}.{{{','.join(names)}}}")
        else:
            result.append(mod)
    return result

def _compact_class(self, cls):
    """Generate compact class representation."""
    return {
        'n': cls.name,
        'd': (cls.docstring or '')[:60],  # Truncated
        'm': [self._inline_method(m) for m in cls.methods]
    }

def _inline_method(self, method):
    """Generate inline method: name(params) -> return  # lines | intent"""
    params = self._compact_params(method.params)
    ret = method.return_type or ''
    intent = (method.intent or '')[:30]
    return f"{method.name}({params}){' -> '+ret if ret else ''}  # {method.lines}L{' | '+intent if intent else ''}"
```

---

## Test Cases dla Walidacji

```python
def test_toon_no_self_in_signatures():
    """Verify self is removed from method signatures."""
    toon = generate_toon(sample_project)
    assert 'self;' not in toon
    assert 'self,' not in toon
    assert '(self)' not in toon

def test_yaml_compact_imports():
    """Verify imports are deduplicated."""
    yaml_out = generate_yaml(sample_project)
    data = yaml.safe_load(yaml_out)
    
    imports = data['modules'][0]['i']
    # Should not have both 'typing' and 'typing.Dict'
    typing_entries = [i for i in imports if i.startswith('typing')]
    assert len(typing_entries) <= 1

def test_toon_size_reduction():
    """Verify TOON v2 is at least 30% smaller."""
    toon_v1 = generate_toon_v1(large_project)
    toon_v2 = generate_toon_v2(large_project)
    
    reduction = 1 - len(toon_v2) / len(toon_v1)
    assert reduction >= 0.30

def test_yaml_short_keys():
    """Verify YAML uses short keys."""
    yaml_out = generate_yaml_compact(sample_project)
    data = yaml.safe_load(yaml_out)
    
    mod = data['modules'][0]
    assert 'p' in mod  # path
    assert 'path' not in mod
    assert 'i' in mod  # imports
    assert 'imports' not in mod

def test_docstring_truncation():
    """Verify docstrings are truncated to 60 chars."""
    toon = generate_toon(project_with_long_docs)
    
    for line in toon.split('\n'):
        if 'doc:' in line:
            doc_part = line.split('doc:')[1].strip()
            assert len(doc_part) <= 65  # 60 + quotes + margin

def test_empty_fields_omitted():
    """Verify empty bases/decorators are not included."""
    yaml_out = generate_yaml_compact(sample_project)
    
    assert 'bases: []' not in yaml_out
    assert 'decorators: []' not in yaml_out
    assert 'decorators: -' not in yaml_out
```

---

## Priorytet Implementacji

| Priorytet | Zmiana | Format | Oszczędność | Wysiłek |
|-----------|--------|--------|-------------|---------|
| **P0** | Usunięcie `self` | TOON/YAML | 5% | Niski |
| **P0** | Skrócenie kluczy YAML | YAML | 15% | Niski |
| **P1** | Deduplikacja importów | Oba | 8% | Średni |
| **P1** | Inline metody | YAML | 20% | Średni |
| **P2** | Skróty typów | TOON | 5% | Niski |
| **P2** | Pomijanie pustych pól | Oba | 5% | Niski |
| **P3** | Truncate docstrings | Oba | 10% | Niski |

---

## Szacowany Wynik Po Optymalizacji

| Format | Obecnie | Po Optymalizacji | Redukcja |
|--------|---------|------------------|----------|
| TOON | 77.8 KB | ~50 KB | 35% |
| YAML | 86.3 KB | ~45 KB | 48% |

**YAML po optymalizacji może być mniejszy niż TOON**, zachowując lepszą czytelność!
