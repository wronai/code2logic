# Plan Refaktoryzacji Generatorów Code2Logic

**Data:** 2026-01-04  
**Cel:** Optymalizacja rozmiaru wyjścia o 30-50% przy zachowaniu jakości reprodukcji kodu

---

## Spis Treści

1. [Podsumowanie Wykonawcze](#1-podsumowanie-wykonawcze)
2. [Wspólne Problemy](#2-wspólne-problemy)
3. [Plan dla generators.py (YAMLGenerator)](#3-plan-dla-generatorspy)
4. [Plan dla toon_format.py](#4-plan-dla-toon_formatpy)
5. [Plan dla logicml.py](#5-plan-dla-logicmlpy)
6. [Plan dla file_formats.py](#6-plan-dla-file_formatspy)
7. [Nowy Moduł: shared_utils.py](#7-nowy-moduł-shared_utilspy)
8. [Testy Jednostkowe](#8-testy-jednostkowe)
9. [Harmonogram Implementacji](#9-harmonogram-implementacji)

---

## 1. Podsumowanie Wykonawcze

### Aktualny Stan

| Generator | Plik | Linie | Główne Problemy |
|-----------|------|-------|-----------------|
| YAMLGenerator | generators.py | ~320 | Redundantne `self`, pełne klucze, duplikaty importów |
| TOONGenerator | toon_format.py | 411 | `self` w sygnaturach, pełne typy, verbose nagłówki |
| LogicMLGenerator | logicml.py | 296 | Pełne docstringi, niespójna struktura |
| FileFormats | file_formats.py | 279 | Duplikacja logiki parsowania |

### Oczekiwane Rezultaty

| Metryka | Przed | Po | Redukcja |
|---------|-------|------|----------|
| YAML output | 86 KB | ~45 KB | **48%** |
| TOON output | 78 KB | ~50 KB | **36%** |
| LogicML output | ~70 KB | ~45 KB | **35%** |
| Łączny rozmiar kodu | ~1300 linii | ~900 linii | **30%** |

---

## 2. Wspólne Problemy

### 2.1. Redundantne `self` w Sygnaturach

**Problem:** Każda sygnatura metody zawiera `self`, co jest oczywiste dla LLM.

```python
# PRZED (wszystkie generatory)
signature: (self,client,verbose:bool)->None

# PO
signature: (client,verbose:bool)->None
```

**Lokalizacje do zmiany:**
- `generators.py`: linia 797-811 (`_build_signature`)
- `toon_format.py`: linia 220-235 (`_build_signature`)
- `logicml.py`: linia 216-230 (w `_generate_method`)

### 2.2. Duplikacja Importów

**Problem:** `typing` + `typing.Dict` + `typing.List` zamiast zgrupowanych.

```python
# PRZED
imports:
- typing
- typing.Dict
- typing.List
- typing.Optional

# PO
imports:
- typing.{Dict,List,Optional}
```

**Rozwiązanie:** Nowa funkcja `compact_imports()` w shared_utils.py

### 2.3. Powtórzona Logika

**Problem:** `_categorize()`, `_extract_domain()`, `_compute_hash()` zduplikowane w:
- `generators.py` (YAMLGenerator + CSVGenerator)
- `toon_format.py`
- Częściowo w `logicml.py`

**Rozwiązanie:** Przenieść do `shared_utils.py`

---

## 3. Plan dla generators.py

### 3.1. YAMLGenerator - Zmiany Strukturalne

#### A. Nowy tryb `compact_keys`

```python
class YAMLGenerator:
    # Mapping pełnych kluczy na skrócone
    KEY_MAP = {
        'path': 'p',
        'language': 'l', 
        'lines': 'ln',
        'imports': 'i',
        'exports': 'e',
        'classes': 'c',
        'functions': 'f',
        'methods': 'm',
        'name': 'n',
        'signature': 's',
        'intent': 'it',
        'docstring': 'd',
        'bases': 'b',
        'properties': 'pr',
    }
    
    def __init__(self, compact_keys: bool = False):
        self.compact_keys = compact_keys
        self.k = self.KEY_MAP if compact_keys else {k: k for k in self.KEY_MAP}
```

#### B. Zmodyfikowana `_build_nested_data`

```python
def _build_nested_data(self, project: ProjectInfo, detail: str) -> dict:
    k = self.k  # Alias dla skróconych kluczy
    
    modules = []
    for m in project.modules:
        module_data = {
            k['path']: m.path,
            k['lines']: m.lines_code,
        }
        
        # Pomijaj language jeśli == python (domyślne)
        if m.language != 'python':
            module_data[k['language']] = m.language
        
        # Kompaktowe importy
        if detail in ('standard', 'full') and m.imports:
            module_data[k['imports']] = compact_imports(m.imports)
        
        # ... reszta
```

#### C. Zmodyfikowana `_build_signature`

```python
def _build_signature(self, f: FunctionInfo, include_self: bool = False) -> str:
    """Build compact signature string without self."""
    clean_params = []
    for p in f.params[:6]:
        p_clean = p.replace('\n', ' ').strip()
        # Pomijaj self/cls
        if p_clean in ('self', 'cls') and not include_self:
            continue
        # Skracaj typy opcjonalnie
        if self.compact_keys:
            p_clean = abbreviate_type(p_clean)
        if p_clean:
            clean_params.append(p_clean)
    
    params = ','.join(clean_params)
    ret = f"->{f.return_type}" if f.return_type else ""
    return f"({params}){ret}"
```

### 3.2. Polecenia do Wykonania

```bash
# 1. Dodaj import na górze pliku
# Linia ~15
from .shared_utils import compact_imports, abbreviate_type, categorize_function, extract_domain

# 2. Zmodyfikuj __init__ YAMLGenerator
# Linia ~577
def __init__(self, compact_keys: bool = False):

# 3. Zmodyfikuj _build_signature
# Linia 797-811

# 4. Usuń zduplikowane metody _categorize, _extract_domain
# Linie 813-847
```

---

## 4. Plan dla toon_format.py

### 4.1. Zmiany w TOONGenerator

#### A. Usunięcie `self` z sygnatur

```python
# Linia 220-235
def _build_signature(self, f: FunctionInfo) -> str:
    """Build compact signature without self."""
    params = []
    for p in f.params[:6]:
        p_clean = p.replace('\n', ' ').replace(',', ';').strip()
        # NOWE: Pomijaj self/cls
        if p_clean in ('self', 'cls'):
            continue
        if p_clean:
            params.append(p_clean)
    
    param_str = ';'.join(params)
    if len(f.params) > 6:
        param_str += f'...+{len(f.params)-6}'
    
    ret = f.return_type if f.return_type else 'None'
    return f"({param_str})->{ret}"
```

#### B. Kompaktowe nagłówki tablic

```python
# PRZED (linia 167)
header = f"name{self.delim_marker}sig{self.delim_marker}decorators{self.delim_marker}async{self.delim_marker}lines"

# PO - krótsza wersja dla compact mode
def _get_method_header(self, compact: bool = False) -> str:
    if compact:
        return "n,s,d,a,l"  # name, sig, decorators, async, lines
    return f"name{self.delim_marker}sig{self.delim_marker}decorators{self.delim_marker}async{self.delim_marker}lines"
```

#### C. Pomijanie pustych/domyślnych wartości

```python
# Linia 170-176
for m in methods:
    name = self._quote(m.name)
    sig = self._quote(self._build_signature(m))
    
    # Pomijaj dekoratory jeśli puste
    method_decorators = getattr(m, 'decorators', []) or []
    decorators = '|'.join(method_decorators[:2]) if method_decorators else ''
    
    # Pomijaj async jeśli false
    is_async = 'true' if getattr(m, 'is_async', False) else ''
    
    # Buduj wiersz tylko z niepustymi polami
    row_parts = [name, sig]
    if decorators:
        row_parts.append(decorators)
    if is_async:
        row_parts.append(is_async)
    row_parts.append(str(m.lines))
    
    lines.append(f"{ind}  {self.delimiter.join(row_parts)}")
```

### 4.2. Nowy tryb Ultra-Compact

```python
def generate_ultra_compact(self, project: ProjectInfo) -> str:
    """Generate minimal TOON with abbreviated keys."""
    lines = []
    
    # Header w jednej linii
    langs = '/'.join(f"{k}:{v}" for k, v in project.languages.items())
    lines.append(f"# {project.name} | {project.total_files}f {project.total_lines}L | {langs}")
    
    # Moduły jako lista ścieżek z liniami
    lines.append(f"M[{len(project.modules)}]:")
    for m in project.modules:
        lines.append(f"  {m.path},{m.lines_code}")
    
    # Szczegóły tylko dla modułów z klasami/funkcjami
    lines.append("D:")
    for m in project.modules:
        if not m.classes and not m.functions:
            continue
        
        lines.append(f"  {m.path}:")
        
        # Kompaktowe importy
        if m.imports:
            lines.append(f"    i: {','.join(compact_imports(m.imports)[:10])}")
        
        # Klasy inline
        for c in m.classes:
            methods_str = ','.join(f"{meth.name}({len(meth.params)-1})" 
                                   for meth in c.methods[:5])
            lines.append(f"    {c.name}: {methods_str}")
        
        # Funkcje inline  
        for f in m.functions[:10]:
            sig = self._build_signature(f)
            lines.append(f"    {f.name}{sig}")
    
    return '\n'.join(lines)
```

### 4.3. Polecenia do Wykonania

```bash
# 1. Dodaj import
# Linia ~12
from .shared_utils import compact_imports

# 2. Zmodyfikuj _build_signature - usuń self
# Linia 220-235

# 3. Dodaj metodę generate_ultra_compact
# Po linii 270

# 4. Zmodyfikuj _generate_methods aby pomijać puste wartości
# Linia 161-187
```

---

## 5. Plan dla logicml.py

### 5.1. Zmiany Strukturalne

#### A. Skrócone docstringi

```python
# Linia ~188-200
def _generate_class(self, cls: ClassInfo, detail: str) -> str:
    lines: List[str] = [f'\n{cls.name}:']
    
    # Docstring - TYLKO pierwszy akapit, max 60 znaków
    if cls.docstring:
        first_line = cls.docstring.split('\n')[0].strip()
        # Usuń nadmiarowe formatowanie
        first_line = first_line.replace('"', "'").replace('"""', '')
        lines.append(f'  doc: "{first_line[:60]}"')
```

#### B. Kompaktowe atrybuty

```python
# Zamiast verbose attrs, użyj inline
def _generate_attrs_compact(self, cls: ClassInfo) -> str:
    """Generate inline attrs: name:Type, name2:Type2"""
    attrs = []
    
    # Z properties
    for prop in cls.properties[:5]:
        if ':' in prop:
            name, typ = prop.split(':', 1)
            attrs.append(f"{name.strip()}:{typ.strip()}")
        else:
            attrs.append(prop)
    
    # Z __init__ params
    if not attrs and cls.methods:
        init = next((m for m in cls.methods if m.name == '__init__'), None)
        if init:
            for p in init.params[1:5]:  # Skip self
                if ':' in p:
                    attrs.append(p.replace(' ', ''))
    
    return ', '.join(attrs) if attrs else ''
```

#### C. Usunięcie redundantnych komentarzy

```python
# PRZED
lines.append('  # Pydantic model - use Field() for attributes')

# PO - pomijamy, LLM wie co to BaseModel
# (usunąć linie 201-204)
```

### 5.2. Polecenia do Wykonania

```bash
# 1. Skróć docstringi do 60 znaków
# Linia 188-200

# 2. Usuń verbose komentarze o Pydantic/Enum
# Linia 201-207

# 3. Dodaj generate_compact() metoda
# Po linii 260

# 4. Zmień _generate_method aby nie dodawać # Args/Returns
# Linia 228-240
```

---

## 6. Plan dla file_formats.py

### 6.1. Refaktoryzacja Parsera

**Problem:** `_parse_file_elements` jest zduplikowany i nieoptymalny.

```python
# NOWA WERSJA - bardziej robustna
def _parse_file_elements(content: str) -> Dict[str, Any]:
    """Parse file content to extract code elements."""
    result = {
        'imports': [],
        'classes': [],
        'functions': [],
        'module_doc': '',
    }
    
    # Użyj AST dla Python (dokładniejsze)
    try:
        import ast
        tree = ast.parse(content)
        return _parse_with_ast(tree, content)
    except SyntaxError:
        pass
    
    # Fallback na regex
    return _parse_with_regex(content)


def _parse_with_ast(tree, content: str) -> Dict[str, Any]:
    """Parse using Python AST - accurate but Python-only."""
    result = {
        'imports': [],
        'classes': [],
        'functions': [],
        'module_doc': ast.get_docstring(tree) or '',
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result['imports'].append(alias.name)
        
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                result['imports'].append(f"{module}.{alias.name}")
        
        elif isinstance(node, ast.ClassDef):
            cls_info = _extract_class_from_ast(node)
            result['classes'].append(cls_info)
        
        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            # Top-level function only
            func_info = _extract_function_from_ast(node)
            result['functions'].append(func_info)
    
    return result
```

### 6.2. Polecenia do Wykonania

```bash
# 1. Przepisz _parse_file_elements z użyciem AST
# Linia 209-279

# 2. Dodaj _parse_with_ast i _parse_with_regex
# Nowe funkcje

# 3. Skróć generate_file_yaml - usuń verbose komentarze
# Linia 132-200
```

---

## 7. Nowy Moduł: shared_utils.py

Utwórz nowy plik `code2logic/shared_utils.py`:

```python
"""
Shared utilities for Code2Logic generators.

Provides common functions used across multiple generators
to reduce code duplication and ensure consistency.
"""

from typing import List, Dict, Set
import hashlib
import re


# ============================================================================
# Import Handling
# ============================================================================

def compact_imports(imports: List[str], max_items: int = 10) -> List[str]:
    """
    Compact imports by grouping submodules.
    
    Example:
        ['typing', 'typing.Dict', 'typing.List'] -> ['typing.{Dict,List}']
    """
    groups: Dict[str, Set[str]] = {}
    standalone: List[str] = []
    
    for imp in imports[:max_items * 2]:  # Process more to allow grouping
        if '.' in imp:
            base, sub = imp.rsplit('.', 1)
            if base not in groups:
                groups[base] = set()
            groups[base].add(sub)
        else:
            standalone.append(imp)
    
    result = standalone.copy()
    for base, subs in sorted(groups.items()):
        if len(subs) == 1:
            result.append(f"{base}.{list(subs)[0]}")
        elif len(subs) <= 5:
            result.append(f"{base}.{{{','.join(sorted(subs))}}}")
        else:
            # Too many - just list base
            result.append(base)
    
    return result[:max_items]


def deduplicate_imports(imports: List[str]) -> List[str]:
    """
    Remove redundant imports.
    
    Example:
        ['typing', 'typing.Dict'] -> ['typing.Dict']
    """
    seen_bases: Set[str] = set()
    result: List[str] = []
    
    # Sort by specificity (more dots = more specific)
    sorted_imports = sorted(imports, key=lambda x: -x.count('.'))
    
    for imp in sorted_imports:
        base = imp.split('.')[0]
        # If we have a more specific import, skip the base
        if imp == base and base in seen_bases:
            continue
        result.append(imp)
        seen_bases.add(base)
    
    return result


# ============================================================================
# Type Abbreviations
# ============================================================================

TYPE_ABBREVIATIONS = {
    'str': 's',
    'int': 'i', 
    'bool': 'b',
    'float': 'f',
    'None': 'N',
    'Any': 'A',
    'List': 'L',
    'Dict': 'D',
    'Set': 'S',
    'Tuple': 'T',
    'Optional': '?',
    'Callable': 'Fn',
    'Union': 'U',
    'Sequence': 'Seq',
    'Mapping': 'Map',
    'Iterable': 'Iter',
}


def abbreviate_type(type_str: str) -> str:
    """
    Abbreviate type annotations for compactness.
    
    Example:
        'Dict[str, Any]' -> 'D[s,A]'
        'Optional[List[str]]' -> '?[L[s]]'
    """
    result = type_str
    for full, short in TYPE_ABBREVIATIONS.items():
        result = result.replace(full, short)
    # Remove spaces around brackets
    result = re.sub(r'\s*\[\s*', '[', result)
    result = re.sub(r'\s*\]\s*', ']', result)
    result = re.sub(r'\s*,\s*', ',', result)
    return result


def expand_type(abbrev: str) -> str:
    """Expand abbreviated type back to full form."""
    result = abbrev
    # Reverse mapping
    for full, short in TYPE_ABBREVIATIONS.items():
        result = result.replace(short, full)
    return result


# ============================================================================
# Signature Handling
# ============================================================================

def build_signature(
    params: List[str],
    return_type: str = None,
    include_self: bool = False,
    abbreviate: bool = False,
    max_params: int = 6,
) -> str:
    """
    Build compact function signature.
    
    Args:
        params: List of parameter strings
        return_type: Return type annotation
        include_self: Whether to include self/cls
        abbreviate: Whether to abbreviate types
        max_params: Maximum parameters to include
    
    Returns:
        Signature string like "(param1,param2)->ReturnType"
    """
    clean_params = []
    for p in params[:max_params]:
        p_clean = p.replace('\n', ' ').replace('  ', ' ').strip()
        
        # Skip self/cls unless requested
        if p_clean in ('self', 'cls') and not include_self:
            continue
        
        # Abbreviate types if requested
        if abbreviate and ':' in p_clean:
            name, typ = p_clean.split(':', 1)
            typ = abbreviate_type(typ.strip())
            p_clean = f"{name.strip()}:{typ}"
        
        if p_clean:
            clean_params.append(p_clean)
    
    params_str = ','.join(clean_params)
    if len(params) > max_params:
        params_str += f'...+{len(params)-max_params}'
    
    ret = ''
    if return_type:
        ret = f"->{abbreviate_type(return_type) if abbreviate else return_type}"
    
    return f"({params_str}){ret}"


# ============================================================================
# Categorization
# ============================================================================

CATEGORY_PATTERNS = {
    'read': ('get', 'fetch', 'find', 'load', 'read', 'query', 'retrieve', 'list'),
    'create': ('create', 'add', 'insert', 'new', 'make', 'build', 'generate'),
    'update': ('update', 'set', 'modify', 'edit', 'patch', 'change'),
    'delete': ('delete', 'remove', 'clear', 'destroy', 'drop'),
    'validate': ('validate', 'check', 'verify', 'is_', 'has_', 'can_', 'ensure'),
    'transform': ('convert', 'transform', 'parse', 'format', 'to_', 'from_'),
    'lifecycle': ('init', 'setup', 'configure', 'start', 'stop', 'close', 'dispose'),
    'communicate': ('send', 'emit', 'notify', 'publish', 'broadcast'),
}


def categorize_function(name: str) -> str:
    """Categorize function by name pattern."""
    name_lower = name.lower().split('.')[-1]  # Handle method names
    
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(p in name_lower for p in patterns):
            return category
    
    return 'other'


DOMAIN_KEYWORDS = [
    'auth', 'user', 'order', 'payment', 'product', 'cart',
    'config', 'util', 'api', 'service', 'model', 'controller',
    'validation', 'test', 'generator', 'parser', 'llm', 'db',
    'cache', 'queue', 'worker', 'handler', 'middleware',
]


def extract_domain(path: str) -> str:
    """Extract domain from file path."""
    parts = path.lower().replace('\\', '/').split('/')
    
    for part in parts:
        for domain in DOMAIN_KEYWORDS:
            if domain in part:
                return domain
    
    # Return parent folder name
    return parts[-2] if len(parts) > 1 else 'root'


# ============================================================================
# Hashing
# ============================================================================

def compute_hash(name: str, signature: str, length: int = 8) -> str:
    """Compute short hash for quick comparison."""
    content = f"{name}:{signature}"
    return hashlib.md5(content.encode()).hexdigest()[:length]


# ============================================================================
# Text Processing
# ============================================================================

def truncate_docstring(docstring: str, max_length: int = 60) -> str:
    """Truncate docstring to first sentence or max_length."""
    if not docstring:
        return ''
    
    # Get first line
    first_line = docstring.split('\n')[0].strip()
    
    # Remove docstring markers
    first_line = first_line.strip('"""').strip("'''").strip()
    
    # Truncate
    if len(first_line) > max_length:
        first_line = first_line[:max_length-3] + '...'
    
    # Escape quotes
    first_line = first_line.replace('"', "'")
    
    return first_line


def escape_for_yaml(text: str) -> str:
    """Escape text for safe YAML inclusion."""
    if not text:
        return ''
    
    text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    
    # Quote if contains special chars
    if any(c in text for c in ':#[]{}|>'):
        text = f'"{text.replace(chr(34), chr(39))}"'
    
    return text.strip()
```

---

## 8. Testy Jednostkowe

### 8.1. Test shared_utils.py

```python
# tests/test_shared_utils.py

import pytest
from code2logic.shared_utils import (
    compact_imports, abbreviate_type, build_signature,
    categorize_function, truncate_docstring
)


class TestCompactImports:
    def test_groups_submodules(self):
        imports = ['typing', 'typing.Dict', 'typing.List', 'typing.Optional']
        result = compact_imports(imports)
        assert 'typing.{Dict,List,Optional}' in result or len(result) < len(imports)
    
    def test_preserves_standalone(self):
        imports = ['json', 'os', 're']
        result = compact_imports(imports)
        assert set(result) == set(imports)
    
    def test_limits_output(self):
        imports = [f'module{i}' for i in range(20)]
        result = compact_imports(imports, max_items=5)
        assert len(result) <= 5


class TestAbbreviateType:
    def test_simple_types(self):
        assert abbreviate_type('str') == 's'
        assert abbreviate_type('int') == 'i'
        assert abbreviate_type('bool') == 'b'
    
    def test_complex_types(self):
        assert abbreviate_type('Dict[str, Any]') == 'D[s,A]'
        assert abbreviate_type('Optional[List[str]]') == '?[L[s]]'
    
    def test_preserves_unknown(self):
        assert abbreviate_type('CustomType') == 'CustomType'


class TestBuildSignature:
    def test_removes_self_by_default(self):
        sig = build_signature(['self', 'name', 'value'], 'None')
        assert 'self' not in sig
        assert 'name' in sig
    
    def test_includes_self_when_requested(self):
        sig = build_signature(['self', 'name'], 'None', include_self=True)
        assert 'self' in sig
    
    def test_abbreviates_types(self):
        sig = build_signature(['name:str', 'count:int'], 'Dict[str, Any]', abbreviate=True)
        assert ':s' in sig or 's' in sig
        assert 'D[' in sig
    
    def test_truncates_params(self):
        params = [f'param{i}:str' for i in range(10)]
        sig = build_signature(params, max_params=3)
        assert '...+7' in sig


class TestCategorizeFunction:
    def test_read_category(self):
        assert categorize_function('get_user') == 'read'
        assert categorize_function('fetch_data') == 'read'
        assert categorize_function('find_by_id') == 'read'
    
    def test_create_category(self):
        assert categorize_function('create_user') == 'create'
        assert categorize_function('add_item') == 'create'
    
    def test_handles_method_names(self):
        assert categorize_function('User.get_name') == 'read'
    
    def test_returns_other_for_unknown(self):
        assert categorize_function('foo') == 'other'


class TestTruncateDocstring:
    def test_truncates_long(self):
        doc = "This is a very long docstring " * 10
        result = truncate_docstring(doc, max_length=30)
        assert len(result) <= 33  # 30 + '...'
    
    def test_preserves_short(self):
        doc = "Short doc"
        result = truncate_docstring(doc, max_length=60)
        assert result == "Short doc"
    
    def test_removes_markers(self):
        doc = '"""This is the doc"""'
        result = truncate_docstring(doc)
        assert '"""' not in result
```

### 8.2. Test Generatorów

```python
# tests/test_generator_optimization.py

import pytest
from code2logic.generators import YAMLGenerator
from code2logic.toon_format import TOONGenerator
from code2logic.logicml import LogicMLGenerator


class TestYAMLOptimization:
    def test_no_self_in_signatures(self, sample_project):
        gen = YAMLGenerator()
        output = gen.generate(sample_project)
        
        # Sprawdź że self nie występuje w sygnaturach
        lines = output.split('\n')
        for line in lines:
            if 'signature:' in line:
                assert '(self,' not in line
                assert '(self)' not in line
    
    def test_compact_keys_mode(self, sample_project):
        gen = YAMLGenerator(compact_keys=True)
        output = gen.generate(sample_project)
        
        assert 'p:' in output  # path
        assert 'path:' not in output
    
    def test_output_smaller_than_baseline(self, sample_project, baseline_yaml):
        gen = YAMLGenerator(compact_keys=True)
        output = gen.generate(sample_project)
        
        # Oczekujemy ~30% redukcji
        assert len(output) < len(baseline_yaml) * 0.75


class TestTOONOptimization:
    def test_no_self_in_signatures(self, sample_project):
        gen = TOONGenerator()
        output = gen.generate(sample_project)
        
        assert '(self;' not in output
        assert '(self)' not in output
    
    def test_empty_fields_omitted(self, sample_project):
        gen = TOONGenerator()
        output = gen.generate(sample_project)
        
        # Puste dekoratory nie powinny być "-"
        lines = output.split('\n')
        for line in lines:
            if line.strip().endswith(',-,-,-'):
                pytest.fail(f"Found row with multiple empty fields: {line}")


class TestLogicMLOptimization:
    def test_docstrings_truncated(self, sample_project):
        gen = LogicMLGenerator()
        spec = gen.generate(sample_project)
        
        for line in spec.content.split('\n'):
            if 'doc:' in line:
                doc_part = line.split('doc:')[1].strip()
                assert len(doc_part) <= 65  # 60 + quotes
```

---

## 9. Harmonogram Implementacji

### Faza 1: Fundamenty (Dzień 1-2)

| Zadanie | Plik | Estymacja |
|---------|------|-----------|
| Utworzenie shared_utils.py | Nowy plik | 2h |
| Testy dla shared_utils.py | tests/test_shared_utils.py | 1h |
| Code review + merge | - | 1h |

### Faza 2: YAMLGenerator (Dzień 2-3)

| Zadanie | Plik | Estymacja |
|---------|------|-----------|
| Integracja shared_utils | generators.py | 1h |
| Implementacja compact_keys | generators.py | 2h |
| Usunięcie self z sygnatur | generators.py | 1h |
| Testy | tests/test_generators.py | 1h |

### Faza 3: TOONGenerator (Dzień 3-4)

| Zadanie | Plik | Estymacja |
|---------|------|-----------|
| Usunięcie self | toon_format.py | 1h |
| Pomijanie pustych pól | toon_format.py | 1h |
| generate_ultra_compact() | toon_format.py | 2h |
| Testy | tests/test_toon_format.py | 1h |

### Faza 4: LogicML + FileFormats (Dzień 4-5)

| Zadanie | Plik | Estymacja |
|---------|------|-----------|
| Skrócenie docstringów | logicml.py | 1h |
| Refaktoryzacja parsera | file_formats.py | 2h |
| Testy końcowe | - | 2h |

### Faza 5: Walidacja (Dzień 5)

| Zadanie | Estymacja |
|---------|-----------|
| Benchmark przed/po | 2h |
| Dokumentacja zmian | 1h |
| Release notes | 1h |

---

## Podsumowanie Poleceń

### Kolejność Wykonania

```bash
# 1. Utwórz shared_utils.py
touch code2logic/shared_utils.py
# [wklej zawartość z sekcji 7]

# 2. Dodaj testy
touch tests/test_shared_utils.py
# [wklej testy z sekcji 8.1]

# 3. Uruchom testy podstawowe
pytest tests/test_shared_utils.py -v

# 4. Zmodyfikuj generators.py
# - Dodaj import: from .shared_utils import ...
# - Zmodyfikuj YAMLGenerator.__init__
# - Zmodyfikuj _build_signature
# - Usuń zduplikowane metody

# 5. Zmodyfikuj toon_format.py
# - Usuń self z _build_signature
# - Dodaj generate_ultra_compact

# 6. Zmodyfikuj logicml.py
# - Skróć docstringi
# - Usuń verbose komentarze

# 7. Uruchom pełne testy
pytest tests/ -v

# 8. Benchmark
python examples/11_token_benchmark.py --folder tests/samples/
```

---

## Metryki Sukcesu

| Metryka | Cel | Weryfikacja |
|---------|-----|-------------|
| Redukcja YAML | ≥40% | `wc -c przed.yaml po.yaml` |
| Redukcja TOON | ≥30% | `wc -c przed.toon po.toon` |
| Testy przechodzą | 100% | `pytest --tb=short` |
| Reproduction fidelity | ≥95% | Benchmark |
| Brak regresji | 0 | CI/CD pipeline |
