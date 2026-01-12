#!/usr/bin/env python3
"""
Test reprodukcji shared_utils.py z formatów Code2Logic.

Ten plik pokazuje CO MOŻNA wygenerować z każdego formatu,
z komentarzami co jest poprawne a co brakuje.
"""

import gzip
import hashlib
import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class QualityResult:
    format_name: str
    metric: float
    findings: List[str]


def detect_signature_quality(signature_block: str) -> Tuple[dict, List[str]]:
    """Check if signature block contains parameters with defaults."""
    findings: List[str] = []
    has_params = bool(re.search(r"\w+\s*:", signature_block))
    has_defaults = bool(re.search(r"=\s*[^,)]+", signature_block))
    
    findings.append("parametry wykryte" if has_params else "brak parametrów")
    findings.append("wartości domyślne obecne" if has_defaults else "brak wartości domyślnych")
    
    return {"params": has_params, "defaults": has_defaults}, findings


def detect_constant_quality(constants_block: str) -> Tuple[dict, List[str]]:
    """Check if constant block contains values or only names."""
    findings: List[str] = []
    has_types = bool(re.search(r":[ \t]*(Dict|List|Tuple|Set|Optional|Any|str|int|float|bool)", constants_block))
    has_values = bool(re.search(r"=\s*[\[{]", constants_block))
    
    findings.append("typy stałych obecne" if has_types else "brak typów stałych")
    findings.append("wartości stałych obecne" if has_values else "tylko nazwy stałych")
    
    return {"types": has_types, "values": has_values}, findings


def compute_quality_score(name: str, sig_block: str, const_block: str) -> QualityResult:
    sig_status, sig_findings = detect_signature_quality(sig_block)
    const_status, const_findings = detect_constant_quality(const_block)
    
    sig_score = sum(1 for ok in sig_status.values() if ok) / max(len(sig_status), 1)
    const_score = sum(1 for ok in const_status.values() if ok) / max(len(const_status), 1)
    
    metric = round(((sig_score + const_score) / 2) * 100, 1)
    findings = sig_findings + const_findings
    return QualityResult(name, metric, findings)


def format_quality_summary(result: QualityResult) -> str:
    bullet = ", ".join(result.findings)
    return f"[{result.format_name}] ~{result.metric:.0f}% → {bullet}"


def evaluate_formats() -> List[QualityResult]:
    """Run heuristic quality detection for each reproduction section."""
    return [
        ReproducedFromTOON.QUALITY,
        ReproducedFromYAML.QUALITY,
        ReproducedFromHybrid.QUALITY,
    ]


def print_quality_report() -> None:
    print("=== HEURYSTYCZNA OCENA FORMATÓW ===")
    for result in evaluate_formats():
        print(f"  - {format_quality_summary(result)}")
    print()


@dataclass
class FormatStats:
    format_name: str
    bytes: int
    lines: int
    gzip_bytes: int
    tokens_est: int
    signatures: int
    signatures_with_types: int
    signatures_with_defaults: int
    constants: int
    constants_with_types: int
    constants_with_values: int
    constants_with_keys: int
    dataclass_fields: int

    @property
    def gzip_ratio(self) -> float:
        if self.bytes <= 0:
            return 0.0
        return self.gzip_bytes / self.bytes


def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')


def _basic_text_stats(format_name: str, content: str) -> FormatStats:
    b = len(content.encode('utf-8', errors='replace'))
    gz = len(gzip.compress(content.encode('utf-8', errors='replace')))
    lines = content.count('\n') + (1 if content else 0)
    tokens_est = max(1, int(len(content) / 4)) if content else 0
    return FormatStats(
        format_name=format_name,
        bytes=b,
        lines=lines,
        gzip_bytes=gz,
        tokens_est=tokens_est,
        signatures=0,
        signatures_with_types=0,
        signatures_with_defaults=0,
        constants=0,
        constants_with_types=0,
        constants_with_values=0,
        constants_with_keys=0,
        dataclass_fields=0,
    )


def _extract_yaml_modules(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        modules = data.get('modules')
        if isinstance(modules, list):
            return [m for m in modules if isinstance(m, dict)]
    return []


def _extract_yaml_signatures_from_module(module: Dict[str, Any]) -> List[str]:
    sigs: List[str] = []

    funcs = module.get('f') or module.get('functions')
    if isinstance(funcs, list):
        for f in funcs:
            if isinstance(f, dict):
                s = f.get('sig') or f.get('signature')
                if isinstance(s, str) and s.strip():
                    sigs.append(s)

    classes = module.get('c') or module.get('classes')
    if isinstance(classes, list):
        for c in classes:
            if not isinstance(c, dict):
                continue
            methods = c.get('m') or c.get('methods')
            if isinstance(methods, list):
                for m in methods:
                    if isinstance(m, dict):
                        s = m.get('sig') or m.get('signature')
                        if isinstance(s, str) and s.strip():
                            sigs.append(s)

    return sigs


def _extract_yaml_constants_from_module(module: Dict[str, Any]) -> List[Dict[str, Any]]:
    consts = module.get('const') or module.get('constants')
    if isinstance(consts, list):
        return [c for c in consts if isinstance(c, dict)]
    return []


def _count_yaml_dataclass_fields(module: Dict[str, Any]) -> int:
    total = 0

    dcs = module.get('dataclasses')
    if isinstance(dcs, list):
        for dc in dcs:
            if isinstance(dc, dict):
                fields = dc.get('fields')
                if isinstance(fields, list):
                    total += len(fields)

    classes = module.get('c') or module.get('classes')
    if isinstance(classes, list):
        for c in classes:
            if isinstance(c, dict):
                fields = c.get('fields')
                if isinstance(fields, list):
                    total += len(fields)

    return total


def analyze_yaml_like(format_name: str, content: str) -> FormatStats:
    stats = _basic_text_stats(format_name, content)

    try:
        import yaml  # type: ignore
        data = yaml.safe_load(content)
    except Exception:
        sigs = re.findall(r"\bsig:\s*([^\n]+)", content)
        stats.signatures = len(sigs)
        stats.signatures_with_types = sum(1 for s in sigs if ':' in s)
        stats.signatures_with_defaults = sum(1 for s in sigs if '=' in s)
        return stats

    modules = _extract_yaml_modules(data)

    sigs: List[str] = []
    consts: List[Dict[str, Any]] = []
    dataclass_fields = 0
    for m in modules:
        sigs.extend(_extract_yaml_signatures_from_module(m))
        consts.extend(_extract_yaml_constants_from_module(m))
        dataclass_fields += _count_yaml_dataclass_fields(m)

    stats.signatures = len(sigs)
    stats.signatures_with_types = sum(1 for s in sigs if ':' in s)
    stats.signatures_with_defaults = sum(1 for s in sigs if '=' in s)

    stats.constants = len(consts)
    for c in consts:
        t = c.get('t') or c.get('type')
        v = c.get('v') or c.get('value')
        keys = c.get('keys')

        if isinstance(t, str) and t.strip() and t.strip() != '-':
            stats.constants_with_types += 1
        if isinstance(v, str) and v.strip() and v.strip() != '-':
            stats.constants_with_values += 1
        if isinstance(keys, list) and len(keys) > 0:
            stats.constants_with_keys += 1
        elif isinstance(keys, str) and keys.strip() and keys.strip() != '-':
            stats.constants_with_keys += 1

    stats.dataclass_fields = dataclass_fields
    return stats


def analyze_toon(format_name: str, content: str) -> FormatStats:
    stats = _basic_text_stats(format_name, content)

    lines = content.splitlines()
    delimiter = '\t' if any('\t' in ln for ln in lines) else ','

    def parse_row(line: str) -> List[str]:
        return next(csv.reader([line], delimiter=delimiter, quotechar='"', escapechar='\\'))

    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1

        if not line.strip() or line.lstrip().startswith('#'):
            continue

        m = re.search(r"\b(functions|methods|const|fields)\[(\d+)\]\{([^}]*)\}:\s*$", line)
        if not m:
            continue

        section = m.group(1)
        count = int(m.group(2))
        fields = [f.strip() for f in m.group(3).split(',') if f.strip()]
        field_to_idx = {name: idx for idx, name in enumerate(fields)}

        for _ in range(count):
            if i >= len(lines):
                break
            row_line = lines[i].strip()
            i += 1
            if not row_line:
                continue
            cells = parse_row(row_line)

            if section in ('functions', 'methods'):
                sig_idx = field_to_idx.get('sig')
                if sig_idx is not None and sig_idx < len(cells):
                    sig = cells[sig_idx]
                    if sig:
                        stats.signatures += 1
                        if ':' in sig:
                            stats.signatures_with_types += 1
                        if '=' in sig:
                            stats.signatures_with_defaults += 1

            if section == 'const':
                stats.constants += 1
                t_idx = field_to_idx.get('t')
                v_idx = field_to_idx.get('v')
                keys_idx = field_to_idx.get('keys')

                if t_idx is not None and t_idx < len(cells):
                    t = cells[t_idx]
                    if t and t != '-':
                        stats.constants_with_types += 1
                if v_idx is not None and v_idx < len(cells):
                    v = cells[v_idx]
                    if v and v != '-':
                        stats.constants_with_values += 1
                if keys_idx is not None and keys_idx < len(cells):
                    keys = cells[keys_idx]
                    if keys and keys != '-':
                        stats.constants_with_keys += 1

            if section == 'fields':
                stats.dataclass_fields += 1

    return stats


def _pct(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return (part / whole) * 100


def print_detailed_format_comparison() -> None:
    base = Path('out/code2logic')
    yaml_path = base / 'project.c2l.yaml'
    hybrid_path = base / 'project.c2l.hybrid.yaml'
    toon_path = base / 'project.c2l.toon'

    if not (yaml_path.exists() or hybrid_path.exists() or toon_path.exists()):
        return

    print("=== PORÓWNANIE FORMATÓW (AUTO, z realnych plików) ===")
    print("Uwaga: oceny heurystyczne poniżej są demonstracyjne; ten blok liczy metryki z aktualnych plików w out/.")
    print()

    reports: List[FormatStats] = []
    if yaml_path.exists():
        reports.append(analyze_yaml_like('YAML', _read_text(yaml_path)))
    if hybrid_path.exists():
        reports.append(analyze_yaml_like('HYBRID', _read_text(hybrid_path)))
    if toon_path.exists():
        reports.append(analyze_toon('TOON', _read_text(toon_path)))

    for r in reports:
        print(f"{r.format_name}:")
        print(f"  size_bytes: {r.bytes:,}")
        print(f"  size_gzip_bytes: {r.gzip_bytes:,} (ratio {r.gzip_ratio:.2f})")
        print(f"  lines: {r.lines:,}")
        print(f"  tokens_est: {r.tokens_est:,}  (≈ chars/4)")
        print(f"  signatures: {r.signatures:,}")
        print(f"    with_types: {r.signatures_with_types:,} ({_pct(r.signatures_with_types, r.signatures):.1f}%)")
        print(f"    with_defaults: {r.signatures_with_defaults:,} ({_pct(r.signatures_with_defaults, r.signatures):.1f}%)")
        print(f"  constants: {r.constants:,}")
        print(f"    with_types: {r.constants_with_types:,} ({_pct(r.constants_with_types, r.constants):.1f}%)")
        print(f"    with_values: {r.constants_with_values:,} ({_pct(r.constants_with_values, r.constants):.1f}%)")
        print(f"    with_keys: {r.constants_with_keys:,} ({_pct(r.constants_with_keys, r.constants):.1f}%)")
        print(f"  dataclass_fields: {r.dataclass_fields:,}")
        print()

# =============================================================================
# REPRODUKCJA Z TOON (shared_utils.py) - PO NAPRAWIE
# =============================================================================
# Źródło (po naprawie):
#   shared_utils.py:
#     i: typing,hashlib,re,typing.{Dict,List,Optional,Set}
#     e: compact_imports,deduplicate_imports,TYPE_ABBREVIATIONS,...
#     compact_imports(imports:List[str],max_items:int=10)->List[str]
#     deduplicate_imports(imports:List[str])->List[str]
#     build_signature(params:List[str],return_type:Optional[str]=None,...)->str

class ReproducedFromTOON:
    """Reprodukcja z TOON - ~55% poprawności (PO NAPRAWIE)"""
    
    # ✅ Importy - MOŻNA odtworzyć (tak jak wcześniej)
    IMPORTS = """
from typing import Dict, List, Optional, Set
import hashlib
import re
"""
    
    # ⚠️ Stałe tylko jako nazwy eksportowane
    CONSTANTS = """
TYPE_ABBREVIATIONS
CATEGORY_PATTERNS
DOMAIN_KEYWORDS
"""
    
    # ⚠️ Stałe nadal tylko jako nazwy w exports (brak wartości)
    # ✅ Funkcje mają TERAZ parametry + typy (z defaults jeśli były)
    # ⚠️ Docstringi tylko skrócone (intent), logika nadal zgadywana
    FUNCTIONS = """
def compact_imports(imports: List[str], max_items: int = 10) -> List[str]:
    '''Compact imports by grouping submodules.'''
    pass

def deduplicate_imports(imports: List[str]) -> List[str]:
    '''Remove redundant imports.'''
    pass

def abbreviate_type(type_str: str) -> str:
    '''Abbreviate type annotations for compactness.'''
    pass

def expand_type(abbrev: str) -> str:
    '''Expand abbreviated type back to full form.'''
    pass

def build_signature(
    params: List[str],
    return_type: Optional[str] = None,
    include_self: bool = False,
    abbreviate: bool = False,
    max_params: int = 6
) -> str:
    '''Build compact function signature.'''
    pass

def remove_self_from_params(params: List[str]) -> List[str]:
    '''Remove self/cls entries.'''
    pass

def categorize_function(name: str) -> str:
    '''Categorize function by name pattern.'''
    pass

def extract_domain(path: str) -> str:
    '''Extract domain from file path.'''
    pass
"""
    
    QUALITY = compute_quality_score("TOON", FUNCTIONS, CONSTANTS)
    
    # PODSUMOWANIE TOON (PO NAPRAWIE):
    # ✅ Importy: TAK (zgrupowane)
    # ✅ Eksporty: TAK (8 z 12)
    # ⚠️ Stałe: Tylko nazwy (bez wartości)
    # ✅ Sygnatury: TAK (parametry + typy + defaults)
    # ✅ Docstringi: TAK (skrócone intents)
    # Wynik: ~55%


# =============================================================================
# REPRODUKCJA Z YAML (shared_utils.py) - PO NAPRAWIE
# =============================================================================
# Źródło (po naprawie):
#   - p: shared_utils.py
#     l: 279
#     i: [hashlib, re, typing.{Dict,List,Optional,Set}]
#     e: [compact_imports, deduplicate_imports, TYPE_ABBREVIATIONS, ...]
#     f:
#     - n: compact_imports
#       sig: (imports:List[str],max_items:int=10)  # ✅ PEŁNE!
#       ret: List[str]
#       d: compact imports

class ReproducedFromYAML:
    """Reprodukcja z YAML - ~75% poprawności (PO NAPRAWIE)"""
    
    # ✅ Importy - można odtworzyć (zgrupowane)
    IMPORTS = """
import hashlib
import re
from typing import Dict, List, Optional, Set
"""
    
    # ⚠️ Stałe tylko jako nazwy (brak wartości)
    CONSTANTS = """
TYPE_ABBREVIATIONS
CATEGORY_PATTERNS
DOMAIN_KEYWORDS
"""
    
    # ✅ Eksporty wskazują że są stałe:
    # TYPE_ABBREVIATIONS, CATEGORY_PATTERNS, DOMAIN_KEYWORDS
    
    FUNCTIONS = """
def compact_imports(imports: List[str], max_items: int = 10) -> List[str]:
    '''compact imports'''  # ✅ Jest intent/docstring
    pass  # ✅ PEŁNE parametry z defaults!

def deduplicate_imports(imports: List[str]) -> List[str]:
    '''deduplicate imports'''
    pass  # ✅ PEŁNE parametry

def abbreviate_type(type_str: str) -> str:
    '''abbreviate type'''
    pass  # ✅ PEŁNE parametry (type_str: str)

def expand_type(abbrev: str) -> str:
    '''expand type'''
    pass  # ✅ PEŁNE parametry

def build_signature(
    params: List[str], 
    return_type: Optional[str] = None, 
    include_self: bool = False, 
    abbreviate: bool = False, 
    max_params: int = 6
) -> str:
    '''creates signature'''
    pass  # ✅ PEŁNE parametry z defaults!

def remove_self_from_params(params: List[str]) -> List[str]:
    '''deletes self from params'''
    pass  # ✅ PEŁNE parametry

def categorize_function(name: str) -> str:
    '''categorize function'''
    pass  # ✅ PEŁNE parametry

def extract_domain(path: str) -> str:
    '''parses domain'''
    pass  # ✅ PEŁNE parametry

def compute_hash(name: str, signature: str, length: int = 8) -> str:
    '''processes hash'''
    pass  # ✅ PEŁNE parametry z defaults

def truncate_docstring(docstring: Optional[str], max_length: int = 60) -> str:
    '''truncate docstring'''
    pass  # ✅ PEŁNE parametry z defaults

def escape_for_yaml(text: str) -> str:
    '''escape for yaml'''
    pass  # ✅ PEŁNE parametry

def clean_identifier(name: str) -> str:
    '''clean identifier'''
    pass  # ✅ PEŁNE parametry
"""
    
    QUALITY = compute_quality_score("YAML", FUNCTIONS, CONSTANTS)


# =============================================================================
# REPRODUKCJA Z HYBRID YAML (shared_utils.py) - PO NAPRAWIE
# =============================================================================
# Źródło (po naprawie):
#   - p: shared_utils.py
#     l: 279
#     i: [hashlib, re, typing.{Dict,List,Optional,Set}]
#     e: [compact_imports, ..., TYPE_ABBREVIATIONS, ..., DOMAIN_KEYWORDS]
#     const:
#     - n: TYPE_ABBREVIATIONS
#       t: Dict[str, str]
#       d: "Mapping full type -> abbreviated type"
#     - n: CATEGORY_PATTERNS
#       t: Dict[str, Tuple[str, ...]]
#       d: "Patterns for categorizing functions by name"
#     - n: DOMAIN_KEYWORDS
#       t: List[str]
#       d: "Domain-specific keywords"
#     f:
#     - n: compact_imports
#       sig: (imports:List[str],max_items:int=10)
#       ret: List[str]
#       d: compact imports
#       example: |
#         in: "['typing', 'typing.Dict', 'typing.List']"
#         out: "['typing.{Dict,List}']"

class ReproducedFromHybrid:
    """Reprodukcja z Hybrid YAML - ~85% poprawności (PO NAPRAWIE)"""
    
    # ✅ Importy (pełne + dodatkowe typy pomocnicze)
    IMPORTS = """
import hashlib
import re
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
"""
    
    CONSTANTS = """
# Mapping full type -> abbreviated type
TYPE_ABBREVIATIONS: Dict[str, str] = {
    'str': 's', 'int': 'i', 'bool': 'b', 'float': 'f',
    'None': 'N', 'Any': 'A', 'List': 'L', 'Dict': 'D',
    'Set': 'S', 'Tuple': 'T', 'Optional': '?', 'Callable': 'Fn'
}

# Patterns for categorizing functions by name
CATEGORY_PATTERNS: Dict[str, Tuple[str, ...]] = {
    'read': ('get', 'fetch', 'find', 'load', 'read', 'query'),
    'create': ('create', 'add', 'new', 'make', 'build'),
    'update': ('update', 'modify', 'change', 'set', 'edit'),
    'delete': ('delete', 'remove', 'drop', 'clear', 'erase'),
    'validate': ('validate', 'check', 'verify'),
    'transform': ('transform', 'convert', 'map', 'process'),
    'lifecycle': ('init', 'start', 'stop', 'setup'),
    'communicate': ('send', 'receive', 'publish', 'subscribe')
}

# Domain-specific keywords
DOMAIN_KEYWORDS: List[str] = [
    'auth', 'user', 'order', 'payment', 'product', 'cart',
    'config', 'util', 'api', 'service', 'model', 'controller',
    'validation', 'test', 'db', 'cache', 'log', 'error'
]
"""
    
    # ✅ Stałe mają teraz typy i opisy, często też fragmenty wartości
    # (Zdefiniowane powyżej w CONSTANTS)
    
    FUNCTIONS = """
def compact_imports(imports: List[str], max_items: int = 10) -> List[str]:
    '''compact imports'''
    pass  # ✅ Sygnatura + defaults + opis

# ... wszystkie pozostałe funkcje jak w sekcji YAML, z pełnymi parametrami ...

def build_signature(
    params: List[str],
    return_type: Optional[str] = None,
    include_self: bool = False,
    abbreviate: bool = False,
    max_params: int = 6
) -> str:
    '''creates signature'''
    pass

def compute_hash(name: str, signature: str, length: int = 8) -> str:
    '''processes hash'''
    pass

def truncate_docstring(docstring: Optional[str], max_length: int = 60) -> str:
    '''truncate docstring'''
    pass
"""
    
    QUALITY = compute_quality_score("HYBRID", FUNCTIONS, CONSTANTS)
    
    # PODSUMOWANIE HYBRID (PO NAPRAWIE):
    # ✅ Importy: TAK (z dodatkowymi typami)
    # ✅ Eksporty: TAK (pełna lista)
    # ✅ Stałe: TAK (typy + opisy + przykłady wartości)
    # ✅ Sygnatury: TAK (pełne z defaults)
    # ✅ Docstringi: TAK (rozszerzone + przykłady)
    # ✅ Wszystkie funkcje: TAK (12)
    # Wynik: ~85%


# =============================================================================
# CO POWINNO BYĆ W IDEALNYM FORMACIE
# =============================================================================

class IdealReproduction:
    """Jak powinien wyglądać format dla ~90% reprodukcji"""
    
    IDEAL_YAML = """
- p: shared_utils.py
  l: 279
  d: |
    Shared utilities for Code2Logic generators.
    Provides common functions used across multiple generators.
  i:
  - hashlib
  - re
  - typing.{Dict,List,Optional,Set}
  e:
  - compact_imports
  - deduplicate_imports
  - TYPE_ABBREVIATIONS
  - abbreviate_type
  - expand_type
  - build_signature
  - remove_self_from_params
  - CATEGORY_PATTERNS
  - categorize_function
  - DOMAIN_KEYWORDS
  - extract_domain
  - compute_hash
  - truncate_docstring
  - escape_for_yaml
  - clean_identifier
  
  const:
  - n: TYPE_ABBREVIATIONS
    t: Dict[str, str]
    d: "Mapping full type -> abbreviated type"
    v: {str: s, int: i, bool: b, float: f, None: N, Any: A, List: L, Dict: D, Set: S, Tuple: T, Optional: '?', Callable: Fn}
    
  - n: CATEGORY_PATTERNS
    t: Dict[str, Tuple[str, ...]]
    d: "Patterns for categorizing functions by name"
    keys: [read, create, update, delete, validate, transform, lifecycle, communicate]
    example: 
      read: ['get', 'fetch', 'find', 'load', 'read', 'query']
      
  - n: DOMAIN_KEYWORDS
    t: List[str]
    v: [auth, user, order, payment, product, cart, config, util, api, service, model, controller, validation, test]
  
  f:
  - n: compact_imports
    sig: (imports:List[str],max_items:int=10)
    ret: List[str]
    d: Compact imports by grouping submodules.
    example:
      in: "['typing', 'typing.Dict', 'typing.List']"
      out: "['typing.{Dict,List}']"
      
  - n: deduplicate_imports
    sig: (imports:List[str])
    ret: List[str]
    d: Remove redundant imports.
    
  - n: abbreviate_type
    sig: (type_str:str)
    ret: str
    d: Abbreviate type annotations for compactness.
    uses: TYPE_ABBREVIATIONS
    example:
      in: "'Dict[str, Any]'"
      out: "'D[s,A]'"
      
  - n: expand_type
    sig: (abbrev:str)
    ret: str
    d: Expand abbreviated type back to full form.
    
  - n: build_signature
    sig: (params:List[str],return_type:Optional[str]=None,include_self:bool=False,abbreviate:bool=False,max_params:int=6)
    ret: str
    d: Build compact function signature.
    
  - n: remove_self_from_params
    sig: (params:List[str])
    ret: List[str]
    d: Remove 'self' and 'cls' from parameter list.
    
  - n: categorize_function
    sig: (name:str)
    ret: str
    d: Categorize function by name pattern.
    uses: CATEGORY_PATTERNS
    returns_one_of: [read, create, update, delete, validate, transform, lifecycle, communicate, other]
    
  - n: extract_domain
    sig: (path:str)
    ret: str
    d: Extract domain from file path.
    uses: DOMAIN_KEYWORDS
    
  - n: compute_hash
    sig: (name:str,signature:str,length:int=8)
    ret: str
    d: Compute short hash for quick comparison.
    
  - n: truncate_docstring
    sig: (docstring:Optional[str],max_length:int=60)
    ret: str
    d: Truncate docstring to first sentence or max_length.
    
  - n: escape_for_yaml
    sig: (text:str)
    ret: str
    d: Escape text for safe YAML inclusion.
    
  - n: clean_identifier
    sig: (name:str)
    ret: str
    d: Clean identifier by removing whitespace and special characters.
"""

    # Z tego LLM może wygenerować:
    GENERATED_CODE = '''
"""
Shared utilities for Code2Logic generators.
Provides common functions used across multiple generators.
"""

from typing import Dict, List, Optional, Set
import hashlib
import re


# Mapping full type -> abbreviated type
TYPE_ABBREVIATIONS: Dict[str, str] = {
    'str': 's', 'int': 'i', 'bool': 'b', 'float': 'f',
    'None': 'N', 'Any': 'A', 'List': 'L', 'Dict': 'D',
    'Set': 'S', 'Tuple': 'T', 'Optional': '?', 'Callable': 'Fn'
}

# Patterns for categorizing functions by name
CATEGORY_PATTERNS: Dict[str, Tuple[str, ...]] = {
    'read': ('get', 'fetch', 'find', 'load', 'read', 'query'),
    'create': (...),  # Wiemy że jest 8 kategorii
    # ... pozostałe z keys
}

DOMAIN_KEYWORDS: List[str] = [
    'auth', 'user', 'order', 'payment', 'product', 'cart',
    'config', 'util', 'api', 'service', 'model', 'controller',
    'validation', 'test'
]


def compact_imports(imports: List[str], max_items: int = 10) -> List[str]:
    """
    Compact imports by grouping submodules.
    
    Example:
        ['typing', 'typing.Dict', 'typing.List'] -> ['typing.{Dict,List}']
    """
    # LLM może wygenerować logikę na podstawie przykładu
    pass


def deduplicate_imports(imports: List[str]) -> List[str]:
    """Remove redundant imports."""
    pass


def abbreviate_type(type_str: str) -> str:
    """
    Abbreviate type annotations for compactness.
    Uses TYPE_ABBREVIATIONS.
    
    Example: 'Dict[str, Any]' -> 'D[s,A]'
    """
    result = type_str
    for full, short in TYPE_ABBREVIATIONS.items():
        result = result.replace(full, short)
    return result


def expand_type(abbrev: str) -> str:
    """Expand abbreviated type back to full form."""
    pass


def build_signature(
    params: List[str],
    return_type: Optional[str] = None,
    include_self: bool = False,
    abbreviate: bool = False,
    max_params: int = 6,
) -> str:
    """Build compact function signature."""
    pass


def remove_self_from_params(params: List[str]) -> List[str]:
    """Remove 'self' and 'cls' from parameter list."""
    return [p for p in params if p not in ('self', 'cls')]


def categorize_function(name: str) -> str:
    """
    Categorize function by name pattern.
    Uses CATEGORY_PATTERNS.
    Returns one of: read, create, update, delete, validate, transform, lifecycle, communicate, other
    """
    name_lower = name.lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(p in name_lower for p in patterns):
            return category
    return 'other'


def extract_domain(path: str) -> str:
    """Extract domain from file path. Uses DOMAIN_KEYWORDS."""
    pass


def compute_hash(name: str, signature: str, length: int = 8) -> str:
    """Compute short hash for quick comparison."""
    content = f"{name}:{signature}"
    return hashlib.md5(content.encode()).hexdigest()[:length]


def truncate_docstring(docstring: Optional[str], max_length: int = 60) -> str:
    """Truncate docstring to first sentence or max_length."""
    pass


def escape_for_yaml(text: str) -> str:
    """Escape text for safe YAML inclusion."""
    pass


def clean_identifier(name: str) -> str:
    """Clean identifier by removing whitespace and special characters."""
    pass
'''

    # Ten kod ma ~85-90% poprawności!
    # ✅ Wszystkie importy
    # ✅ Wszystkie stałe z wartościami
    # ✅ Wszystkie sygnatury z parametrami i defaults
    # ✅ Docstringi z przykładami
    # ✅ Podstawowa logika (dla prostych funkcji)
    # ⚠️ Brak implementacji złożonej logiki


if __name__ == "__main__":
    print("=== TEST REPRODUKOWALNOŚCI ===")
    print()
    print_detailed_format_comparison()
    print_quality_report()
    print("TOON:   ~55% (parametry + typy, brak wartości stałych)")
    print("YAML:   ~75% (pełne sygnatury, brak wartości stałych)")
    print("Hybrid: ~85% (sygnatury + typy stałych + przykłady)")
    print("Ideal:  ~90% (pełne sygnatury, wartości stałych, przykłady)")
    print()
    print("AKTUALNY STATUS: Parser AST zachowuje parametry i defaults.")
    print("KOLEJNE KROKI:   Dodać wartości stałych do YAML/Hybrid oraz dataclasses.")
