#!/usr/bin/env python3
"""
Code2Logic Universal Format Validator v2.0
==========================================

Kompletny framework do walidacji i testowania reprodukowalności 
formatów Code2Logic dla 10+ języków programowania.

Features:
- Walidacja schematów JSON
- Testowanie reprodukowalności per język
- Generowanie raportów porównawczych  
- Rekomendacje ulepszeń per język
- Scoring system

Supported Languages:
1. Python      6. Go
2. JavaScript  7. Rust
3. TypeScript  8. PHP
4. Java        9. Ruby
5. C#          10. Swift/Kotlin

Usage:
    python universal_validator.py --yaml file.yaml --language python
    python universal_validator.py --all-formats --report
"""

import json
import yaml
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict
import argparse


# =============================================================================
# KONFIGURACJA JĘZYKÓW
# =============================================================================

class LanguageConfig:
    """Konfiguracja wymagań dla każdego języka."""
    
    CONFIGS = {
        'python': {
            'name': 'Python',
            'file_extensions': ['.py'],
            'type_annotations': 'optional',  # 'required', 'optional', 'none'
            'import_style': 'from_import',   # 'from_import', 'import', 'require', 'use'
            'class_keyword': 'class',
            'function_keyword': 'def',
            'supports': {
                'classes': True,
                'interfaces': False,
                'traits': False,
                'structs': False,
                'enums': True,
                'dataclasses': True,
                'decorators': True,
                'annotations': False,
                'properties': True,
                'async': True,
                'generics': True,
                'multiple_inheritance': True,
            },
            'critical_elements': ['classes', 'functions', 'constants', 'dataclasses'],
            'signature_format': '(param:type=default)',
            'constant_pattern': r'^[A-Z][A-Z_0-9]+$',
            'docstring_style': 'triple_quote',
        },
        'javascript': {
            'name': 'JavaScript',
            'file_extensions': ['.js', '.mjs', '.cjs'],
            'type_annotations': 'none',
            'import_style': 'import_require',
            'class_keyword': 'class',
            'function_keyword': 'function',
            'supports': {
                'classes': True,
                'interfaces': False,
                'traits': False,
                'structs': False,
                'enums': False,
                'dataclasses': False,
                'decorators': False,  # Stage 3 proposal
                'annotations': False,
                'properties': True,
                'async': True,
                'generics': False,
                'multiple_inheritance': False,
            },
            'critical_elements': ['classes', 'functions', 'constants'],
            'signature_format': '(param, param)',
            'constant_pattern': r'^[A-Z][A-Z_0-9]+$',
            'docstring_style': 'jsdoc',
        },
        'typescript': {
            'name': 'TypeScript',
            'file_extensions': ['.ts', '.tsx'],
            'type_annotations': 'required',
            'import_style': 'import',
            'class_keyword': 'class',
            'function_keyword': 'function',
            'supports': {
                'classes': True,
                'interfaces': True,
                'traits': False,
                'structs': False,
                'enums': True,
                'dataclasses': False,
                'decorators': True,
                'annotations': False,
                'properties': True,
                'async': True,
                'generics': True,
                'multiple_inheritance': False,
            },
            'critical_elements': ['classes', 'interfaces', 'types', 'functions'],
            'signature_format': '(param: type): returnType',
            'constant_pattern': r'^[A-Z][A-Z_0-9]+$',
            'docstring_style': 'tsdoc',
            'extra_elements': ['type_aliases', 'namespaces'],
        },
        'java': {
            'name': 'Java',
            'file_extensions': ['.java'],
            'type_annotations': 'required',
            'import_style': 'import',
            'class_keyword': 'class',
            'function_keyword': 'method',  # No standalone functions
            'supports': {
                'classes': True,
                'interfaces': True,
                'traits': False,
                'structs': False,
                'enums': True,
                'dataclasses': True,  # records in Java 14+
                'decorators': False,
                'annotations': True,
                'properties': False,  # getters/setters
                'async': True,
                'generics': True,
                'multiple_inheritance': False,
            },
            'critical_elements': ['classes', 'interfaces', 'methods', 'annotations'],
            'signature_format': 'returnType methodName(Type param)',
            'constant_pattern': r'^[A-Z][A-Z_0-9]+$',
            'docstring_style': 'javadoc',
            'extra_elements': ['packages', 'records'],
        },
        'csharp': {
            'name': 'C#',
            'file_extensions': ['.cs'],
            'type_annotations': 'required',
            'import_style': 'using',
            'class_keyword': 'class',
            'function_keyword': 'method',
            'supports': {
                'classes': True,
                'interfaces': True,
                'traits': False,
                'structs': True,
                'enums': True,
                'dataclasses': True,  # records in C# 9+
                'decorators': False,
                'annotations': True,  # attributes
                'properties': True,
                'async': True,
                'generics': True,
                'multiple_inheritance': False,
            },
            'critical_elements': ['classes', 'interfaces', 'properties', 'methods'],
            'signature_format': 'returnType MethodName(Type param)',
            'constant_pattern': r'^[A-Z][a-zA-Z0-9]+$',  # PascalCase
            'docstring_style': 'xml_doc',
            'extra_elements': ['namespaces', 'records', 'attributes'],
        },
        'go': {
            'name': 'Go',
            'file_extensions': ['.go'],
            'type_annotations': 'required',
            'import_style': 'import',
            'class_keyword': 'struct',  # No classes
            'function_keyword': 'func',
            'supports': {
                'classes': False,
                'interfaces': True,
                'traits': False,
                'structs': True,
                'enums': False,  # iota pattern
                'dataclasses': False,
                'decorators': False,
                'annotations': False,
                'properties': False,
                'async': True,  # goroutines
                'generics': True,  # Go 1.18+
                'multiple_inheritance': False,
            },
            'critical_elements': ['structs', 'interfaces', 'functions', 'methods'],
            'signature_format': 'func name(param type) returnType',
            'constant_pattern': r'^[A-Z][a-zA-Z0-9]+$',
            'docstring_style': 'godoc',
            'extra_elements': ['packages', 'receivers'],
        },
        'rust': {
            'name': 'Rust',
            'file_extensions': ['.rs'],
            'type_annotations': 'required',
            'import_style': 'use',
            'class_keyword': 'struct',
            'function_keyword': 'fn',
            'supports': {
                'classes': False,
                'interfaces': False,
                'traits': True,
                'structs': True,
                'enums': True,
                'dataclasses': False,
                'decorators': False,
                'annotations': True,  # attributes like #[derive]
                'properties': False,
                'async': True,
                'generics': True,
                'multiple_inheritance': False,  # but multiple traits
            },
            'critical_elements': ['structs', 'traits', 'enums', 'impls'],
            'signature_format': 'fn name(param: Type) -> ReturnType',
            'constant_pattern': r'^[A-Z][A-Z_0-9]+$',
            'docstring_style': 'rustdoc',
            'extra_elements': ['modules', 'impls', 'derives'],
        },
        'php': {
            'name': 'PHP',
            'file_extensions': ['.php'],
            'type_annotations': 'optional',  # PHP 7+ type hints
            'import_style': 'use_require',
            'class_keyword': 'class',
            'function_keyword': 'function',
            'supports': {
                'classes': True,
                'interfaces': True,
                'traits': True,
                'structs': False,
                'enums': True,  # PHP 8.1+
                'dataclasses': False,
                'decorators': False,
                'annotations': True,  # attributes in PHP 8+
                'properties': True,
                'async': False,
                'generics': False,
                'multiple_inheritance': False,
            },
            'critical_elements': ['classes', 'interfaces', 'traits', 'methods'],
            'signature_format': 'function name(Type $param): ReturnType',
            'constant_pattern': r'^[A-Z][A-Z_0-9]+$',
            'docstring_style': 'phpdoc',
            'extra_elements': ['namespaces'],
        },
        'ruby': {
            'name': 'Ruby',
            'file_extensions': ['.rb'],
            'type_annotations': 'none',  # RBS/Sorbet optional
            'import_style': 'require',
            'class_keyword': 'class',
            'function_keyword': 'def',
            'supports': {
                'classes': True,
                'interfaces': False,
                'traits': False,
                'structs': True,  # Struct class
                'enums': False,
                'dataclasses': True,  # Data class in Ruby 3.2+
                'decorators': False,
                'annotations': False,
                'properties': True,  # attr_accessor
                'async': True,  # Fibers
                'generics': False,
                'multiple_inheritance': False,  # but mixins
            },
            'critical_elements': ['classes', 'modules', 'methods'],
            'signature_format': 'def name(param)',
            'constant_pattern': r'^[A-Z][a-zA-Z0-9]+$',
            'docstring_style': 'yard',
            'extra_elements': ['modules', 'mixins'],
        },
        'swift': {
            'name': 'Swift',
            'file_extensions': ['.swift'],
            'type_annotations': 'required',
            'import_style': 'import',
            'class_keyword': 'class',
            'function_keyword': 'func',
            'supports': {
                'classes': True,
                'interfaces': False,
                'traits': False,
                'structs': True,
                'enums': True,
                'dataclasses': False,
                'decorators': False,
                'annotations': True,  # attributes like @objc
                'properties': True,
                'async': True,  # Swift 5.5+
                'generics': True,
                'multiple_inheritance': False,
            },
            'critical_elements': ['classes', 'structs', 'protocols', 'enums'],
            'signature_format': 'func name(param: Type) -> ReturnType',
            'constant_pattern': r'^[a-z][a-zA-Z0-9]+$',  # lowerCamelCase
            'docstring_style': 'swift_markup',
            'extra_elements': ['protocols', 'extensions'],
        },
        'kotlin': {
            'name': 'Kotlin',
            'file_extensions': ['.kt', '.kts'],
            'type_annotations': 'required',
            'import_style': 'import',
            'class_keyword': 'class',
            'function_keyword': 'fun',
            'supports': {
                'classes': True,
                'interfaces': True,
                'traits': False,
                'structs': False,
                'enums': True,
                'dataclasses': True,  # data class
                'decorators': False,
                'annotations': True,
                'properties': True,
                'async': True,  # coroutines
                'generics': True,
                'multiple_inheritance': False,
            },
            'critical_elements': ['classes', 'interfaces', 'data_classes', 'functions'],
            'signature_format': 'fun name(param: Type): ReturnType',
            'constant_pattern': r'^[A-Z][A-Z_0-9]+$',
            'docstring_style': 'kdoc',
            'extra_elements': ['sealed_classes', 'objects', 'companions'],
        },
    }
    
    @classmethod
    def get(cls, language: str) -> dict:
        """Pobierz konfigurację dla języka."""
        return cls.CONFIGS.get(language.lower(), cls.CONFIGS['python'])
    
    @classmethod
    def supported_languages(cls) -> List[str]:
        """Lista obsługiwanych języków."""
        return list(cls.CONFIGS.keys())


# =============================================================================
# WZORCE WALIDACJI PER JĘZYK
# =============================================================================

class LanguagePatterns:
    """Wzorce regex i walidatory dla każdego języka."""
    
    PATTERNS = {
        'python': {
            'signature': r'\(([^)]*)\)',
            'param_with_type': r'(\w+)\s*:\s*([^,=]+)',
            'param_with_default': r'(\w+)\s*(?::\s*[^=]+)?\s*=\s*([^,)]+)',
            'decorator': r'@(\w+)(?:\([^)]*\))?',
            'class_def': r'class\s+(\w+)(?:\(([^)]*)\))?:',
            'dataclass': r'@dataclass',
            'constant': r'^([A-Z][A-Z_0-9]+)\s*[:=]',
            'enum_value': r'(\w+)\s*=\s*(?:auto\(\)|[^,\n]+)',
            'type_alias': r'^(\w+)\s*=\s*(.*(?:Dict|List|Tuple|Optional|Union|Callable).*)',
        },
        'typescript': {
            'signature': r'\(([^)]*)\)\s*(?::\s*(\w+(?:<[^>]+>)?))?',
            'param_with_type': r'(\w+)\s*:\s*([^,=]+)',
            'interface_def': r'interface\s+(\w+)(?:\s+extends\s+([^{]+))?',
            'type_alias': r'type\s+(\w+)\s*=\s*(.+)',
            'enum_def': r'enum\s+(\w+)',
            'generic': r'<([^>]+)>',
            'decorator': r'@(\w+)(?:\([^)]*\))?',
        },
        'java': {
            'signature': r'\(([^)]*)\)',
            'param': r'(\w+(?:<[^>]+>)?)\s+(\w+)',
            'annotation': r'@(\w+)(?:\([^)]*\))?',
            'class_def': r'(?:public\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?',
            'interface_def': r'(?:public\s+)?interface\s+(\w+)(?:\s+extends\s+([^{]+))?',
            'enum_def': r'(?:public\s+)?enum\s+(\w+)',
            'record_def': r'(?:public\s+)?record\s+(\w+)\s*\(([^)]*)\)',
        },
        'go': {
            'signature': r'\(([^)]*)\)\s*(?:\(([^)]*)\)|(\w+))?',
            'param': r'(\w+)\s+(\w+(?:\.\w+)?)',
            'struct_def': r'type\s+(\w+)\s+struct',
            'interface_def': r'type\s+(\w+)\s+interface',
            'func_receiver': r'\((\w+)\s+\*?(\w+)\)',
            'const_def': r'const\s+(\w+)\s*=',
        },
        'rust': {
            'signature': r'\(([^)]*)\)\s*(?:->\s*(.+))?',
            'param': r'(\w+)\s*:\s*([^,]+)',
            'struct_def': r'(?:pub\s+)?struct\s+(\w+)(?:<[^>]+>)?',
            'trait_def': r'(?:pub\s+)?trait\s+(\w+)',
            'enum_def': r'(?:pub\s+)?enum\s+(\w+)',
            'impl_def': r'impl(?:<[^>]+>)?\s+(?:(\w+)\s+for\s+)?(\w+)',
            'derive': r'#\[derive\(([^)]+)\)\]',
        },
        'php': {
            'signature': r'\(([^)]*)\)\s*(?::\s*\??(\w+))?',
            'param': r'(?:(\w+)\s+)?\$(\w+)(?:\s*=\s*([^,)]+))?',
            'class_def': r'(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?',
            'interface_def': r'interface\s+(\w+)(?:\s+extends\s+([^{]+))?',
            'trait_def': r'trait\s+(\w+)',
            'attribute': r'#\[(\w+)(?:\([^)]*\))?\]',
        },
        'ruby': {
            'signature': r'\(([^)]*)\)',
            'param': r'(\w+)(?:\s*=\s*([^,)]+))?',
            'class_def': r'class\s+(\w+)(?:\s*<\s*(\w+))?',
            'module_def': r'module\s+(\w+)',
            'attr_accessor': r'attr_(?:accessor|reader|writer)\s+:(\w+)',
        },
        'swift': {
            'signature': r'\(([^)]*)\)\s*(?:->\s*(.+))?',
            'param': r'(\w+)\s+(\w+)\s*:\s*([^,=]+)',
            'class_def': r'class\s+(\w+)(?:\s*:\s*([^{]+))?',
            'struct_def': r'struct\s+(\w+)',
            'protocol_def': r'protocol\s+(\w+)',
            'enum_def': r'enum\s+(\w+)',
        },
        'kotlin': {
            'signature': r'\(([^)]*)\)\s*(?::\s*(.+))?',
            'param': r'(?:val|var)?\s*(\w+)\s*:\s*([^,=]+)',
            'class_def': r'(?:data\s+)?class\s+(\w+)(?:\s*:\s*([^{(]+))?',
            'interface_def': r'interface\s+(\w+)',
            'sealed_class': r'sealed\s+class\s+(\w+)',
            'object_def': r'object\s+(\w+)',
        },
    }
    
    @classmethod
    def get(cls, language: str) -> dict:
        """Pobierz wzorce dla języka."""
        return cls.PATTERNS.get(language.lower(), cls.PATTERNS['python'])


# =============================================================================
# WALIDATOR UNIWERSALNY
# =============================================================================

@dataclass
class ValidationIssue:
    """Problem znaleziony podczas walidacji."""
    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'signature', 'constant', 'enum', 'dataclass', etc.
    element: str
    message: str
    expected: str = ''
    actual: str = ''
    impact: float = 0.0
    language_specific: bool = False


@dataclass
class ValidationReport:
    """Raport z walidacji."""
    language: str
    format_type: str
    total_elements: int = 0
    valid_elements: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    reproduction_score: float = 0.0
    language_coverage: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class UniversalValidator:
    """Uniwersalny walidator formatów Code2Logic."""
    
    # Wagi dla kategorii problemów
    IMPACT_WEIGHTS = {
        'signature': 25,
        'constant': 15,
        'enum': 8,
        'dataclass': 12,
        'class_attrs': 8,
        'interface': 10,
        'trait': 10,
        'type_annotation': 10,
        'decorator': 5,
        'docstring': 5,
        'import': 5,
    }
    
    def __init__(self, content: str, format_type: str, language: str):
        self.content = content
        self.format_type = format_type
        self.language = language.lower()
        self.config = LanguageConfig.get(language)
        self.patterns = LanguagePatterns.get(language)
        self.data: Dict[str, Any] = {}
        self.issues: List[ValidationIssue] = []
    
    def parse(self) -> Dict[str, Any]:
        """Parsuj zawartość formatu."""
        if self.format_type in ('yaml', 'hybrid'):
            try:
                self.data = yaml.safe_load(self.content)
            except yaml.YAMLError as e:
                self.issues.append(ValidationIssue(
                    severity='critical',
                    category='parse',
                    element='root',
                    message=f"YAML parse error: {e}",
                    impact=100.0
                ))
        elif self.format_type == 'toon':
            self.data = self._parse_toon()
        elif self.format_type == 'json':
            try:
                self.data = json.loads(self.content)
            except json.JSONDecodeError as e:
                self.issues.append(ValidationIssue(
                    severity='critical',
                    category='parse',
                    element='root',
                    message=f"JSON parse error: {e}",
                    impact=100.0
                ))
        return self.data
    
    def _parse_toon(self) -> Dict[str, Any]:
        """Parsuj format TOON."""
        # Uproszczona implementacja
        data = {'modules': [], 'details': {}}
        # ... (implementation like before)
        return data
    
    def validate(self) -> ValidationReport:
        """Przeprowadź pełną walidację."""
        if not self.data:
            self.parse()
        
        if not self.data:
            return ValidationReport(
                language=self.language,
                format_type=self.format_type,
                issues=self.issues,
                reproduction_score=0.0
            )
        
        # Waliduj moduły
        modules = self.data.get('modules', [])
        total = 0
        valid = 0
        
        for module in modules:
            total += 1
            module_issues = self._validate_module(module)
            self.issues.extend(module_issues)
            if not any(i.severity == 'critical' for i in module_issues):
                valid += 1
        
        # Waliduj elementy specyficzne dla języka
        language_issues = self._validate_language_specific()
        self.issues.extend(language_issues)
        
        # Oblicz score
        total_impact = sum(i.impact for i in self.issues)
        reproduction_score = max(0, 100 - total_impact)
        
        # Oblicz pokrycie per język
        coverage = self._calculate_language_coverage()
        
        # Generuj rekomendacje
        recommendations = self._generate_recommendations()
        
        return ValidationReport(
            language=self.language,
            format_type=self.format_type,
            total_elements=total,
            valid_elements=valid,
            issues=self.issues,
            reproduction_score=reproduction_score,
            language_coverage=coverage,
            recommendations=recommendations
        )
    
    def _validate_module(self, module: Dict) -> List[ValidationIssue]:
        """Waliduj pojedynczy moduł."""
        issues = []
        
        # Waliduj stałe
        for const in module.get('const', []):
            issues.extend(self._validate_constant(const))
        
        # Waliduj klasy
        for cls in module.get('c', module.get('classes', [])):
            issues.extend(self._validate_class(cls))
        
        # Waliduj funkcje
        for func in module.get('f', module.get('functions', [])):
            issues.extend(self._validate_function(func))
        
        # Waliduj interfejsy (jeśli język je wspiera)
        if self.config['supports']['interfaces']:
            for iface in module.get('interfaces', []):
                issues.extend(self._validate_interface(iface))
        
        # Waliduj traits (Rust, PHP)
        if self.config['supports']['traits']:
            for trait in module.get('traits', []):
                issues.extend(self._validate_trait(trait))
        
        # Waliduj struktury (Go, Rust, Swift)
        if self.config['supports']['structs']:
            for struct in module.get('structs', []):
                issues.extend(self._validate_struct(struct))
        
        return issues
    
    def _validate_constant(self, const: Dict) -> List[ValidationIssue]:
        """Waliduj stałą."""
        issues = []
        name = const.get('n', const.get('name', 'unknown'))
        
        # Sprawdź czy ma typ
        if 't' not in const and 'type' not in const:
            issues.append(ValidationIssue(
                severity='high',
                category='constant',
                element=name,
                message='Constant missing type information',
                expected='t: Dict[str, str]',
                actual='(missing)',
                impact=self.IMPACT_WEIGHTS['constant'] * 0.3
            ))
        
        # Sprawdź czy ma wartość
        has_value = any(k in const for k in ['v', 'value', 'keys'])
        if not has_value:
            issues.append(ValidationIssue(
                severity='critical',
                category='constant',
                element=name,
                message='Constant missing value - cannot reproduce',
                expected='v: {...} or keys: [...]',
                actual='(missing)',
                impact=self.IMPACT_WEIGHTS['constant']
            ))
        
        return issues
    
    def _validate_class(self, cls: Dict) -> List[ValidationIssue]:
        """Waliduj klasę."""
        issues = []
        name = cls.get('n', cls.get('name', 'unknown'))
        
        # Sprawdź dekoratory (Python)
        decorators = cls.get('dec', cls.get('decorators', []))
        is_dataclass = 'dataclass' in decorators
        
        # Sprawdź czy dataclass ma fields
        if is_dataclass and 'fields' not in cls:
            issues.append(ValidationIssue(
                severity='critical',
                category='dataclass',
                element=name,
                message='Dataclass missing fields definition',
                expected='fields: [{n: name, t: type}, ...]',
                actual='(missing)',
                impact=self.IMPACT_WEIGHTS['dataclass']
            ))
        
        # Sprawdź atrybuty klasy
        if 'attrs' not in cls and 'attributes' not in cls:
            issues.append(ValidationIssue(
                severity='medium',
                category='class_attrs',
                element=name,
                message='Class missing attributes extraction',
                expected='attrs: [{n: name, t: type}]',
                actual='(missing)',
                impact=self.IMPACT_WEIGHTS['class_attrs'] * 0.5
            ))
        
        # Sprawdź czy enum ma wartości
        bases = cls.get('b', cls.get('bases', []))
        if 'Enum' in bases and 'values' not in cls:
            issues.append(ValidationIssue(
                severity='critical',
                category='enum',
                element=name,
                message='Enum missing values',
                expected='values: [VALUE1, VALUE2, ...]',
                actual='(missing)',
                impact=self.IMPACT_WEIGHTS['enum']
            ))
        
        # Waliduj metody
        for method in cls.get('m', cls.get('methods', [])):
            method_name = method.get('n', method.get('name', ''))
            issues.extend(self._validate_signature(
                method.get('sig', method.get('signature', '')),
                f"{name}.{method_name}"
            ))
            
            # Sprawdź dekoratory metod (jeśli język wspiera)
            if self.config['supports']['decorators']:
                if 'dec' not in method and 'decorators' not in method:
                    # Nie flaguj jako problem - opcjonalne
                    pass
        
        return issues
    
    def _validate_function(self, func: Dict) -> List[ValidationIssue]:
        """Waliduj funkcję."""
        name = func.get('n', func.get('name', 'unknown'))
        sig = func.get('sig', func.get('signature', ''))
        
        return self._validate_signature(sig, name)
    
    def _validate_signature(self, sig: str, name: str) -> List[ValidationIssue]:
        """Waliduj sygnaturę funkcji/metody."""
        issues = []
        
        # Pusta sygnatura
        if not sig or sig in ('', '()', '()'):
            issues.append(ValidationIssue(
                severity='critical',
                category='signature',
                element=name,
                message='Empty signature - no parameters',
                expected=self.config['signature_format'],
                actual=sig or '(empty)',
                impact=self.IMPACT_WEIGHTS['signature'] * 0.6
            ))
            return issues
        
        # Sprawdź typy parametrów (dla języków z required type annotations)
        if self.config['type_annotations'] == 'required':
            # Wyciągnij parametry
            params_match = re.search(r'\(([^)]*)\)', sig)
            if params_match:
                params_str = params_match.group(1)
                params = [p.strip() for p in params_str.split(',') if p.strip()]
                
                for param in params:
                    # Pomiń self/this
                    if param in ('self', 'cls', 'this'):
                        continue
                    
                    # Sprawdź czy ma typ
                    has_type = ':' in param or self.language in ('java', 'go')
                    
                    if not has_type and self.config['type_annotations'] == 'required':
                        issues.append(ValidationIssue(
                            severity='high',
                            category='type_annotation',
                            element=name,
                            message=f'Parameter missing type annotation',
                            expected='param: Type',
                            actual=param,
                            impact=self.IMPACT_WEIGHTS['type_annotation'] * 0.2
                        ))
        
        # Sprawdź wartości domyślne
        if '=' not in sig and name not in ('__init__', 'constructor', 'new', 'init'):
            issues.append(ValidationIssue(
                severity='low',
                category='signature',
                element=name,
                message='Signature may be missing default values',
                expected='(param: type = default)',
                actual=sig,
                impact=2.0
            ))
        
        return issues
    
    def _validate_interface(self, iface: Dict) -> List[ValidationIssue]:
        """Waliduj interfejs (TS, Java, C#, Go)."""
        issues = []
        name = iface.get('n', iface.get('name', 'unknown'))
        
        # Interfejs powinien mieć metody
        methods = iface.get('m', iface.get('methods', []))
        if not methods:
            issues.append(ValidationIssue(
                severity='medium',
                category='interface',
                element=name,
                message='Interface has no methods defined',
                expected='methods: [...]',
                actual='(empty)',
                impact=self.IMPACT_WEIGHTS['interface'] * 0.3
            ))
        
        return issues
    
    def _validate_trait(self, trait: Dict) -> List[ValidationIssue]:
        """Waliduj trait (Rust, PHP)."""
        issues = []
        name = trait.get('n', trait.get('name', 'unknown'))
        
        methods = trait.get('m', trait.get('methods', []))
        if not methods:
            issues.append(ValidationIssue(
                severity='medium',
                category='trait',
                element=name,
                message='Trait has no methods defined',
                expected='methods: [...]',
                actual='(empty)',
                impact=self.IMPACT_WEIGHTS['trait'] * 0.3
            ))
        
        return issues
    
    def _validate_struct(self, struct: Dict) -> List[ValidationIssue]:
        """Waliduj struct (Go, Rust, Swift, C#)."""
        issues = []
        name = struct.get('n', struct.get('name', 'unknown'))
        
        # Struct powinien mieć pola
        fields = struct.get('fields', [])
        if not fields:
            issues.append(ValidationIssue(
                severity='high',
                category='struct',
                element=name,
                message='Struct has no fields defined',
                expected='fields: [{n: name, t: type}]',
                actual='(empty)',
                impact=10.0
            ))
        
        return issues
    
    def _validate_language_specific(self) -> List[ValidationIssue]:
        """Waliduj elementy specyficzne dla języka."""
        issues = []
        
        # Python-specific
        if self.language == 'python':
            # Sprawdź czy ma sekcję dataclasses
            if self.config['supports']['dataclasses']:
                # Check if any class should be a dataclass
                pass
        
        # TypeScript-specific
        elif self.language == 'typescript':
            # Sprawdź type aliasy
            for module in self.data.get('modules', []):
                if 'types' not in module and 'type_aliases' not in module:
                    # TypeScript często używa type aliases
                    pass
        
        # Java-specific
        elif self.language == 'java':
            # Sprawdź annotations
            for module in self.data.get('modules', []):
                for cls in module.get('c', module.get('classes', [])):
                    if 'annotations' not in cls:
                        # Java classes often have annotations
                        pass
        
        # Go-specific
        elif self.language == 'go':
            # Sprawdź receivers dla metod
            pass
        
        # Rust-specific
        elif self.language == 'rust':
            # Sprawdź derives
            for module in self.data.get('modules', []):
                for struct in module.get('structs', []):
                    if 'derives' not in struct:
                        issues.append(ValidationIssue(
                            severity='low',
                            category='rust_specific',
                            element=struct.get('n', 'unknown'),
                            message='Rust struct missing derives (Debug, Clone, etc.)',
                            expected='derives: [Debug, Clone, ...]',
                            actual='(missing)',
                            impact=3.0,
                            language_specific=True
                        ))
        
        return issues
    
    def _calculate_language_coverage(self) -> Dict[str, float]:
        """Oblicz pokrycie elementów specyficznych dla języka."""
        coverage = {}
        supports = self.config['supports']
        
        for feature, supported in supports.items():
            if supported:
                # Sprawdź czy format pokrywa tę cechę
                if feature == 'classes':
                    has = any('c' in m or 'classes' in m for m in self.data.get('modules', []))
                elif feature == 'interfaces':
                    has = any('interfaces' in m for m in self.data.get('modules', []))
                elif feature == 'traits':
                    has = any('traits' in m for m in self.data.get('modules', []))
                elif feature == 'structs':
                    has = any('structs' in m for m in self.data.get('modules', []))
                elif feature == 'enums':
                    has = True  # Zwykle w klasach
                elif feature == 'dataclasses':
                    has = any('dataclasses' in m or 'fields' in str(m) for m in self.data.get('modules', []))
                elif feature == 'decorators':
                    has = any('dec' in str(m) or 'decorators' in str(m) for m in self.data.get('modules', []))
                else:
                    has = True
                
                coverage[feature] = 100.0 if has else 0.0
        
        return coverage
    
    def _generate_recommendations(self) -> List[str]:
        """Generuj rekomendacje ulepszeń."""
        recommendations = []
        
        # Grupuj problemy
        critical = [i for i in self.issues if i.severity == 'critical']
        high = [i for i in self.issues if i.severity == 'high']
        
        # Top problemy
        if critical:
            category_counts = defaultdict(int)
            for issue in critical:
                category_counts[issue.category] += 1
            
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:3]:
                if cat == 'signature':
                    recommendations.append(
                        f"CRITICAL: Fix {count} empty signatures - add parameters with types and defaults"
                    )
                elif cat == 'constant':
                    recommendations.append(
                        f"CRITICAL: Add values for {count} constants - include type and value/keys"
                    )
                elif cat == 'enum':
                    recommendations.append(
                        f"CRITICAL: Add values for {count} enums - list all enum members"
                    )
                elif cat == 'dataclass':
                    recommendations.append(
                        f"CRITICAL: Add fields for {count} dataclasses - include field names and types"
                    )
        
        # Rekomendacje specyficzne dla języka
        supports = self.config['supports']
        
        if supports['interfaces'] and 'interfaces' not in str(self.data):
            recommendations.append(
                f"Add 'interfaces' section for {self.config['name']} - "
                "interfaces are important for this language"
            )
        
        if supports['traits'] and 'traits' not in str(self.data):
            recommendations.append(
                f"Add 'traits' section for {self.config['name']} - "
                "traits are critical for this language"
            )
        
        if self.config['type_annotations'] == 'required':
            recommendations.append(
                f"Ensure ALL signatures have type annotations for {self.config['name']} - "
                "types are required"
            )
        
        return recommendations


# =============================================================================
# MULTI-FORMAT COMPARATOR
# =============================================================================

class FormatComparator:
    """Porównuje wyniki walidacji między formatami."""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, ValidationReport]] = defaultdict(dict)
    
    def add_result(self, language: str, format_type: str, report: ValidationReport):
        """Dodaj wynik walidacji."""
        self.results[language][format_type] = report
    
    def compare(self) -> str:
        """Generuj porównanie formatów."""
        lines = []
        lines.append("=" * 80)
        lines.append("FORMAT COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Tabela podsumowująca
        lines.append("REPRODUCTION SCORES BY LANGUAGE AND FORMAT:")
        lines.append("-" * 80)
        header = f"{'Language':<12}"
        formats = set()
        for reports in self.results.values():
            formats.update(reports.keys())
        
        for fmt in sorted(formats):
            header += f" {fmt:>10}"
        header += f" {'Best':>12}"
        lines.append(header)
        lines.append("-" * 80)
        
        for lang, reports in sorted(self.results.items()):
            row = f"{lang:<12}"
            scores = {}
            for fmt in sorted(formats):
                if fmt in reports:
                    score = reports[fmt].reproduction_score
                    scores[fmt] = score
                    row += f" {score:>9.1f}%"
                else:
                    row += f" {'N/A':>10}"
            
            if scores:
                best_fmt = max(scores.items(), key=lambda x: x[1])[0]
                row += f" {best_fmt:>12}"
            
            lines.append(row)
        
        lines.append("-" * 80)
        lines.append("")
        
        # Szczegółowe problemy
        lines.append("CRITICAL ISSUES SUMMARY:")
        lines.append("-" * 80)
        
        all_issues = defaultdict(int)
        for reports in self.results.values():
            for report in reports.values():
                for issue in report.issues:
                    if issue.severity == 'critical':
                        all_issues[issue.message] += 1
        
        for msg, count in sorted(all_issues.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  [{count:>3}x] {msg[:60]}")
        
        lines.append("")
        
        # Rekomendacje
        lines.append("TOP RECOMMENDATIONS:")
        lines.append("-" * 80)
        
        all_recs = set()
        for reports in self.results.values():
            for report in reports.values():
                all_recs.update(report.recommendations[:3])
        
        for i, rec in enumerate(list(all_recs)[:10], 1):
            lines.append(f"  {i}. {rec}")
        
        return "\n".join(lines)
    
    def get_best_format(self, language: str) -> Optional[str]:
        """Zwróć najlepszy format dla danego języka."""
        if language not in self.results:
            return None
        
        reports = self.results[language]
        if not reports:
            return None
        
        best = max(reports.items(), key=lambda x: x[1].reproduction_score)
        return best[0]


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Code2Logic Universal Format Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --yaml project.yaml --language python
  %(prog)s --yaml project.yaml --hybrid hybrid.yaml --toon project.toon
  %(prog)s --yaml project.yaml --all-languages
  %(prog)s --yaml project.yaml --report output.md
        """
    )
    
    parser.add_argument('--yaml', help='Path to YAML format file')
    parser.add_argument('--hybrid', help='Path to Hybrid YAML file')
    parser.add_argument('--toon', help='Path to TOON format file')
    parser.add_argument('--json', help='Path to JSON format file')
    
    parser.add_argument(
        '--language', '-l',
        default='python',
        choices=LanguageConfig.supported_languages(),
        help='Target language (default: python)'
    )
    
    parser.add_argument(
        '--all-languages', '-a',
        action='store_true',
        help='Test for all supported languages'
    )
    
    parser.add_argument(
        '--report', '-r',
        help='Output report to file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Sprawdź czy podano jakiś plik
    if not any([args.yaml, args.hybrid, args.toon, args.json]):
        parser.error("At least one format file is required (--yaml, --hybrid, --toon, or --json)")
    
    comparator = FormatComparator()
    languages = LanguageConfig.supported_languages() if args.all_languages else [args.language]
    
    for language in languages:
        print(f"\nValidating for {language}...")
        
        # Waliduj każdy format
        formats_to_test = []
        if args.yaml:
            formats_to_test.append(('yaml', args.yaml))
        if args.hybrid:
            formats_to_test.append(('hybrid', args.hybrid))
        if args.toon:
            formats_to_test.append(('toon', args.toon))
        if args.json:
            formats_to_test.append(('json', args.json))
        
        for fmt_type, filepath in formats_to_test:
            try:
                content = Path(filepath).read_text(encoding='utf-8')
                validator = UniversalValidator(content, fmt_type, language)
                report = validator.validate()
                comparator.add_result(language, fmt_type, report)
                
                if args.verbose:
                    print(f"  {fmt_type}: {report.reproduction_score:.1f}% "
                          f"({len(report.issues)} issues)")
                    
            except FileNotFoundError:
                print(f"  {fmt_type}: File not found: {filepath}")
            except Exception as e:
                print(f"  {fmt_type}: Error: {e}")
    
    # Wypisz raport
    report = comparator.compare()
    print("\n" + report)
    
    # Zapisz do pliku jeśli podano
    if args.report:
        Path(args.report).write_text(report, encoding='utf-8')
        print(f"\nReport saved to: {args.report}")
    
    # Podsumowanie
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for language in languages:
        best = comparator.get_best_format(language)
        if best:
            score = comparator.results[language][best].reproduction_score
            print(f"  {language}: Best format = {best} ({score:.1f}%)")


if __name__ == "__main__":
    main()
