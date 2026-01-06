#!/usr/bin/env python3
"""
Code2Logic Format Validator - Multi-Language Reproduction Test Framework

Uniwersalny framework do testowania reprodukowalności formatów Code2Logic
dla 10 najpopularniejszych języków programowania.

Obsługiwane języki:
1. Python      6. Go
2. JavaScript  7. Rust  
3. TypeScript  8. PHP
4. Java        9. Ruby
5. C#          10. Swift/Kotlin

Autor: Code2Logic Team
"""

import json
import yaml
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict


# =============================================================================
# ENUMS I TYPY
# =============================================================================

class Language(Enum):
    """Obsługiwane języki programowania."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"


class ElementType(Enum):
    """Typy elementów kodu."""
    MODULE = auto()
    CLASS = auto()
    INTERFACE = auto()
    TRAIT = auto()
    STRUCT = auto()
    ENUM = auto()
    FUNCTION = auto()
    METHOD = auto()
    PROPERTY = auto()
    CONSTANT = auto()
    VARIABLE = auto()
    TYPE_ALIAS = auto()
    DECORATOR = auto()
    ANNOTATION = auto()
    IMPORT = auto()
    EXPORT = auto()
    DATACLASS = auto()
    PROTOCOL = auto()


class Severity(Enum):
    """Poziom ważności brakującej informacji."""
    CRITICAL = "critical"      # Blokuje reprodukcję
    HIGH = "high"              # Znacząco wpływa na jakość
    MEDIUM = "medium"          # Wpływa na szczegóły
    LOW = "low"                # Kosmetyczne


# =============================================================================
# MODELE DANYCH
# =============================================================================

@dataclass
class ValidationIssue:
    """Problem znaleziony podczas walidacji."""
    severity: Severity
    element_type: ElementType
    element_name: str
    issue: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    impact_percent: float = 0.0  # Szacowany wpływ na reprodukcję


@dataclass
class LanguageFeature:
    """Cecha języka programowania."""
    name: str
    element_type: ElementType
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    examples: Dict[str, str] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Wynik walidacji formatu."""
    language: Language
    format_name: str
    total_elements: int
    valid_elements: int
    issues: List[ValidationIssue] = field(default_factory=list)
    reproduction_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def validity_percent(self) -> float:
        if self.total_elements == 0:
            return 0.0
        return (self.valid_elements / self.total_elements) * 100


@dataclass 
class LanguageRequirements:
    """Wymagania dla danego języka."""
    language: Language
    features: List[LanguageFeature]
    critical_elements: Set[ElementType]
    import_style: str  # 'module', 'package', 'require', 'use'
    class_style: str   # 'class', 'struct', 'type'
    function_style: str  # 'def', 'function', 'fn', 'func'
    supports_interfaces: bool = False
    supports_traits: bool = False
    supports_generics: bool = True
    supports_decorators: bool = False
    supports_annotations: bool = False
    supports_dataclasses: bool = False
    supports_enums: bool = True
    supports_async: bool = False
    supports_properties: bool = False
    type_annotations: str = "optional"  # 'required', 'optional', 'none'


# =============================================================================
# DEFINICJE JĘZYKÓW
# =============================================================================

LANGUAGE_REQUIREMENTS: Dict[Language, LanguageRequirements] = {
    
    Language.PYTHON: LanguageRequirements(
        language=Language.PYTHON,
        features=[
            LanguageFeature(
                name="dataclass",
                element_type=ElementType.DATACLASS,
                required_fields=["name", "fields"],
                optional_fields=["decorators", "docstring", "methods"],
                examples={"Intent": "@dataclass\nclass Intent:\n    type: IntentType"}
            ),
            LanguageFeature(
                name="enum",
                element_type=ElementType.ENUM,
                required_fields=["name", "values"],
                optional_fields=["bases", "docstring"],
                examples={"IntentType": "class IntentType(Enum):\n    REFACTOR = auto()"}
            ),
            LanguageFeature(
                name="constant",
                element_type=ElementType.CONSTANT,
                required_fields=["name", "type"],
                optional_fields=["value", "keys"],
                examples={"TYPE_MAP": "TYPE_MAP: Dict[str, str] = {...}"}
            ),
        ],
        critical_elements={ElementType.CLASS, ElementType.FUNCTION, ElementType.CONSTANT},
        import_style="from X import Y",
        class_style="class",
        function_style="def",
        supports_decorators=True,
        supports_dataclasses=True,
        supports_async=True,
        supports_properties=True,
        type_annotations="optional",
    ),
    
    Language.JAVASCRIPT: LanguageRequirements(
        language=Language.JAVASCRIPT,
        features=[
            LanguageFeature(
                name="class",
                element_type=ElementType.CLASS,
                required_fields=["name", "methods"],
                optional_fields=["extends", "constructor"],
                examples={"Parser": "class Parser extends Base { }"}
            ),
            LanguageFeature(
                name="constant",
                element_type=ElementType.CONSTANT,
                required_fields=["name"],
                optional_fields=["value"],
                examples={"CONFIG": "const CONFIG = { ... }"}
            ),
        ],
        critical_elements={ElementType.CLASS, ElementType.FUNCTION, ElementType.CONSTANT},
        import_style="import/require",
        class_style="class",
        function_style="function",
        supports_async=True,
        type_annotations="none",
    ),
    
    Language.TYPESCRIPT: LanguageRequirements(
        language=Language.TYPESCRIPT,
        features=[
            LanguageFeature(
                name="interface",
                element_type=ElementType.INTERFACE,
                required_fields=["name", "properties"],
                optional_fields=["extends", "methods"],
                examples={"IParser": "interface IParser { parse(): Result }"}
            ),
            LanguageFeature(
                name="type_alias",
                element_type=ElementType.TYPE_ALIAS,
                required_fields=["name", "definition"],
                examples={"Result": "type Result = Success | Error"}
            ),
            LanguageFeature(
                name="enum",
                element_type=ElementType.ENUM,
                required_fields=["name", "values"],
                examples={"Status": "enum Status { OK, ERROR }"}
            ),
        ],
        critical_elements={ElementType.CLASS, ElementType.INTERFACE, ElementType.FUNCTION},
        import_style="import",
        class_style="class",
        function_style="function",
        supports_interfaces=True,
        supports_async=True,
        type_annotations="required",
    ),
    
    Language.JAVA: LanguageRequirements(
        language=Language.JAVA,
        features=[
            LanguageFeature(
                name="interface",
                element_type=ElementType.INTERFACE,
                required_fields=["name", "methods"],
                optional_fields=["extends"],
                examples={"Parser": "public interface Parser { void parse(); }"}
            ),
            LanguageFeature(
                name="annotation",
                element_type=ElementType.ANNOTATION,
                required_fields=["name"],
                optional_fields=["parameters"],
                examples={"Override": "@Override"}
            ),
            LanguageFeature(
                name="enum",
                element_type=ElementType.ENUM,
                required_fields=["name", "values"],
                optional_fields=["methods"],
                examples={"Status": "enum Status { OK, ERROR }"}
            ),
        ],
        critical_elements={ElementType.CLASS, ElementType.INTERFACE, ElementType.METHOD},
        import_style="import",
        class_style="class",
        function_style="method",
        supports_interfaces=True,
        supports_annotations=True,
        type_annotations="required",
    ),
    
    Language.CSHARP: LanguageRequirements(
        language=Language.CSHARP,
        features=[
            LanguageFeature(
                name="interface",
                element_type=ElementType.INTERFACE,
                required_fields=["name", "methods"],
                examples={"IParser": "interface IParser { void Parse(); }"}
            ),
            LanguageFeature(
                name="property",
                element_type=ElementType.PROPERTY,
                required_fields=["name", "type"],
                optional_fields=["getter", "setter"],
                examples={"Name": "public string Name { get; set; }"}
            ),
            LanguageFeature(
                name="record",
                element_type=ElementType.DATACLASS,
                required_fields=["name", "properties"],
                examples={"Person": "record Person(string Name, int Age);"}
            ),
        ],
        critical_elements={ElementType.CLASS, ElementType.INTERFACE, ElementType.PROPERTY},
        import_style="using",
        class_style="class",
        function_style="method",
        supports_interfaces=True,
        supports_properties=True,
        supports_async=True,
        type_annotations="required",
    ),
    
    Language.GO: LanguageRequirements(
        language=Language.GO,
        features=[
            LanguageFeature(
                name="struct",
                element_type=ElementType.STRUCT,
                required_fields=["name", "fields"],
                examples={"Config": "type Config struct {\n    Name string\n}"}
            ),
            LanguageFeature(
                name="interface",
                element_type=ElementType.INTERFACE,
                required_fields=["name", "methods"],
                examples={"Parser": "type Parser interface {\n    Parse() error\n}"}
            ),
            LanguageFeature(
                name="constant",
                element_type=ElementType.CONSTANT,
                required_fields=["name", "type", "value"],
                examples={"MaxSize": "const MaxSize = 1024"}
            ),
        ],
        critical_elements={ElementType.STRUCT, ElementType.INTERFACE, ElementType.FUNCTION},
        import_style="import",
        class_style="struct",
        function_style="func",
        supports_interfaces=True,
        type_annotations="required",
    ),
    
    Language.RUST: LanguageRequirements(
        language=Language.RUST,
        features=[
            LanguageFeature(
                name="struct",
                element_type=ElementType.STRUCT,
                required_fields=["name", "fields"],
                optional_fields=["derives", "visibility"],
                examples={"Config": "#[derive(Debug)]\nstruct Config { name: String }"}
            ),
            LanguageFeature(
                name="trait",
                element_type=ElementType.TRAIT,
                required_fields=["name", "methods"],
                examples={"Parser": "trait Parser { fn parse(&self) -> Result<T, E>; }"}
            ),
            LanguageFeature(
                name="enum",
                element_type=ElementType.ENUM,
                required_fields=["name", "variants"],
                optional_fields=["derives"],
                examples={"Result": "enum Result<T, E> { Ok(T), Err(E) }"}
            ),
            LanguageFeature(
                name="impl",
                element_type=ElementType.METHOD,
                required_fields=["for_type", "methods"],
                examples={"Config": "impl Config { fn new() -> Self { } }"}
            ),
        ],
        critical_elements={ElementType.STRUCT, ElementType.TRAIT, ElementType.ENUM},
        import_style="use",
        class_style="struct",
        function_style="fn",
        supports_traits=True,
        type_annotations="required",
    ),
    
    Language.PHP: LanguageRequirements(
        language=Language.PHP,
        features=[
            LanguageFeature(
                name="interface",
                element_type=ElementType.INTERFACE,
                required_fields=["name", "methods"],
                examples={"ParserInterface": "interface ParserInterface { public function parse(); }"}
            ),
            LanguageFeature(
                name="trait",
                element_type=ElementType.TRAIT,
                required_fields=["name", "methods"],
                examples={"Loggable": "trait Loggable { public function log($msg); }"}
            ),
        ],
        critical_elements={ElementType.CLASS, ElementType.INTERFACE, ElementType.METHOD},
        import_style="use/require",
        class_style="class",
        function_style="function",
        supports_interfaces=True,
        supports_traits=True,
        type_annotations="optional",
    ),
    
    Language.RUBY: LanguageRequirements(
        language=Language.RUBY,
        features=[
            LanguageFeature(
                name="module",
                element_type=ElementType.MODULE,
                required_fields=["name"],
                optional_fields=["methods", "constants"],
                examples={"Parsers": "module Parsers\n  def parse\n  end\nend"}
            ),
            LanguageFeature(
                name="attr_accessor",
                element_type=ElementType.PROPERTY,
                required_fields=["name"],
                examples={"name": "attr_accessor :name"}
            ),
        ],
        critical_elements={ElementType.CLASS, ElementType.MODULE, ElementType.METHOD},
        import_style="require",
        class_style="class",
        function_style="def",
        type_annotations="none",
    ),
    
    Language.SWIFT: LanguageRequirements(
        language=Language.SWIFT,
        features=[
            LanguageFeature(
                name="protocol",
                element_type=ElementType.PROTOCOL,
                required_fields=["name", "requirements"],
                examples={"Parseable": "protocol Parseable { func parse() -> Result }"}
            ),
            LanguageFeature(
                name="struct",
                element_type=ElementType.STRUCT,
                required_fields=["name", "properties"],
                examples={"Config": "struct Config { let name: String }"}
            ),
            LanguageFeature(
                name="enum",
                element_type=ElementType.ENUM,
                required_fields=["name", "cases"],
                optional_fields=["associated_values"],
                examples={"Result": "enum Result { case success(T), case failure(E) }"}
            ),
        ],
        critical_elements={ElementType.CLASS, ElementType.STRUCT, ElementType.PROTOCOL},
        import_style="import",
        class_style="class",
        function_style="func",
        supports_properties=True,
        type_annotations="required",
    ),
    
    Language.KOTLIN: LanguageRequirements(
        language=Language.KOTLIN,
        features=[
            LanguageFeature(
                name="data_class",
                element_type=ElementType.DATACLASS,
                required_fields=["name", "properties"],
                examples={"Person": "data class Person(val name: String, val age: Int)"}
            ),
            LanguageFeature(
                name="sealed_class",
                element_type=ElementType.CLASS,
                required_fields=["name", "subclasses"],
                examples={"Result": "sealed class Result { data class Ok(val v: T) : Result() }"}
            ),
            LanguageFeature(
                name="interface",
                element_type=ElementType.INTERFACE,
                required_fields=["name", "methods"],
                examples={"Parser": "interface Parser { fun parse(): Result }"}
            ),
        ],
        critical_elements={ElementType.CLASS, ElementType.INTERFACE, ElementType.DATACLASS},
        import_style="import",
        class_style="class",
        function_style="fun",
        supports_interfaces=True,
        supports_properties=True,
        supports_async=True,
        type_annotations="required",
    ),
}


# =============================================================================
# WALIDATORY FORMATÓW
# =============================================================================

class FormatValidator(ABC):
    """Bazowa klasa walidatora formatu."""
    
    def __init__(self, content: str, language: Language):
        self.content = content
        self.language = language
        self.requirements = LANGUAGE_REQUIREMENTS.get(language)
        self.issues: List[ValidationIssue] = []
        
    @abstractmethod
    def parse(self) -> Dict[str, Any]:
        """Parsuj zawartość formatu."""
        pass
    
    @abstractmethod
    def validate(self) -> ValidationResult:
        """Waliduj format i zwróć wynik."""
        pass
    
    def _check_signature(self, sig: str, func_name: str) -> List[ValidationIssue]:
        """Sprawdź czy sygnatura jest kompletna."""
        issues = []
        
        # Pusta sygnatura
        if sig is None or sig == '':
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                element_type=ElementType.FUNCTION,
                element_name=func_name,
                issue="Empty signature - no parameters",
                expected="(param1:type1, param2:type2=default)",
                actual=sig,
                impact_percent=15.0
            ))
            return issues
        
        # Brak typów parametrów (dla języków z required type annotations)
        if self.requirements and self.requirements.type_annotations == "required":
            params = sig.strip('()').split(',')
            for param in params:
                param = param.strip()
                if param and ':' not in param and '=' not in param:
                    issues.append(ValidationIssue(
                        severity=Severity.HIGH,
                        element_type=ElementType.FUNCTION,
                        element_name=func_name,
                        issue=f"Parameter '{param}' missing type annotation",
                        expected=f"{param}:type",
                        actual=param,
                        impact_percent=5.0
                    ))
        
        # Brak wartości domyślnych (tylko jeśli są parametry)
        inner = sig.strip()[1:-1].strip() if sig.strip().startswith('(') and sig.strip().endswith(')') else sig
        if inner and '=' not in sig and func_name not in ('__init__', 'constructor', 'new'):
            issues.append(ValidationIssue(
                severity=Severity.MEDIUM,
                element_type=ElementType.FUNCTION,
                element_name=func_name,
                issue="Signature may be missing default values",
                expected="(param:type=default)",
                actual=sig,
                impact_percent=3.0
            ))
        
        return issues
    
    def _check_constant(self, const: Dict[str, Any]) -> List[ValidationIssue]:
        """Sprawdź czy stała ma wymagane informacje."""
        issues = []
        name = const.get('n', const.get('name', 'unknown'))
        
        # Tylko nazwa bez typu i wartości
        if 't' not in const and 'type' not in const:
            issues.append(ValidationIssue(
                severity=Severity.HIGH,
                element_type=ElementType.CONSTANT,
                element_name=name,
                issue="Constant missing type information",
                expected="t: Dict[str, str]",
                actual="t: (missing)",
                impact_percent=5.0
            ))
        
        # Brak wartości lub kluczy
        has_value = 'v' in const or 'value' in const
        has_keys = 'keys' in const
        
        if not has_value and not has_keys:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                element_type=ElementType.CONSTANT,
                element_name=name,
                issue="Constant missing value - cannot reproduce",
                expected="v: {...} or keys: [...]",
                actual="(missing)",
                impact_percent=10.0
            ))
        
        return issues
    
    def _check_enum(self, enum_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Sprawdź czy enum ma wartości."""
        issues = []
        name = enum_data.get('n', enum_data.get('name', 'unknown'))
        
        has_values = 'values' in enum_data or 'v' in enum_data or 'variants' in enum_data
        
        if not has_values:
            issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                element_type=ElementType.ENUM,
                element_name=name,
                issue="Enum missing values - cannot reproduce",
                expected="values: [VALUE1, VALUE2, ...]",
                actual="(missing)",
                impact_percent=8.0
            ))
        
        return issues
    
    def _check_class(self, cls_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Sprawdź klasę pod kątem kompletności."""
        issues = []
        name = cls_data.get('n', cls_data.get('name', 'unknown'))
        
        # Sprawdź czy to dataclass
        decorators = cls_data.get('dec', cls_data.get('decorators', []))
        is_dataclass = 'dataclass' in decorators or cls_data.get('is_dataclass', False)
        
        if is_dataclass:
            # Dataclass powinien mieć fields
            if 'fields' not in cls_data:
                issues.append(ValidationIssue(
                    severity=Severity.CRITICAL,
                    element_type=ElementType.DATACLASS,
                    element_name=name,
                    issue="Dataclass missing fields definition",
                    expected="fields: [{n: field_name, t: field_type}, ...]",
                    actual="(missing)",
                    impact_percent=12.0
                ))
        
        # Sprawdź atrybuty klasy
        if 'attrs' not in cls_data and 'attributes' not in cls_data:
            issues.append(ValidationIssue(
                severity=Severity.MEDIUM,
                element_type=ElementType.CLASS,
                element_name=name,
                issue="Class missing attributes (self.x = ...) extraction",
                expected="attrs: [{n: attr_name, t: attr_type}, ...]",
                actual="(missing)",
                impact_percent=5.0
            ))
        
        # Sprawdź metody
        methods = cls_data.get('m', cls_data.get('methods', []))
        for method in methods:
            method_name = method.get('n', method.get('name', ''))
            sig = method.get('sig', method.get('signature', ''))
            issues.extend(self._check_signature(sig, f"{name}.{method_name}"))
            
            # Sprawdź dekoratory metod
            if 'dec' not in method and 'decorators' not in method:
                if method_name in ('__str__', '__repr__', '__eq__', '__hash__'):
                    pass  # Standardowe metody
                elif method_name.startswith('_') and not method_name.startswith('__'):
                    pass  # Prywatne metody
        
        return issues


class YAMLValidator(FormatValidator):
    """Walidator dla formatu YAML."""
    
    def __init__(self, content: str, language: Language):
        super().__init__(content, language)
        self.data: Dict[str, Any] = {}
    
    def parse(self) -> Dict[str, Any]:
        """Parsuj YAML."""
        try:
            self.data = yaml.safe_load(self.content)
            return self.data
        except yaml.YAMLError as e:
            self.issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                element_type=ElementType.MODULE,
                element_name="root",
                issue=f"YAML parse error: {e}",
                impact_percent=100.0
            ))
            return {}
    
    def validate(self) -> ValidationResult:
        """Waliduj format YAML."""
        if not self.data:
            self.parse()
        
        if not self.data:
            return ValidationResult(
                language=self.language,
                format_name="YAML",
                total_elements=0,
                valid_elements=0,
                issues=self.issues,
                reproduction_score=0.0
            )
        
        total = 0
        valid = 0
        
        modules = self.data.get('modules', [])
        
        for module in modules:
            total += 1
            module_valid = True
            
            # Sprawdź stałe
            constants = module.get('const', [])
            for const in constants:
                total += 1
                const_issues = self._check_constant(const)
                if const_issues:
                    self.issues.extend(const_issues)
                    module_valid = False
                else:
                    valid += 1
            
            # Sprawdź klasy
            classes = module.get('c', module.get('classes', []))
            for cls in classes:
                total += 1
                cls_issues = self._check_class(cls)
                
                # Sprawdź czy to enum
                bases = cls.get('b', cls.get('bases', []))
                if 'Enum' in bases:
                    cls_issues.extend(self._check_enum(cls))
                
                if cls_issues:
                    self.issues.extend(cls_issues)
                else:
                    valid += 1
            
            # Sprawdź funkcje
            functions = module.get('f', module.get('functions', []))
            for func in functions:
                total += 1
                func_name = func.get('n', func.get('name', ''))
                sig = func.get('sig', func.get('signature', ''))
                func_issues = self._check_signature(sig, func_name)
                
                if func_issues:
                    self.issues.extend(func_issues)
                else:
                    valid += 1
            
            if module_valid:
                valid += 1
        
        # Oblicz score reprodukcji
        total_impact = sum(issue.impact_percent for issue in self.issues)
        normalized_impact = total_impact / max(total, 1)
        reproduction_score = max(0, 100 - normalized_impact)
        
        return ValidationResult(
            language=self.language,
            format_name="YAML",
            total_elements=total,
            valid_elements=valid,
            issues=self.issues,
            reproduction_score=reproduction_score,
            details={
                'modules_count': len(modules),
                'has_legend': 'meta' in self.data and 'legend' in self.data.get('meta', {}),
                'has_defaults': 'defaults' in self.data,
            }
        )


class HybridYAMLValidator(YAMLValidator):
    """Walidator dla formatu Hybrid YAML."""
    
    def validate(self) -> ValidationResult:
        """Waliduj format Hybrid YAML."""
        result = super().validate()
        result.format_name = "Hybrid YAML"
        
        # Dodatkowe sprawdzenia dla Hybrid
        if self.data:
            # Sprawdź czy ma sekcję M (module overview)
            if 'M' not in self.data:
                self.issues.append(ValidationIssue(
                    severity=Severity.LOW,
                    element_type=ElementType.MODULE,
                    element_name="root",
                    issue="Missing M section (compact module overview)",
                    impact_percent=1.0
                ))
            
            # Sprawdź czy ma header
            if 'header' not in self.data:
                self.issues.append(ValidationIssue(
                    severity=Severity.LOW,
                    element_type=ElementType.MODULE,
                    element_name="root",
                    issue="Missing header section",
                    impact_percent=1.0
                ))
            
            # Sprawdź conditional_imports
            for module in self.data.get('modules', []):
                if 'conditional_imports' in module:
                    result.details['has_conditional_imports'] = True
                if 'dataclasses' in module:
                    result.details['has_dataclasses_section'] = True
        
        return result


class TOONValidator(FormatValidator):
    """Walidator dla formatu TOON."""
    
    def parse(self) -> Dict[str, Any]:
        """Parsuj TOON (ultra-compact M/D or standard TOON)."""
        content = (self.content or '').lstrip()
        if content.startswith('project:'):
            return self._parse_standard()
        return self._parse_ultra_compact()

    def _detect_delimiter(self) -> str:
        # Standard TOON produced by this project uses either ',' or '\t' as row delimiter.
        # Note: '|' is used inside some cell values (e.g., decorators), so it must not be
        # treated as a delimiter.
        if '\t' in self.content:
            return '\t'
        return ','

    def _strip_quotes(self, s: str) -> str:
        s = (s or '').strip()
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            return s[1:-1]
        return s

    def _parse_ultra_compact(self) -> Dict[str, Any]:
        """Parsuj ultra-compact TOON (M/D)."""
        data = {
            'modules': [],
            'details': {}
        }
        
        lines = self.content.split('\n')
        current_section = None
        current_module = None
        
        for line in lines:
            line = line.rstrip()
            
            # Pomiń komentarze
            if line.startswith('#'):
                continue
            
            # Sekcja M[n]:
            if line.startswith('M['):
                current_section = 'modules'
                continue
            
            # Sekcja D:
            if line == 'D:':
                current_section = 'details'
                continue
            
            # Parsuj zawartość sekcji
            if current_section == 'modules' and line.strip():
                parts = line.strip().split(',')
                if len(parts) == 2:
                    data['modules'].append({
                        'path': parts[0],
                        'lines': int(parts[1])
                    })
            
            elif current_section == 'details':
                if line.endswith(':') and not line.startswith(' '):
                    current_module = line[:-1].strip()
                    data['details'][current_module] = {
                        'imports': [],
                        'exports': [],
                        'classes': [],
                        'functions': []
                    }
                elif current_module and line.strip():
                    self._parse_detail_line(line.strip(), data['details'][current_module])
        
        return data

    def _parse_standard(self) -> Dict[str, Any]:
        """Parsuj standard TOON w wersji generowanej przez code2logic.toon_format.TOONGenerator."""
        import csv

        delimiter = self._detect_delimiter()
        lines = self.content.split('\n')

        data: Dict[str, Any] = {'modules': [], 'details': {}}
        in_module_details = False
        current_module: Optional[str] = None
        
        i = 0
        while i < len(lines):
            raw = lines[i].rstrip('\n')
            i += 1

            if not raw.strip() or raw.lstrip().startswith('#'):
                continue

            if raw.strip() == 'module_details:':
                in_module_details = True
                current_module = None
                continue

            if not in_module_details:
                continue

            # Module headers are indented by exactly 2 spaces. Do not match deeper nested blocks.
            m_mod = re.match(r'^ {2}([^ ].*):\s*$', raw)
            if m_mod:
                current_module = self._strip_quotes(m_mod.group(1))
                data['details'][current_module] = {
                    'functions': [],
                    'const': [],
                    'types': [],
                    'has_dataclass': False,
                    'has_dataclass_fields': False,
                }
                continue

            if not current_module:
                continue

            # classes table (detect @dataclass via decorators column)
            m_classes = re.match(r'^\s{4}classes\[(\d+)\]\{([^}]+)\}:\s*$', raw)
            if m_classes:
                n = int(m_classes.group(1))
                fields = [f.strip() for f in m_classes.group(2).split(',')]
                for _ in range(n):
                    if i >= len(lines):
                        break
                    row = lines[i].strip()
                    i += 1
                    if not row:
                        continue
                    values = next(csv.reader([row], delimiter=delimiter, quotechar='"', escapechar='\\'))
                    row_map: Dict[str, str] = {}
                    for fi, fn in enumerate(fields):
                        if fi < len(values):
                            row_map[fn] = self._strip_quotes(values[fi])
                    dec = row_map.get('decorators', '')
                    if dec and dec != '-' and 'dataclass' in dec:
                        data['details'][current_module]['has_dataclass'] = True
                continue

            # functions table
            m_funcs = re.match(r'^\s{4}functions\[(\d+)\]\{([^}]+)\}:\s*$', raw)
            if m_funcs:
                n = int(m_funcs.group(1))
                fields = [f.strip() for f in m_funcs.group(2).split(',')]
                for _ in range(n):
                    if i >= len(lines):
                        break
                    row = lines[i].strip()
                    i += 1
                    if not row:
                        continue
                    values = next(csv.reader([row], delimiter=delimiter, quotechar='"', escapechar='\\'))
                    row_map: Dict[str, str] = {}
                    for fi, fn in enumerate(fields):
                        if fi < len(values):
                            row_map[fn] = self._strip_quotes(values[fi])
                    name = row_map.get('name') or ''
                    sig = row_map.get('sig') or ''
                    if name:
                        data['details'][current_module]['functions'].append({'name': name, 'sig': sig})
                continue

            # constants table
            m_const = re.match(r'^\s{4}const\[(\d+)\]\{([^}]+)\}:\s*$', raw)
            if m_const:
                n = int(m_const.group(1))
                fields = [f.strip() for f in m_const.group(2).split(',')]
                for _ in range(n):
                    if i >= len(lines):
                        break
                    row = lines[i].strip()
                    i += 1
                    if not row:
                        continue
                    values = next(csv.reader([row], delimiter=delimiter, quotechar='"', escapechar='\\'))
                    row_map: Dict[str, str] = {}
                    for fi, fn in enumerate(fields):
                        if fi < len(values):
                            row_map[fn] = self._strip_quotes(values[fi])
                    name = row_map.get('n') or row_map.get('name') or ''
                    if name:
                        data['details'][current_module]['const'].append(row_map)
                continue

            # types table (enums/interfaces/type aliases)
            m_types = re.match(r'^\s{4}types\[(\d+)\]\{([^}]+)\}:\s*$', raw)
            if m_types:
                n = int(m_types.group(1))
                fields = [f.strip() for f in m_types.group(2).split(',')]
                for _ in range(n):
                    if i >= len(lines):
                        break
                    row = lines[i].strip()
                    i += 1
                    if not row:
                        continue
                    values = next(csv.reader([row], delimiter=delimiter, quotechar='"', escapechar='\\'))
                    row_map: Dict[str, str] = {}
                    for fi, fn in enumerate(fields):
                        if fi < len(values):
                            row_map[fn] = self._strip_quotes(values[fi])
                    if row_map.get('name'):
                        data['details'][current_module]['types'].append(row_map)
                continue

            # dataclass fields table
            if re.match(r'^\s{8}fields\[\d+\]\{', raw):
                data['details'][current_module]['has_dataclass_fields'] = True

        return data
    
    def _parse_detail_line(self, line: str, module_data: Dict):
        """Parsuj linię szczegółów modułu."""
        if line.startswith('i:'):
            module_data['imports'] = line[2:].strip().split(',')
        elif line.startswith('e:'):
            module_data['exports'] = line[2:].strip().split(',')
        elif ':' in line and '(' in line:
            # Klasa z metodami
            match = re.match(r'(\w+):\s*(.+?)(?:\s*#\s*(.*))?$', line)
            if match:
                class_name = match.group(1)
                methods_str = match.group(2)
                docstring = match.group(3) if match.group(3) else ''
                
                module_data['classes'].append({
                    'name': class_name,
                    'methods': methods_str,
                    'docstring': docstring
                })
        elif '(' in line and ')' in line:
            # Funkcja
            match = re.match(r'(\w+)\(([^)]*)\)(?:->(.+))?', line)
            if match:
                module_data['functions'].append({
                    'name': match.group(1),
                    'params': match.group(2),
                    'return_type': match.group(3).strip() if match.group(3) else ''
                })
    
    def validate(self) -> ValidationResult:
        """Waliduj format TOON."""
        data = self.parse()
        
        total = 0
        valid = 0
        constants_total = 0
        constants_with_value = 0
        enums_total = 0
        enums_with_values = 0
        dataclass_seen = False
        dataclass_fields_seen = False
        
        for module_path, module_data in data.get('details', {}).items():
            total += 1

            # Sprawdź funkcje (standard: sig, ultra: params)
            for func in module_data.get('functions', []):
                total += 1
                func_name = func.get('name', '')
                sig = func.get('sig')
                if sig is None:
                    params = func.get('params', '')
                    sig = f"({params})"

                # Require the signature field to exist, but allow empty-arg functions.
                if not sig:
                    self.issues.append(ValidationIssue(
                        severity=Severity.CRITICAL,
                        element_type=ElementType.FUNCTION,
                        element_name=func_name,
                        issue="Missing signature",
                        expected="(param1:type, param2:type)",
                        actual=sig or "(missing)",
                        impact_percent=15.0
                    ))
                else:
                    valid += 1
            
            # Sprawdź klasy - TOON ma tylko method counts
            for cls in module_data.get('classes', []):
                total += 1
                cls_name = cls.get('name', '')
                
                # Metody są jako string, np. "__init__(2),run(1)"
                methods_str = cls.get('methods', '')
                if methods_str:
                    # Parsuj methods count
                    method_matches = re.findall(r'(\w+)\((\d+)\)', methods_str)
                    for method_name, param_count in method_matches:
                        self.issues.append(ValidationIssue(
                            severity=Severity.HIGH,
                            element_type=ElementType.METHOD,
                            element_name=f"{cls_name}.{method_name}",
                            issue=f"TOON shows only parameter count ({param_count}), not names/types",
                            expected="(param1:type, param2:type)",
                            actual=f"({param_count})",
                            impact_percent=10.0
                        ))
                
                valid += 1  # Klasa istnieje

            # Stałe (standard TOON)
            for const in module_data.get('const', []):
                constants_total += 1
                name = const.get('n') or const.get('name') or 'unknown'
                t = const.get('t') or const.get('type')
                v = const.get('v') or const.get('value')
                keys = const.get('keys')

                if not t or t == '-':
                    self.issues.append(ValidationIssue(
                        severity=Severity.HIGH,
                        element_type=ElementType.CONSTANT,
                        element_name=name,
                        issue="Constant missing type information",
                        expected="t: Dict[str, str]",
                        actual="t: (missing)",
                        impact_percent=5.0
                    ))

                has_value = bool(v) and v != '-'
                has_keys = bool(keys) and keys != '-'
                if has_value or has_keys:
                    constants_with_value += 1
                else:
                    self.issues.append(ValidationIssue(
                        severity=Severity.CRITICAL,
                        element_type=ElementType.CONSTANT,
                        element_name=name,
                        issue="Constant missing value - cannot reproduce",
                        expected="v: ... OR keys: [...]",
                        actual="(missing)",
                        impact_percent=15.0
                    ))

            # Types/enums (standard TOON)
            for t in module_data.get('types', []):
                kind = (t.get('kind') or '').strip().lower()
                if kind != 'enum':
                    continue
                enums_total += 1
                name = t.get('name') or 'unknown'
                values = t.get('values')
                if values and values != '-':
                    enums_with_values += 1
                else:
                    self.issues.append(ValidationIssue(
                        severity=Severity.CRITICAL,
                        element_type=ElementType.ENUM,
                        element_name=name,
                        issue="Enum missing values - cannot reproduce",
                        expected="values: A|B|C or A=1|B=2",
                        actual=values or "(missing)",
                        impact_percent=8.0
                    ))

            # Dataclasses
            if module_data.get('has_dataclass'):
                dataclass_seen = True
            if module_data.get('has_dataclass_fields'):
                dataclass_fields_seen = True
        
        # Only report missing sections if they are actually absent/needed
        if constants_total == 0:
            self.issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                element_type=ElementType.CONSTANT,
                element_name="*",
                issue="TOON format does not capture constant values",
                impact_percent=15.0
            ))

        if dataclass_seen and not dataclass_fields_seen:
            self.issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                element_type=ElementType.DATACLASS,
                element_name="*",
                issue="TOON format does not capture dataclass fields",
                impact_percent=10.0
            ))

        
        total_impact = sum(issue.impact_percent for issue in self.issues)
        normalized_impact = total_impact / max(total, 1)
        reproduction_score = max(0, 100 - normalized_impact)
        
        return ValidationResult(
            language=self.language,
            format_name="TOON",
            total_elements=total,
            valid_elements=valid,
            issues=self.issues,
            reproduction_score=reproduction_score,
            details={
                'modules_count': len(data.get('modules', [])),
            }
        )


# =============================================================================
# MULTI-LANGUAGE TESTER
# =============================================================================

class MultiLanguageReproductionTester:
    """Tester reprodukowalności dla wielu języków."""
    
    # Wzorce oczekiwanych elementów dla każdego języka
    LANGUAGE_PATTERNS = {
        Language.PYTHON: {
            'signature_pattern': r'\(([^)]*)\)',
            'type_annotation': r':(\w+)',
            'default_value': r'=([^,)]+)',
            'decorator': r'@(\w+)',
            'dataclass_marker': '@dataclass',
            'constant_pattern': r'^[A-Z][A-Z_0-9]+\s*[=:]',
        },
        Language.TYPESCRIPT: {
            'signature_pattern': r'\(([^)]*)\)',
            'type_annotation': r':\s*(\w+(?:<[^>]+>)?)',
            'interface_pattern': r'interface\s+(\w+)',
            'type_alias_pattern': r'type\s+(\w+)\s*=',
            'enum_pattern': r'enum\s+(\w+)',
        },
        Language.JAVA: {
            'signature_pattern': r'\(([^)]*)\)',
            'type_pattern': r'(\w+(?:<[^>]+>)?)\s+\w+',
            'annotation_pattern': r'@(\w+)',
            'interface_pattern': r'interface\s+(\w+)',
            'enum_pattern': r'enum\s+(\w+)',
        },
        Language.GO: {
            'signature_pattern': r'\(([^)]*)\)\s*(?:\(([^)]*)\)|(\w+))?',
            'struct_pattern': r'type\s+(\w+)\s+struct',
            'interface_pattern': r'type\s+(\w+)\s+interface',
            'const_pattern': r'const\s+(\w+)',
        },
        Language.RUST: {
            'signature_pattern': r'\(([^)]*)\)\s*(?:->\s*(.+))?',
            'struct_pattern': r'struct\s+(\w+)',
            'trait_pattern': r'trait\s+(\w+)',
            'enum_pattern': r'enum\s+(\w+)',
            'impl_pattern': r'impl(?:<[^>]+>)?\s+(?:(\w+)\s+for\s+)?(\w+)',
        },
    }
    
    def __init__(self):
        self.results: Dict[Language, Dict[str, ValidationResult]] = {}
    
    def test_format(
        self, 
        content: str, 
        format_type: str, 
        language: Language
    ) -> ValidationResult:
        """Testuj pojedynczy format dla danego języka."""
        
        if format_type == 'yaml':
            validator = YAMLValidator(content, language)
        elif format_type == 'hybrid':
            validator = HybridYAMLValidator(content, language)
        elif format_type == 'toon':
            validator = TOONValidator(content, language)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        return validator.validate()
    
    def test_all_formats(
        self, 
        yaml_content: str, 
        hybrid_content: str, 
        toon_content: str,
        language: Language
    ) -> Dict[str, ValidationResult]:
        """Testuj wszystkie formaty dla danego języka."""
        
        results = {}
        
        if yaml_content:
            results['yaml'] = self.test_format(yaml_content, 'yaml', language)
        
        if hybrid_content:
            results['hybrid'] = self.test_format(hybrid_content, 'hybrid', language)
        
        if toon_content:
            results['toon'] = self.test_format(toon_content, 'toon', language)
        
        self.results[language] = results
        return results
    
    def generate_report(self) -> str:
        """Generuj raport porównawczy."""
        lines = []
        lines.append("=" * 80)
        lines.append("MULTI-LANGUAGE REPRODUCTION TEST REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Tabela podsumowująca
        lines.append("SUMMARY BY LANGUAGE AND FORMAT:")
        lines.append("-" * 80)
        lines.append(f"{'Language':<15} {'TOON':>10} {'YAML':>10} {'Hybrid':>10} {'Best Format':>15}")
        lines.append("-" * 80)
        
        for lang, formats in self.results.items():
            scores = {}
            for fmt_name, result in formats.items():
                scores[fmt_name] = result.reproduction_score
            
            toon_score = scores.get('toon', 0)
            yaml_score = scores.get('yaml', 0)
            hybrid_score = scores.get('hybrid', 0)
            
            best = max(scores.items(), key=lambda x: x[1]) if scores else ('N/A', 0)
            
            lines.append(
                f"{lang.value:<15} {toon_score:>9.1f}% {yaml_score:>9.1f}% "
                f"{hybrid_score:>9.1f}% {best[0]:>15}"
            )
        
        lines.append("-" * 80)
        lines.append("")
        
        # Szczegółowe problemy per język
        lines.append("CRITICAL ISSUES BY LANGUAGE:")
        lines.append("-" * 80)
        
        for lang, formats in self.results.items():
            lines.append(f"\n{lang.value.upper()}:")
            
            for fmt_name, result in formats.items():
                critical = [i for i in result.issues if i.severity == Severity.CRITICAL]
                if critical:
                    lines.append(f"  {fmt_name}: {len(critical)} critical issues")
                    for issue in critical[:3]:  # Top 3
                        lines.append(f"    - {issue.element_name}: {issue.issue}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 80)
        
        # Zbierz rekomendacje
        all_issues = []
        for formats in self.results.values():
            for result in formats.values():
                all_issues.extend(result.issues)
        
        # Grupuj po typie problemu
        issue_counts = defaultdict(int)
        for issue in all_issues:
            issue_counts[issue.issue] += 1
        
        # Top 5 problemów
        top_issues = sorted(issue_counts.items(), key=lambda x: -x[1])[:5]
        for i, (issue, count) in enumerate(top_issues, 1):
            lines.append(f"{i}. [{count}x] {issue}")
        
        return "\n".join(lines)
    
    def generate_language_specific_recommendations(self, language: Language) -> List[str]:
        """Generuj rekomendacje specyficzne dla języka."""
        recommendations = []
        reqs = LANGUAGE_REQUIREMENTS.get(language)
        
        if not reqs:
            return ["No requirements defined for this language"]
        
        # Sprawdź czy format obsługuje wymagane elementy
        if reqs.supports_interfaces:
            recommendations.append(
                f"Add 'interfaces' section for {language.value} - "
                "interfaces are critical for this language"
            )
        
        if reqs.supports_traits:
            recommendations.append(
                f"Add 'traits' section for {language.value} - "
                "traits are critical for this language"
            )
        
        if reqs.supports_dataclasses:
            recommendations.append(
                f"Add 'dataclasses' section with fields for {language.value}"
            )
        
        if reqs.type_annotations == "required":
            recommendations.append(
                f"Ensure all signatures have type annotations for {language.value} - "
                "types are required"
            )
        
        if reqs.supports_decorators:
            recommendations.append(
                f"Add 'decorators' field to methods for {language.value}"
            )
        
        if reqs.supports_annotations:
            recommendations.append(
                f"Add 'annotations' field to classes/methods for {language.value}"
            )
        
        return recommendations


# =============================================================================
# SCORING SYSTEM
# =============================================================================

class ReproductionScorer:
    """System oceny reprodukowalności."""
    
    # Wagi dla różnych elementów (suma = 100)
    WEIGHTS = {
        'signature_completeness': 25,    # Pełne sygnatury z params i defaults
        'constant_values': 15,           # Wartości stałych
        'enum_values': 8,                # Wartości enumów
        'dataclass_fields': 10,          # Pola dataclassów
        'class_attributes': 8,           # Atrybuty klas
        'type_annotations': 10,          # Adnotacje typów
        'docstrings': 5,                 # Docstringi
        'decorators': 5,                 # Dekoratory
        'imports': 5,                    # Poprawne importy
        'exports': 4,                    # Eksporty
        'interfaces': 3,                 # Interfejsy (dla języków je wspierających)
        'conditional_imports': 2,        # Warunkowe importy
    }
    
    @classmethod
    def calculate_score(cls, result: ValidationResult, language: Language) -> float:
        """Oblicz szczegółowy score reprodukcji."""
        score = 100.0
        reqs = LANGUAGE_REQUIREMENTS.get(language)
        
        for issue in result.issues:
            # Podstawowa kara
            penalty = issue.impact_percent
            
            # Modyfikuj w zależności od ważności dla języka
            if reqs:
                if issue.element_type in reqs.critical_elements:
                    penalty *= 1.5  # Większa kara dla krytycznych elementów
            
            score -= penalty
        
        return max(0, min(100, score))
    
    @classmethod
    def get_improvement_priorities(
        cls, 
        result: ValidationResult
    ) -> List[Tuple[str, float, str]]:
        """Zwróć priorytety ulepszeń (nazwa, potencjalny zysk, opis)."""
        priorities = []
        
        # Grupuj problemy po typie
        issue_groups = defaultdict(list)
        for issue in result.issues:
            key = issue.issue.split(' - ')[0] if ' - ' in issue.issue else issue.issue
            issue_groups[key].append(issue)
        
        for issue_type, issues in issue_groups.items():
            total_impact = sum(i.impact_percent for i in issues)
            priorities.append((
                issue_type,
                total_impact,
                f"Fix {len(issues)} occurrences for +{total_impact:.1f}% reproduction"
            ))
        
        return sorted(priorities, key=lambda x: -x[1])


# =============================================================================
# PRZYKŁADOWE DANE TESTOWE
# =============================================================================

SAMPLE_YAML = """
meta:
  legend:
    p: path
    l: lines
    n: name
    sig: signature
    d: docstring
defaults:
  lang: python
modules:
- p: shared_utils.py
  l: 279
  i:
  - hashlib
  - re
  - typing.{Dict,List,Optional,Set}
  e:
  - compact_imports
  - TYPE_ABBREVIATIONS
  - CATEGORY_PATTERNS
  f:
  - n: compact_imports
    sig: ''
    ret: List[str]
    d: compact imports
  - n: build_signature
    sig: ''
    ret: str
    d: creates signature
"""

SAMPLE_HYBRID = """
header:
  project: test
  files: 1
  lines: 279
M:
- shared_utils.py:279
defaults:
  lang: python
modules:
- p: shared_utils.py
  l: 279
  i:
  - hashlib
  - re
  - typing.{Dict,List,Optional,Set}
  const:
  - n: TYPE_ABBREVIATIONS
    t: constant
  - n: CATEGORY_PATTERNS
    t: constant
  f:
  - n: compact_imports
    sig: ()
    ret: List[str]
    d: compact imports
  - n: build_signature
    sig: ()
    ret: str
    d: creates signature
"""

SAMPLE_TOON = """
# code2logic | 1f 279L | python:1
M[1]:
  shared_utils.py,279
D:
  shared_utils.py:
    i: hashlib,re,typing.{Dict,List,Optional,Set}
    e: compact_imports,TYPE_ABBREVIATIONS,CATEGORY_PATTERNS
    compact_imports()->List[str]
    build_signature()->str
"""


# =============================================================================
# MAIN - CLI
# =============================================================================

def run_tests(yaml_path: str = None, hybrid_path: str = None, toon_path: str = None):
    """Uruchom testy reprodukowalności."""
    
    tester = MultiLanguageReproductionTester()
    
    # Wczytaj pliki lub użyj przykładów
    yaml_content = Path(yaml_path).read_text() if yaml_path else SAMPLE_YAML
    hybrid_content = Path(hybrid_path).read_text() if hybrid_path else SAMPLE_HYBRID
    toon_content = Path(toon_path).read_text() if toon_path else SAMPLE_TOON
    
    # Testuj dla Pythona (główny test)
    print("Testing Python formats...")
    results = tester.test_all_formats(
        yaml_content, hybrid_content, toon_content, Language.PYTHON
    )
    
    # Wypisz wyniki
    print("\n" + "=" * 60)
    print("REPRODUCTION TEST RESULTS")
    print("=" * 60)
    
    for fmt_name, result in results.items():
        print(f"\n{fmt_name.upper()}:")
        print(f"  Reproduction Score: {result.reproduction_score:.1f}%")
        print(f"  Total Elements: {result.total_elements}")
        print(f"  Valid Elements: {result.valid_elements}")
        print(f"  Issues Found: {len(result.issues)}")
        
        # Top 3 critical issues
        critical = [i for i in result.issues if i.severity == Severity.CRITICAL]
        if critical:
            print(f"  Critical Issues ({len(critical)}):")
            for issue in critical[:3]:
                print(f"    - {issue.element_name}: {issue.issue}")
    
    # Rekomendacje
    print("\n" + "=" * 60)
    print("IMPROVEMENT PRIORITIES")
    print("=" * 60)
    
    for fmt_name, result in results.items():
        priorities = ReproductionScorer.get_improvement_priorities(result)
        if priorities:
            print(f"\n{fmt_name.upper()} - Top improvements:")
            for name, impact, desc in priorities[:3]:
                print(f"  [{impact:>5.1f}%] {desc}")
    
    # Pełny raport
    print("\n")
    print(tester.generate_report())
    
    # Rekomendacje per język
    print("\n" + "=" * 60)
    print("LANGUAGE-SPECIFIC RECOMMENDATIONS")
    print("=" * 60)
    
    for lang in [Language.PYTHON, Language.TYPESCRIPT, Language.JAVA, Language.GO, Language.RUST]:
        recs = tester.generate_language_specific_recommendations(lang)
        print(f"\n{lang.value}:")
        for rec in recs[:3]:
            print(f"  - {rec}")


def main():
    """Główna funkcja CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Code2Logic Multi-Language Reproduction Tester"
    )
    parser.add_argument('--yaml', help='Path to YAML format file')
    parser.add_argument('--hybrid', help='Path to Hybrid YAML file')
    parser.add_argument('--toon', help='Path to TOON format file')
    parser.add_argument('--language', default='python', 
                       choices=[l.value for l in Language],
                       help='Target language')
    parser.add_argument('--all-languages', action='store_true',
                       help='Test for all supported languages')
    
    args = parser.parse_args()
    
    run_tests(args.yaml, args.hybrid, args.toon)


if __name__ == "__main__":
    main()
