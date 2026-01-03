"""
Data models for code2logic.

This module contains the core dataclasses that represent
the structure of analyzed code projects.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class DependencyType(Enum):
    """Types of dependencies."""
    IMPORT = "import"
    INHERITANCE = "inheritance"
    FUNCTION_CALL = "function_call"
    ATTRIBUTE_ACCESS = "attribute_access"
    COMPOSITION = "composition"


@dataclass
class Function:
    """Represents a function in the codebase."""
    name: str
    parameters: List[str] = field(default_factory=list)
    lines_of_code: int = 0
    complexity: int = 1
    docstring: Optional[str] = None
    code: str = ""
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_generator: bool = False
    is_private: bool = False
    is_static: bool = False
    is_class_method: bool = False
    is_property: bool = False


@dataclass
class Class:
    """Represents a class in the codebase."""
    name: str
    methods: List[Function] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)
    lines_of_code: int = 0
    docstring: Optional[str] = None
    attributes: List[str] = field(default_factory=list)
    is_abstract: bool = False
    is_interface: bool = False
    decorators: List[str] = field(default_factory=list)
    inner_classes: List['Class'] = field(default_factory=list)


@dataclass
class Module:
    """Represents a module in the codebase."""
    name: str
    path: str
    functions: List[Function] = field(default_factory=list)
    classes: List[Class] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    lines_of_code: int = 0
    docstring: Optional[str] = None
    constants: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    is_package: bool = False
    is_test_module: bool = False


@dataclass
class Dependency:
    """Represents a dependency between components."""
    source: str
    target: str
    type: DependencyType
    strength: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Similarity:
    """Represents similarity between code components."""
    item1: str
    item2: str
    score: float
    similarity_type: str  # "structural", "semantic", "syntactic"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Project:
    """Represents an entire project."""
    name: str
    path: str
    modules: List[Module] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    similarities: List[Similarity] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_module_by_name(self, name: str) -> Optional[Module]:
        """Get a module by name."""
        for module in self.modules:
            if module.name == name:
                return module
        return None
    
    def get_function_by_name(self, name: str) -> Optional[Function]:
        """Get a function by fully qualified name."""
        parts = name.split('.')
        if len(parts) < 2:
            return None
        
        module_name = '.'.join(parts[:-1])
        function_name = parts[-1]
        
        module = self.get_module_by_name(module_name)
        if module:
            for func in module.functions:
                if func.name == function_name:
                    return func
        return None
    
    def get_class_by_name(self, name: str) -> Optional[Class]:
        """Get a class by fully qualified name."""
        parts = name.split('.')
        if len(parts) < 2:
            return None
        
        module_name = '.'.join(parts[:-1])
        class_name = parts[-1]
        
        module = self.get_module_by_name(module_name)
        if module:
            for cls in module.classes:
                if cls.name == class_name:
                    return cls
        return None
    
    def get_statistics(self) -> Dict[str, int]:
        """Get project statistics."""
        return {
            'modules': len(self.modules),
            'functions': sum(len(m.functions) for m in self.modules),
            'classes': sum(len(m.classes) for m in self.modules),
            'dependencies': len(self.dependencies),
            'similarities': len(self.similarities),
            'lines_of_code': sum(m.lines_of_code for m in self.modules),
            'functions_with_docs': sum(
                1 for m in self.modules 
                for f in m.functions 
                if f.docstring
            ),
            'classes_with_docs': sum(
                1 for m in self.modules 
                for c in m.classes 
                if c.docstring
            ),
        }
    
    def get_complex_modules(self, threshold: int = 500) -> List[Module]:
        """Get modules with high complexity."""
        return [m for m in self.modules if m.lines_of_code > threshold]
    
    def get_complex_functions(self, threshold: int = 10) -> List[Function]:
        """Get functions with high complexity."""
        return [
            f for m in self.modules 
            for f in m.functions 
            if f.complexity > threshold
        ]
    
    def get_large_classes(self, threshold: int = 15) -> List[Class]:
        """Get classes with many methods."""
        return [
            c for m in self.modules 
            for c in m.classes 
            if len(c.methods) > threshold
        ]


@dataclass
class AnalysisResult:
    """Result of project analysis."""
    project: Project
    analysis_time: float
    parser_used: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary."""
        stats = self.project.get_statistics()
        
        return {
            'project_name': self.project.name,
            'analysis_time': self.analysis_time,
            'parser_used': self.parser_used,
            'statistics': stats,
            'errors_count': len(self.errors),
            'warnings_count': len(self.warnings),
            'has_errors': len(self.errors) > 0,
            'has_warnings': len(self.warnings) > 0,
        }


@dataclass
class CodeSmell:
    """Represents a code smell detected in the code."""
    type: str
    severity: str  # "low", "medium", "high", "critical"
    target: str
    description: str
    suggestion: str
    line_number: Optional[int] = None
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefactoringSuggestion:
    """Represents a refactoring suggestion."""
    type: str
    target: str
    description: str
    benefits: List[str]
    effort: str  # "low", "medium", "high"
    risk: str  # "low", "medium", "high"
    automated: bool = False
    code_changes: Optional[str] = None


@dataclass
class TestCoverage:
    """Represents test coverage information."""
    module: str
    functions_covered: int
    functions_total: int
    classes_covered: int
    classes_total: int
    lines_covered: int
    lines_total: int
    coverage_percentage: float
    
    @property
    def is_well_covered(self) -> bool:
        """Check if the module is well covered."""
        return self.coverage_percentage >= 80.0


@dataclass
class PerformanceMetric:
    """Represents a performance metric."""
    name: str
    value: float
    unit: str
    threshold: Optional[float] = None
    is_acceptable: bool = True
    
    def __post_init__(self):
        """Check if metric is acceptable."""
        if self.threshold is not None:
            self.is_acceptable = self.value <= self.threshold


@dataclass
class SecurityIssue:
    """Represents a security issue."""
    type: str
    severity: str  # "low", "medium", "high", "critical"
    target: str
    description: str
    recommendation: str
    cwe_id: Optional[str] = None
    line_number: Optional[int] = None
    confidence: float = 0.8


# Utility functions for working with models
def create_function(
    name: str,
    code: str = "",
    **kwargs
) -> Function:
    """Create a Function instance."""
    return Function(name=name, code=code, **kwargs)


def create_class(
    name: str,
    **kwargs
) -> Class:
    """Create a Class instance."""
    return Class(name=name, **kwargs)


def create_module(
    name: str,
    path: str,
    **kwargs
) -> Module:
    """Create a Module instance."""
    return Module(name=name, path=path, **kwargs)


def create_dependency(
    source: str,
    target: str,
    dep_type: str,
    **kwargs
) -> Dependency:
    """Create a Dependency instance."""
    return Dependency(
        source=source,
        target=target,
        type=DependencyType(dep_type),
        **kwargs
    )


def create_project(
    name: str,
    path: str,
    **kwargs
) -> Project:
    """Create a Project instance."""
    return Project(name=name, path=path, **kwargs)
