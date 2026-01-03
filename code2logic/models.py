"""
Data models for Code2Logic.

Contains dataclasses representing the analyzed code structure:
- FunctionInfo: Function/method details
- ClassInfo: Class/interface details
- TypeInfo: Type alias/interface/enum details
- ModuleInfo: File/module details
- DependencyNode: Dependency graph node with metrics
- ProjectInfo: Complete project analysis results
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class FunctionInfo:
    """Information about a function or method.
    
    Attributes:
        name: Function name
        params: List of parameter strings (e.g., ["x:int", "y:str"])
        return_type: Return type annotation if available
        docstring: First line of docstring
        calls: List of function calls made within this function
        raises: List of exceptions raised
        complexity: Cyclomatic complexity estimate
        lines: Number of lines in the function
        decorators: List of decorator names
        is_async: Whether function is async
        is_static: Whether function is static method
        is_private: Whether function is private (starts with _)
        intent: Generated intent/purpose description
        start_line: Starting line number in source
        end_line: Ending line number in source
    """
    name: str
    params: List[str]
    return_type: Optional[str]
    docstring: Optional[str]
    calls: List[str]
    raises: List[str]
    complexity: int
    lines: int
    decorators: List[str]
    is_async: bool
    is_static: bool
    is_private: bool
    intent: str
    start_line: int = 0
    end_line: int = 0


@dataclass
class ClassInfo:
    """Information about a class or interface.
    
    Attributes:
        name: Class name
        bases: List of base class names
        docstring: First line of docstring
        methods: List of methods in the class
        properties: List of property/attribute names
        is_interface: Whether this is an interface (TypeScript)
        is_abstract: Whether this is an abstract class
        generic_params: List of generic type parameters
    """
    name: str
    bases: List[str]
    docstring: Optional[str]
    methods: List[FunctionInfo]
    properties: List[str]
    is_interface: bool
    is_abstract: bool
    generic_params: List[str]


@dataclass
class TypeInfo:
    """Information about a type alias, interface, or enum.
    
    Attributes:
        name: Type name
        kind: Type kind ('type', 'interface', 'enum', 'struct', 'trait')
        definition: Short definition string
    """
    name: str
    kind: str  # 'type', 'interface', 'enum', 'struct', 'trait'
    definition: str


@dataclass
class ModuleInfo:
    """Information about a source file/module.
    
    Attributes:
        path: Relative path to the file
        language: Programming language
        imports: List of import statements
        exports: List of exported symbols
        classes: List of classes in the module
        functions: List of top-level functions
        types: List of type definitions
        constants: List of constant names (UPPERCASE)
        docstring: Module docstring
        lines_total: Total line count
        lines_code: Lines of actual code (excluding comments/blanks)
    """
    path: str
    language: str
    imports: List[str]
    exports: List[str]
    classes: List[ClassInfo]
    functions: List[FunctionInfo]
    types: List[TypeInfo]
    constants: List[str]
    docstring: Optional[str]
    lines_total: int
    lines_code: int


@dataclass
class DependencyNode:
    """Node in the dependency graph with metrics.
    
    Attributes:
        path: Module path
        in_degree: Number of incoming dependencies
        out_degree: Number of outgoing dependencies  
        pagerank: PageRank score (importance metric)
        is_hub: Whether this is a hub module (high centrality)
        cluster: Cluster ID for grouping related modules
    """
    path: str
    in_degree: int = 0
    out_degree: int = 0
    pagerank: float = 0.0
    is_hub: bool = False
    cluster: int = 0


@dataclass
class ProjectInfo:
    """Complete project analysis results.
    
    Attributes:
        name: Project name (directory name)
        root_path: Absolute path to project root
        languages: Dict mapping language to file count
        modules: List of all analyzed modules
        dependency_graph: Dict mapping module path to list of dependencies
        dependency_metrics: Dict mapping module path to DependencyNode
        entrypoints: List of detected entry point files
        similar_functions: Dict mapping function to list of similar functions
        total_files: Total number of analyzed files
        total_lines: Total line count across all files
        generated_at: ISO timestamp of analysis
    """
    name: str
    root_path: str
    languages: Dict[str, int]
    modules: List[ModuleInfo]
    dependency_graph: Dict[str, List[str]]
    dependency_metrics: Dict[str, DependencyNode]
    entrypoints: List[str]
    similar_functions: Dict[str, List[str]]
    total_files: int
    total_lines: int
    generated_at: str


# Backwards compatibility aliases for tests
Project = ProjectInfo
Module = ModuleInfo
Function = FunctionInfo
Class = ClassInfo
