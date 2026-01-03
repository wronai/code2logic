"""
Code2Logic - Convert source code to logical representation for LLM analysis.

A Python library that analyzes codebases and generates compact, LLM-friendly
representations with semantic understanding using NLP and AST parsing.

Features:
- Multi-language support (Python, JavaScript, TypeScript, Java, Go, Rust, etc.)
- Tree-sitter AST parsing for 99% accuracy
- NetworkX dependency graph analysis with PageRank
- Rapidfuzz similarity detection for duplicate functions
- NLP-powered intent extraction from function names and docstrings

Example:
    >>> from code2logic import analyze_project, MarkdownGenerator
    >>> project = analyze_project("/path/to/project")
    >>> output = MarkdownGenerator().generate(project)
    >>> print(output)
"""

__version__ = "1.0.0"
__author__ = "Softreck"
__email__ = "info@softreck.dev"
__license__ = "MIT"

from .analyzer import (
    ProjectAnalyzer,
    analyze_project,
)
from .models import (
    FunctionInfo,
    ClassInfo,
    TypeInfo,
    ModuleInfo,
    DependencyNode,
    ProjectInfo,
)
from .generators import (
    MarkdownGenerator,
    CompactGenerator,
    JSONGenerator,
    YAMLGenerator,
    CSVGenerator,
)
from .gherkin import (
    GherkinGenerator,
    StepDefinitionGenerator,
    CucumberYAMLGenerator,
    csv_to_gherkin,
    gherkin_to_test_data,
)
from .intent import EnhancedIntentGenerator
from .parsers import TreeSitterParser, UniversalParser
from .dependency import DependencyAnalyzer
from .similarity import SimilarityDetector

__all__ = [
    # Version
    "__version__",
    # Main API
    "analyze_project",
    "ProjectAnalyzer",
    # Models
    "FunctionInfo",
    "ClassInfo", 
    "TypeInfo",
    "ModuleInfo",
    "DependencyNode",
    "ProjectInfo",
    # Generators
    "MarkdownGenerator",
    "CompactGenerator",
    "JSONGenerator",
    "YAMLGenerator",
    "CSVGenerator",
    # Gherkin/BDD
    "GherkinGenerator",
    "StepDefinitionGenerator",
    "CucumberYAMLGenerator",
    "csv_to_gherkin",
    "gherkin_to_test_data",
    # Components
    "EnhancedIntentGenerator",
    "TreeSitterParser",
    "UniversalParser",
    "DependencyAnalyzer",
    "SimilarityDetector",
]
