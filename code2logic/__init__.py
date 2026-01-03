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
from .config import Config, load_env, get_api_key, get_model
from .llm_clients import (
    BaseLLMClient,
    OpenRouterClient,
    OllamaLocalClient,
    LiteLLMClient,
    get_client,
)
from .reproduction import (
    generate_file_gherkin,
    compare_code,
    extract_code_block,
    CodeReproducer,
)
from .code_review import (
    analyze_code_quality,
    check_security_issues,
    check_performance_issues,
    CodeReviewer,
)
from .benchmark import (
    ReproductionBenchmark,
    run_benchmark,
    FormatResult,
    BenchmarkResult,
)

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
    # Configuration
    "Config",
    "load_env",
    "get_api_key",
    "get_model",
    # LLM Clients
    "BaseLLMClient",
    "OpenRouterClient",
    "OllamaLocalClient",
    "LiteLLMClient",
    "get_client",
    # Reproduction
    "generate_file_gherkin",
    "compare_code",
    "extract_code_block",
    "CodeReproducer",
    # Code Review
    "analyze_code_quality",
    "check_security_issues",
    "check_performance_issues",
    "CodeReviewer",
    # Benchmark
    "ReproductionBenchmark",
    "run_benchmark",
    "FormatResult",
    "BenchmarkResult",
]