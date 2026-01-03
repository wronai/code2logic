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
from .file_formats import (
    generate_file_csv,
    generate_file_json,
    generate_file_yaml,
)
from .adaptive import (
    AdaptiveReproducer,
    AdaptiveResult,
    get_llm_capabilities,
    LLM_CAPABILITIES,
)
from .universal import (
    UniversalReproducer,
    UniversalParser,
    CodeGenerator,
    CodeLogic,
    CodeElement,
    Language,
    ElementType,
    reproduce_file,
)
from .project_reproducer import (
    ProjectReproducer,
    ProjectResult,
    FileResult,
    reproduce_project,
)
from .refactor import (
    find_duplicates,
    analyze_quality,
    suggest_refactoring,
    compare_codebases,
    quick_analyze,
    RefactoringReport,
    DuplicateGroup,
)
from .metrics import (
    ReproductionMetrics,
    ReproductionResult,
    TextMetrics,
    StructuralMetrics,
    SemanticMetrics,
    FormatMetrics,
    analyze_reproduction,
    compare_formats,
)
from .base import (
    VerboseMixin,
    BaseParser,
    BaseGenerator,
)
from .markdown_format import (
    MarkdownHybridGenerator,
    MarkdownSpec,
    generate_markdown_hybrid,
)
from .chunked_reproduction import (
    ChunkedReproducer,
    ChunkedResult,
    ChunkedSpec,
    Chunk,
    chunk_spec,
    auto_chunk_reproduce,
    get_llm_limit,
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
    # File Formats
    "generate_file_csv",
    "generate_file_json",
    "generate_file_yaml",
    # Adaptive
    "AdaptiveReproducer",
    "AdaptiveResult",
    "get_llm_capabilities",
    "LLM_CAPABILITIES",
    # Universal
    "UniversalReproducer",
    "UniversalParser",
    "CodeGenerator",
    "CodeLogic",
    "CodeElement",
    "Language",
    "ElementType",
    "reproduce_file",
    # Project
    "ProjectReproducer",
    "ProjectResult",
    "FileResult",
    "reproduce_project",
    # Refactoring
    "find_duplicates",
    "analyze_quality",
    "suggest_refactoring",
    "compare_codebases",
    "quick_analyze",
    "RefactoringReport",
    "DuplicateGroup",
    # Metrics
    "ReproductionMetrics",
    "ReproductionResult",
    "TextMetrics",
    "StructuralMetrics",
    "SemanticMetrics",
    "FormatMetrics",
    "analyze_reproduction",
    "compare_formats",
    # Base
    "VerboseMixin",
    "BaseParser",
    "BaseGenerator",
    # Markdown Format
    "MarkdownHybridGenerator",
    "MarkdownSpec",
    "generate_markdown_hybrid",
    "generate_file_markdown",
]