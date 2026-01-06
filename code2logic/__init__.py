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

__version__ = "1.0.11"
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
    reproduce_project as _reproduce_project_from_source,
)
from .refactor import (
    find_duplicates,
    analyze_quality as _analyze_quality_from_path,
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
from .logicml import (
    LogicMLGenerator,
    LogicMLSpec,
    generate_logicml,
)
from .function_logic import (
    FunctionLogicGenerator,
)
from .prompts import (
    FORMAT_HINTS,
    get_reproduction_prompt,
    get_review_prompt,
    get_fix_prompt,
)
from .schemas import (
    validate_yaml,
    validate_logicml,
    validate_markdown,
    validate_json,
    YAMLSchema,
    LogicMLSchema,
    MarkdownSchema,
    JSONSchema,
)
from .quality import (
    QualityAnalyzer,
    QualityReport,
    QualityIssue,
    analyze_quality as _analyze_quality_from_project,
    get_quality_summary,
)
from .similarity import get_refactoring_suggestions
from .errors import (
    ErrorHandler,
    ErrorType,
    ErrorSeverity,
    AnalysisError,
    AnalysisResult,
    create_error_handler,
)
from .reproducer import (
    SpecReproducer,
    SpecValidator,
    ReproductionResult as SpecReproductionResult,
    FileValidation,
    reproduce_project as _reproduce_project_from_spec,
    validate_files,
)


def analyze_quality(target, *args, **kwargs):
    if isinstance(target, ProjectInfo):
        return _analyze_quality_from_project(target, *args, **kwargs)
    return _analyze_quality_from_path(str(target), *args, **kwargs)


def reproduce_project(source: str, *args, **kwargs):
    src = str(source)
    if src.endswith(('.yaml', '.yml', '.json')):
        return _reproduce_project_from_spec(src, *args, **kwargs)
    return _reproduce_project_from_source(src, *args, **kwargs)
from .toon_format import (
    TOONGenerator,
    TOONParser,
    generate_toon,
    parse_toon,
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
from .llm_profiler import (
    LLMProfiler,
    LLMProfile,
    AdaptiveChunker,
    profile_llm,
    get_profile,
    get_or_create_profile,
    get_adaptive_chunker,
    load_profiles,
    save_profile,
)
from .terminal import (
    ShellRenderer,
    render,
    get_renderer,
    set_renderer,
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
    "SpecReproductionResult",
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
    # LLM Profiler
    "LLMProfiler",
    "LLMProfile",
    "AdaptiveChunker",
    "profile_llm",
    "get_profile",
    "get_or_create_profile",
    "get_adaptive_chunker",
    "load_profiles",
    "save_profile",
    # Terminal Rendering
    "ShellRenderer",
    "render",
    "get_renderer",
    "set_renderer",
]