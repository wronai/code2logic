# 📦 code2logic

```yaml
generated: 2026-02-26T20:35:06.647138
files: 152
lines: 39758
languages: {"python": 140, "javascript": 3, "csharp": 1, "go": 1, "java": 1, "rust": 1, "sql": 1, "typescript": 4}
entrypoints: ["code2logic/__init__.py", "code2logic/__main__.py", "code2logic/base_generator.py", "code2logic/benchmarks/__init__.py", "code2logic/core/__init__.py"]
```

## 📁 Structure

```
├── code2logic/
│   ├── __init__.py: [python] analyze_quality, reproduce_project
│   ├── __main__.py: [python]
│   ├── adaptive.py: [python] LLM_CAPABILITIES, ChunkInfo, AdaptiveResult +2
│   ├── analyzer.py: [python] ProjectAnalyzer, analyze_project, get_library_status
│   ├── base.py: [python] VerboseMixin, BaseParser, BaseGenerator
│   ├── base_generator.py: [python] ProjectGenerator
│   ├── benchmark.py: [python] FormatResult, BenchmarkResult, FORMAT_PROMPTS +2
│   ├── benchmarks/
│   │   ├── __init__.py: [python]
│   │   ├── common.py: [python] create_single_project, generate_spec, generate_spec_token +3
│   │   ├── results.py: [python] FormatResult, FileResult, FunctionResult +2
│   │   └── runner.py: [python] BenchmarkRunner, run_benchmark
│   ├── chunked_reproduction.py: [python] LLM_CONTEXT_LIMITS, Chunk, ChunkedSpec +11
│   ├── cli.py: [python] Colors, Logger, ensure_dependencies +1
│   ├── code_review.py: [python] SECURITY_PATTERNS, PERFORMANCE_PATTERNS, COMPLEXITY_HIGH +7
│   ├── config.py: [python] Config, load_env, get_api_key +2
│   ├── core/
│   │   └── __init__.py: [python]
│   ├── dependency.py: [python] NETWORKX_AVAILABLE, DependencyAnalyzer, is_networkx_available
│   ├── errors.py: [python] ErrorSeverity, ErrorType, AnalysisError +3
│   ├── file_formats.py: [python] generate_file_csv, generate_file_json, generate_file_yaml
│   ├── formats/
│   │   └── __init__.py: [python]
│   ├── function_logic.py: [python] FunctionLogicGenerator
│   ├── generators.py: [python] bytes_to_kb, MarkdownGenerator, CompactGenerator +3
│   ├── gherkin.py: [python] GherkinScenario, GherkinFeature, StepDefinition +5
│   ├── integrations/
│   │   └── __init__.py: [python]
│   ├── intent.py: [python] IntentType, Intent, EnhancedIntentGenerator +1
│   ├── llm/
│   │   └── __init__.py: [python]
│   ├── llm.py: [python] LLMConfig, CodeAnalyzer, get_available_backends
│   ├── llm_clients.py: [python] get_priority_mode, get_effective_provider_priorities
│   ├── llm_profiler.py: [python] PROFILE_TEST_CASES, LLMProfile, ProfileTestResult +8
│   ├── logicml.py: [python] LogicMLSpec, LogicMLGenerator, generate_logicml +1
│   ├── markdown_format.py: [python] MarkdownSpec, MarkdownHybridGenerator, generate_markdown_hybrid +1
│   ├── mcp_server.py: [python] handle_request, call_tool, run_server
│   ├── metrics.py: [python] TextMetrics, StructuralMetrics, SemanticMetrics +5
│   ├── models.py: [python] FunctionInfo, ClassInfo, TypeInfo +11
│   ├── parsers.py: [python] TREE_SITTER_AVAILABLE, TreeSitterParser, UniversalParser +1
│   ├── project_reproducer.py: [python] SUPPORTED_EXTENSIONS, FileResult, ProjectResult +2
│   ├── prompts.py: [python] FORMAT_HINTS, get_reproduction_prompt, get_review_prompt +1
│   ├── quality.py: [python] QualityIssue, QualityReport, QualityAnalyzer +2
│   ├── refactor.py: [python] DuplicateGroup, RefactoringSuggestion, RefactoringReport +5
│   ├── reproducer.py: [python] ReproductionStatus, FileValidation, ReproductionResult +4
│   ├── reproduction.py: [python] generate_file_gherkin, compare_code, extract_code_block +1
│   ├── schemas/
│   │   ├── __init__.py: [python]
│   │   ├── json_schema.py: [python] JSONMethodSchema, JSONClassSchema, JSONFunctionSchema +4
│   │   ├── logicml_schema.py: [python] LogicMLMethod, LogicMLClass, LogicMLModule +4
│   │   ├── markdown_schema.py: [python] MarkdownMethod, MarkdownClass, MarkdownModule +3
│   │   └── yaml_schema.py: [python] MethodSchema, ClassSchema, FunctionSchema +3
│   ├── shared_utils.py: [python] compact_imports, deduplicate_imports, TYPE_ABBREVIATIONS +12
│   ├── similarity.py: [python] RAPIDFUZZ_AVAILABLE, SimilarityDetector, is_rapidfuzz_available +1
│   ├── terminal.py: [python] COLORS, ShellRenderer, get_renderer +2
│   ├── tools/
│   │   └── __init__.py: [python]
│   ├── toon_format.py: [python] TOONGenerator, TOONParser, generate_toon +1
│   ├── universal.py: [python] ElementType, Language, Parameter +6
│   └── utils.py: [python] estimate_tokens, write_text_atomic, cleanup_generated_root
├── examples/
│   ├── 01_quick_start.py: [python] main
│   ├── 02_refactoring.py: [python] main
│   ├── 03_reproduction.py: [python] main
│   ├── 04_project.py: [python] main
│   ├── 05_llm_integration.py: [python] main
│   ├── 06_metrics.py: [python] analyze_file, main
│   ├── 06_metrics_simple.py: [python]
│   ├── 08_format_benchmark.py: [python] print_format_comparison, print_per_file_results, main
│   ├── 09_async_benchmark.py: [python] print_results, main
│   ├── 10_function_reproduction.py: [python] print_results, main
│   ├── 11_token_benchmark.py: [python] print_token_efficiency, main
│   ├── 12_comprehensive_analysis.py: [python] ALL_FORMATS, print_comprehensive_analysis, main
│   ├── 12_comprehensive_analysis_simple.py: [python]
│   ├── 13_project_benchmark.py: [python] print_project_results, main
│   ├── 14_repeatability_test.py: [python] RunResult, RepeatabilityResult, generate_spec +8
│   ├── 15_unified_benchmark.py: [python] print_format_results, print_function_results, print_project_results +1
│   ├── 16_terminal_demo.py: [python] demo_headings, demo_codeblocks, demo_status_messages +7
│   ├── behavioral_benchmark.py: [python] CaseResult, FunctionBehaviorResult, main
│   ├── benchmark_report.py: [python] Artifact, main
│   ├── benchmark_summary.py: [python] main
│   ├── code2logic/
│   │   └── sample_project/
│   │       ├── __init__.py: [python]
│   │       ├── api/
│   │       │   ...
│   │       ├── calculator.py: [python] Calculator, factorial
│   │       └── models/
│   │           ...
│   ├── duplicate_detection.py: [python]
│   ├── token_efficiency.py: [python]
│   └── windsurf_mcp_integration.py: [python]
├── logic2code/
│   ├── __init__.py: [python]
│   ├── __main__.py: [python]
│   ├── cli.py: [python] main
│   ├── examples/
│   │   ├── 01_quickstart.py: [python] example_basic_generation, example_with_config, example_stubs_only +1
│   │   └── 02_llm_enhanced.py: [python] example_llm_generation, example_hybrid_generation, example_compare_outputs
│   ├── generator.py: [python] GeneratorConfig, GenerationResult, CodeGenerator
│   ├── renderers.py: [python] RenderConfig, BaseRenderer, PythonRenderer
│   └── tests/
│       ├── __init__.py: [python]
│       └── test_basic.py: [python] test_import_logic2code, test_import_generator, test_config_defaults +2
├── logic2test/
│   ├── __init__.py: [python]
│   ├── __main__.py: [python]
│   ├── cli.py: [python] main
│   ├── examples/
│   │   ├── 01_quickstart.py: [python] example_basic_generation, example_with_config, example_generate_all_types
│   │   └── 02_custom_templates.py: [python] example_function_test, example_class_test, example_dataclass_test +1
│   ├── generator.py: [python] GeneratorConfig, GenerationResult, TestGenerator
│   ├── parsers.py: [python] FunctionSpec, ClassSpec, ModuleSpec +2
│   ├── templates.py: [python] TestTemplate
│   └── tests/
│       ├── __init__.py: [python]
│       └── test_basic.py: [python] test_import_logic2test, test_import_generator, test_config_defaults +1
├── lolm/
│   ├── __init__.py: [python]
│   ├── __main__.py: [python]
│   ├── cli.py: [python] cmd_status, cmd_set_provider, cmd_set_model +8
│   ├── clients.py: [python] LLMRateLimitError, OpenRouterClient, OllamaClient +3
│   ├── config.py: [python] RECOMMENDED_MODELS, DEFAULT_MODELS, DEFAULT_PROVIDER_PRIORITIES +13
│   ├── examples/
│   │   ├── 01_quickstart.py: [python] example_simple_client, example_specific_provider, example_manager +1
│   │   ├── 02_configuration.py: [python] show_defaults, show_recommended_models, show_current_config +2
│   │   └── 03_code_generation.py: [python] SYSTEM_PROMPT, generate_function, generate_class +2
│   ├── manager.py: [python] ProviderInfo, LLMManager, get_client +1
│   ├── provider.py: [python] LLMProviderStatus, GenerateOptions, LLMResponse +3
│   ├── rotation.py: [python] ProviderState, RateLimitType, RateLimitInfo +6
│   └── tests/
│       ├── __init__.py: [python]
│       └── test_basic.py: [python] test_import_lolm, test_import_config, test_import_clients +3
├── raport/
│   └── mermaid-init.js: [javascript] renderMermaid, convertMermaidCodeBlocks
├── scripts/
│   └── configure_llm.py: [python] CONFIG_DIR, CONFIG_FILE, log +10
└── tests/
    ├── __init__.py: [python]
    ├── conftest.py: [python] sample_python_code, sample_javascript_code, sample_java_code +6
    ├── samples/
    │   ├── sample_algorithms.py: [python] T, binary_search, quicksort +9
    │   ├── sample_api.py: [python] APIResponse, User, APIError +4
    │   ├── sample_async.py: [python] T, Task, TaskResult +7
    │   ├── sample_class.py: [python] Calculator
    │   ├── sample_csharp.cs: [csharp] IHasId, User
    │   ├── sample_dataclasses.py: [python] User, Product, Order +1
    │   ├── sample_enum.py: [python] Status, Priority, Color +2
    │   ├── sample_functions.py: [python] calculate_total, filter_by_status, merge_configs +5
    │   ├── sample_go.go: [go] User, Product, Order +9
    │   ├── sample_java.java: [java] SampleJava, Identifiable, Status +1
    │   ├── sample_javascript.js: [javascript] debounce, filterBy, calculateTotal +4
    │   ├── sample_javascript_advanced.js: [javascript] FileProcessor, processItem, shouldIgnore +8
    │   ├── sample_pydantic.py: [python] TaskStatus, Task, TaskQueue +1
    │   ├── sample_reexport/
    │   │   ├── __init__.py: [python]
    │   │   ├── exceptions.py: [python] ValidationError, ProcessingError
    │   │   ├── models.py: [python] User, Order, Product
    │   │   └── utils.py: [python] process_data, validate_input
    │   ├── sample_rust.rs: [rust] User, Product, Order +20
    │   ├── sample_sql.sql: [sql] users, products, orders +4
    │   ├── sample_sql_dsl.py: [python] SQLOperator, JoinType, Condition +7
    │   ├── sample_ts_reexport/
    │   │   ├── index.ts: [typescript] add, multiply
    │   │   ├── math.ts: [typescript] add, multiply
    │   │   └── types.ts: [typescript] User, Result
    │   └── sample_typescript.ts: [typescript] createUser, OrderItem, User +10
    ├── test_analyzer.py: [python] TestProjectAnalyzer, TestAnalyzeProjectFunction, TestGetLibraryStatus
    ├── test_e2e_projects.py: [python] test_e2e_pipeline_code2logic_logic2test_logic2code, test_e2e_logic2test_on_examples_input, test_e2e_logic2code_on_examples_input
    ├── test_error_handling.py: [python] error_handler, strict_handler, temp_project +7
    ├── test_formats.py: [python] sample_code, sample_project, samples_project +18
    ├── test_generators.py: [python] sample_project, TestMarkdownGenerator, TestCompactGenerator +1
    ├── test_intent.py: [python] make_function, make_class, make_module +2
    ├── test_llm_priority.py: [python] test_get_client_auto_prefers_override_provider_on_tie, test_get_client_auto_model_first_uses_model_priority
    ├── test_llm_profiler.py: [python] TestLLMProfile, TestDefaultProfiles, TestProfileStorage +4
    ├── test_parser_integrity.py: [python] parser, parse_python, TestFunctionNameExtraction +12
    ├── test_reproduction.py: [python] TestYAMLGenerator, TestGherkinGenerator, TestMarkdownGenerator +5
    ├── test_shared_utils.py: [python] TestCompactImports, TestDeduplicateImports, TestAbbreviateType +9
    └── test_yaml_compact.py: [python] sample_project, TestYAMLShortKeys, TestSelfRemoval +5
```

## 🔗 Dependencies

```yaml
__init__: [analyzer, adaptive, base, benchmark, chunked_reproduction]
__main__: [cli]
adaptive: [file_formats, reproduction, llm_clients]
analyzer: [parsers, models, dependency, intent]
base_generator: [models]
benchmark: [file_formats, gherkin, analyzer, generators]
benchmarks/__init__: [results, runner, common]
common: [function_logic, gherkin, models, generators, toon_format, +2]
runner: [metrics, analyzer, common, terminal, llm_clients, +1]
chunked_reproduction: [utils]
core/__init__: [dependency, models, analyzer, errors]
dependency: [models]
formats/__init__: [file_formats, gherkin, generators, toon_format, logicml, +1]
function_logic: [models, shared_utils, toon_format]
generators: [models, shared_utils]
gherkin: [models]
integrations/__init__: [mcp_server]
llm: [llm_clients]
llm/__init__: [intent, llm_clients]
llm_profiler: [utils]
logicml: [models, shared_utils]
markdown_format: [models, gherkin, generators]
parsers: [models, intent]
project_reproducer: [universal, llm_clients]
quality: [models]
refactor: [llm_clients, analyzer, code_review]
reproduction: [llm_clients]
schemas/__init__: [yaml_schema, json_schema, logicml_schema, markdown_schema]
similarity: [models]
tools/__init__: [refactor, adaptive, benchmark, code_review]
```

## 📄 Modules

### 📂 code2logic

#### `__init__.py`

```yaml
path: code2logic/__init__.py
lang: python | lines: 392/440
imports: [adaptive.LLM_CAPABILITIES, adaptive.AdaptiveReproducer, adaptive.AdaptiveResult, adaptive.get_llm_capabilities, analyzer.ProjectAnalyzer... +15]
constants: [conditional:llm_clients.BaseLLMClient, conditional:llm_clients.LiteLLMClient, conditional:llm_clients.OllamaLocalClient, conditional:llm_clients.OpenRouterClient, conditional:llm_clients.get_client]
```

> Code2Logic - Convert source code to logical representation for LLM analysis.

**Functions:**

- `analyze_quality(target)` — processes quality
- `reproduce_project(source:str)` — reproduce project

---

#### `__main__.py`

```yaml
path: code2logic/__main__.py
lang: python | lines: 12/16
imports: [cli.main]
constants: [conditional:cli.main]
```

> Allow running code2logic as a module: python -m code2logic

Usage:
    python -m...

---

#### `adaptive.py`

```yaml
path: code2logic/adaptive.py
lang: python | lines: 469/614
imports: [dataclasses.dataclass, pathlib.Path, typing.Any, typing.Dict, typing.List... +9]
constants: [LLM_CAPABILITIES, conditional:dotenv.load_dotenv]
```

> Adaptive Format System for LLM-based Code Reproduction.

**class `ChunkInfo`**

> Information about a code chunk.


**class `AdaptiveResult`**

> Result of adaptive reproduction.


**class `AdaptiveReproducer`**

> Adaptive code reproduction with LLM capability detection.

```yaml
methods:
  __init__(client:BaseLLMClient=None, model:str=None)  # creates
  select_format(file_path:Path, content:str) -> str  # retrieves format
  should_chunk(content:str) -> bool  # checks chunk
  chunk_content(content:str, file_path:Path) -> List[ChunkInfo]  # chunk content
  generate_chunk_spec(chunk:ChunkInfo, format_name:str) -> str  # creates chunk spec
  reproduce(file_path:str, output_dir:str=None) -> AdaptiveResult  # reproduce
```

**Functions:**

- `get_llm_capabilities(model:str) -> Dict[str, Any]` — retrieves llm capabilities

---

#### `analyzer.py`

```yaml
path: code2logic/analyzer.py
lang: python | lines: 410/494
imports: [logging, os, subprocess, sys, time... +15]
```

> Main project analyzer orchestrating all analysis components.

**class `ProjectAnalyzer`**

> Main class for analyzing software projects.

```yaml
methods:
  __init__(root_path:str, use_treesitter:bool=True, verbose:bool=False, include_private:bool=False, ...+2)  # creates
  analyze() -> ProjectInfo  # processes
  get_statistics() -> Dict  # retrieves statistics
```

**Functions:**

- `analyze_project(path:str, use_treesitter:bool=True, verbose:bool=False) -> ProjectInfo` — processes project
- `get_library_status() -> Dict[str, bool]` — retrieves library status

---

#### `base.py`

```yaml
path: code2logic/base.py
lang: python | lines: 49/69
imports: [logging]
```

> Base classes and mixins for code2logic.

**class `VerboseMixin`**

> Mixin providing verbose logging functionality.

```yaml
methods:
  __init__(verbose:bool=False)  # creates
  log(msg:str, level:str='info')  # logs
  debug(msg:str)  # debug
  info(msg:str)  # info
  warn(msg:str)  # warn
  error(msg:str)  # error
```

**class `BaseParser`(VerboseMixin)**

> Base class for code parsers.

```yaml
methods:
  __init__(verbose:bool=False)  # creates
  parse(content:str, language:str=None)  # parses
  parse_file(path:str)  # parses file
```

**class `BaseGenerator`(VerboseMixin)**

> Base class for output generators.

```yaml
methods:
  __init__(verbose:bool=False)  # creates
  generate(project, detail:str='full') -> str  # creates
```

---

#### `base_generator.py`

```yaml
path: code2logic/base_generator.py
lang: python | lines: 5/7
imports: [typing.Protocol, typing.Any, models.ProjectInfo]
```

**class `ProjectGenerator`(Protocol)**

```yaml
methods:
  generate(project:ProjectInfo) -> Any  # creates
```

---

#### `benchmark.py`

```yaml
path: code2logic/benchmark.py
lang: python | lines: 349/448
imports: [json, time, dotenv.load_dotenv, dataclasses.asdict, dataclasses.dataclass... +15]
constants: [FORMAT_PROMPTS, conditional:dotenv.load_dotenv]
```

> Reproduction Benchmark for Code2Logic.

**class `FormatResult`**

> Result for a single format test.


**class `BenchmarkResult`**

> Complete benchmark result.


**class `ReproductionBenchmark`**

> Benchmark reproduction quality across formats.

```yaml
methods:
  __init__(client:BaseLLMClient=None)  # creates
  generate_spec(file_path:Path, format_name:str, detail:str='full') -> str  # creates spec
  reproduce_with_format(file_path:Path, format_name:str, original_code:str) -> FormatResult  # reproduce with format
  run_single(file_path:str, formats:List[str]=None) -> BenchmarkResult  # starts single
  run_all(files:List[str], output_dir:str=None) -> Dict[str, Any]  # starts all
```

**Functions:**

- `run_benchmark(files:List[str], output_dir:str='benchmark_results', provider:str=None, model:str=None) -> Dict[str, Any]` — starts benchmark

---

#### `chunked_reproduction.py`

```yaml
path: code2logic/chunked_reproduction.py
lang: python | lines: 358/480
imports: [re, dataclasses.dataclass, typing.List, typing.Optional, utils.estimate_tokens]
constants: [LLM_CONTEXT_LIMITS]
```

> Chunked Reproduction for Smaller LLMs.

**class `Chunk`**

> A chunk of specification for reproduction.


**class `ChunkedSpec`**

> Chunked specification.


**class `ChunkedResult`**

> Result of chunked reproduction.


**class `ChunkedReproducer`**

> Reproduce code from chunked specifications.

```yaml
methods:
  __init__(client, model_name:str='default', max_tokens:Optional[int]=None)  # creates
  reproduce(spec:str, fmt:str, file_name:str) -> ChunkedResult  # reproduce
```

**Functions:**

- `get_llm_limit(model_name:str) -> int` — retrieves llm limit
- `chunk_yaml_spec(spec:str, max_tokens:int=2000) -> List[Chunk]` — chunk yaml spec
- `chunk_gherkin_spec(spec:str, max_tokens:int=2000) -> List[Chunk]` — chunk gherkin spec
- `chunk_markdown_spec(spec:str, max_tokens:int=2000) -> List[Chunk]` — chunk markdown spec
- `chunk_spec(spec:str, fmt:str, max_tokens:int=2000) -> ChunkedSpec` — chunk spec
- `get_chunk_prompt(chunk:Chunk, fmt:str, file_name:str, chunk_num:int, ...+1) -> str` — retrieves chunk prompt
- `merge_chunk_codes(codes:List[str], file_name:str) -> str` — merges chunk codes
- `auto_chunk_reproduce(spec:str, fmt:str, file_name:str, client, ...+1) -> ChunkedResult` — auto chunk reproduce
- `adaptive_chunk_reproduce(spec:str, fmt:str, file_name:str, client, ...+2) -> ChunkedResult` — adaptive chunk reproduce

---

#### `cli.py`

```yaml
path: code2logic/cli.py
lang: python | lines: 907/1100
imports: [argparse, json, logging, os, signal... +5]
```

> Command-line interface for Code2Logic.

**class `Colors`**


**class `Logger`**

> Enhanced logger for CLI output.

```yaml
methods:
  __init__(verbose:bool=False, debug:bool=False)  # creates
  info(msg:str)  # info
  success(msg:str)  # success
  warning(msg:str)  # warning
  error(msg:str)  # error
  step(msg:str)  # step
  detail(msg:str)  # detail
  debug_msg(msg:str)  # debug msg
  stats(label:str, value)  # stats
  separator()  # separator
  header(msg:str)  # header
```

**Functions:**

- `ensure_dependencies()` — ensure dependencies
- `main(argv=None)` — main

---

#### `code_review.py`

```yaml
path: code2logic/code_review.py
lang: python | lines: 205/272
imports: [collections.defaultdict, typing.Any, typing.Dict, typing.List]
constants: [SECURITY_PATTERNS, PERFORMANCE_PATTERNS, COMPLEXITY_HIGH, COMPLEXITY_MEDIUM, LINES_MAX]
```

> Code review utilities.

**class `CodeReviewer`**

> Automated code review with optional LLM enhancement.

```yaml
methods:
  __init__(client=None)  # creates
  review(project, focus:str='all') -> Dict[str, Any]  # review
  generate_report(results:Dict[str,Any], project_name:str='Project') -> str  # creates report
```

**Functions:**

- `analyze_code_quality(project) -> Dict[str, List[Dict]]` — processes code quality
- `check_security_issues(project) -> Dict[str, List[Dict]]` — checks security issues
- `check_performance_issues(project) -> Dict[str, List[Dict]]` — checks performance issues

---

#### `config.py`

```yaml
path: code2logic/config.py
lang: python | lines: 174/249
imports: [json, os, pathlib.Path, typing.Any, typing.Dict... +1]
constants: [SHELL_COMMANDS]
```

> Configuration management for Code2Logic.

**class `Config`**

> Configuration manager for Code2Logic.

```yaml
methods:
  __init__(env_file:str=None)  # creates
  get_api_key(provider:str) -> Optional[str]  # retrieves api key
  get_model(provider:str) -> str  # retrieves model
  get_ollama_host() -> str  # retrieves ollama host
  get_default_provider() -> str  # retrieves default provider
  is_verbose() -> bool  # is verbose
  get_project_name() -> str  # retrieves project name
  get_cache_dir() -> Path  # retrieves cache dir
  list_configured_providers() -> Dict[str, bool]  # list configured providers
  to_dict() -> Dict[str, Any]  # converts dict
```

**Functions:**

- `load_env()` — retrieves env
- `get_api_key(provider:str) -> Optional[str]` — retrieves api key
- `get_model(provider:str) -> str` — retrieves model

---

#### `dependency.py`

```yaml
path: code2logic/dependency.py
lang: python | lines: 187/247
imports: [pathlib.Path, typing.Dict, typing.List, models.DependencyNode, models.ModuleInfo... +1]
constants: [NETWORKX_AVAILABLE, conditional:networkx]
```

> Dependency graph analyzer using NetworkX.

**class `DependencyAnalyzer`**

> Analyzes dependency graphs using NetworkX.

```yaml
methods:
  __init__()  # creates
  build_graph(modules:List[ModuleInfo]) -> Dict[str, List[str]]  # creates graph
  analyze_metrics() -> Dict[str, DependencyNode]  # processes metrics
  get_entrypoints() -> List[str]  # retrieves entrypoints
  get_hubs() -> List[str]  # retrieves hubs
  detect_cycles() -> List[List[str]]  # detect cycles
  get_strongly_connected_components() -> List[List[str]]  # retrieves strongly connected components
  get_dependency_depth(module_path:str) -> int  # retrieves dependency depth
```

**Functions:**

- `is_networkx_available() -> bool` — is networkx available

---

#### `errors.py`

```yaml
path: code2logic/errors.py
lang: python | lines: 371/437
imports: [logging, dataclasses.dataclass, dataclasses.field, enum.Enum, pathlib.Path... +5]
```

> Error handling for Code2Logic.

**enum `ErrorSeverity`**

**enum `ErrorType`**

**class `ErrorSeverity`(Enum)**

> Error severity levels.


**class `ErrorType`(Enum)**

> Types of errors that can occur during analysis.


**class `AnalysisError`**

> Represents an error during analysis.

```yaml
methods:
  to_dict() -> Dict[str, Any]  # converts dict
```

**class `AnalysisResult`**

> Result of analysis with errors tracked.

```yaml
methods:
  add_error(error:AnalysisError)  # creates error
  has_errors() -> bool  # has errors
  summary() -> str  # summary
```

**class `ErrorHandler`**

> Handles errors during analysis with configurable behavior.

```yaml
methods:
  __init__(mode:str='lenient', max_file_size_mb:float=10.0, timeout_seconds:float=30.0, logger:Optional[Any]=None)  # creates
  reset()  # reset
  handle_error(error_type:ErrorType, path:str, message:str, exception:Optional[Exception]=None, ...+1) -> bool  # handles error
  safe_read_file(path:Path) -> Optional[str]  # safe read file
  safe_write_file(path:Path, content:str) -> bool  # safe write file
  safe_parse(path:str, content:str, parser_func:Callable) -> Any  # safe parse
```

**Functions:**

- `create_error_handler(mode:str='lenient', max_file_size_mb:float=10.0) -> ErrorHandler` — creates error handler

---

#### `file_formats.py`

```yaml
path: code2logic/file_formats.py
lang: python | lines: 278/352
imports: [json, pathlib.Path, typing.Any, typing.Dict]
```

> File-specific format generators for better reproduction.

**Functions:**

- `generate_file_csv(file_path:Path) -> str` — creates file csv
- `generate_file_json(file_path:Path) -> str` — creates file json
- `generate_file_yaml(file_path:Path) -> str` — creates file yaml

---

#### `function_logic.py`

```yaml
path: code2logic/function_logic.py
lang: python | lines: 326/401
imports: [typing.List, typing.Tuple, models.FunctionInfo, models.ProjectInfo, shared_utils.remove_self_from_params... +2]
```

**class `FunctionLogicGenerator`**

```yaml
methods:
  __init__(verbose:bool=False) -> None  # creates
  generate(project:ProjectInfo, detail:str='full') -> str  # creates
  generate_json(project:ProjectInfo, detail:str='full') -> str  # creates json
  generate_yaml(project:ProjectInfo, detail:str='full') -> str  # creates yaml
  generate_toon(project:ProjectInfo, detail:str='full', no_repeat_name:bool=False, no_repeat_details:bool=False, ...+2) -> str  # creates toon
  generate_toon_schema() -> str  # creates toon schema
```

---

#### `generators.py`

```yaml
path: code2logic/generators.py
lang: python | lines: 1787/2179
imports: [json, collections.defaultdict, pathlib.Path, typing.List, typing.Optional... +12]
```

> Output generators for Code2Logic analysis results.

**class `MarkdownGenerator`**

> Generates Markdown output for project analysis.

```yaml
methods:
  generate(project:ProjectInfo, detail_level:str='standard') -> str  # creates
```

**class `CompactGenerator`**

> Generates ultra-compact output for token efficiency.

```yaml
methods:
  generate(project:ProjectInfo) -> str  # creates
```

**class `JSONGenerator`**

> Generates JSON output for machine processing.

```yaml
methods:
  generate(project:ProjectInfo, flat:bool=False, detail:str='standard') -> str  # creates
  generate_from_module(module:ModuleInfo, detail:str='full') -> str  # creates from module
```

**class `YAMLGenerator`**

> Generates YAML output for human-readable representation.

```yaml
methods:
  generate(project:ProjectInfo, flat:bool=False, detail:str='standard', compact:bool=True) -> str  # creates
  generate_schema(format_type:str='compact') -> str  # creates schema
  generate_hybrid(project:ProjectInfo, detail:str='standard') -> str  # creates hybrid
  generate_from_module(module:ModuleInfo, detail:str='full') -> str  # creates from module
```

**class `CSVGenerator`**

> Generates CSV output optimized for LLM processing.

```yaml
methods:
  generate(project:ProjectInfo, detail:str='standard') -> str  # creates
```

**Functions:**

- `bytes_to_kb(bytes_value:int) -> float` — bytes to kb

---

#### `gherkin.py`

```yaml
path: code2logic/gherkin.py
lang: python | lines: 764/979
imports: [re, collections.defaultdict, dataclasses.dataclass, typing.Any, typing.Dict... +4]
```

> Gherkin/BDD Generator for Code2Logic.

**class `GherkinScenario`**

> Represents a single Gherkin scenario.


**class `GherkinFeature`**

> Represents a Gherkin feature file.


**class `StepDefinition`**

> Represents a step definition.


**class `GherkinGenerator`**

> Generates Gherkin feature files from code analysis.

```yaml
methods:
  __init__(language:str='en')  # creates
  generate(project:ProjectInfo, detail:str='standard', group_by:str='domain') -> str  # creates
  generate_test_scenarios(project:ProjectInfo, group_by:str='domain') -> List[GherkinFeature]  # creates test scenarios
  get_step_definitions() -> List[StepDefinition]  # retrieves step definitions
```

**class `StepDefinitionGenerator`**

> Generates step definition stubs from Gherkin features.

```yaml
methods:
  generate_pytest_bdd(features:List[GherkinFeature]) -> str  # creates pytest bdd
  generate_behave(features:List[GherkinFeature]) -> str  # creates behave
  generate_cucumber_js(features:List[GherkinFeature]) -> str  # creates cucumber js
```

**class `CucumberYAMLGenerator`**

> Generates Cucumber YAML configuration and test data.

```yaml
methods:
  generate(project:ProjectInfo, detail:str='standard') -> str  # creates
```

**Functions:**

- `csv_to_gherkin(csv_content:str, language:str='en') -> str` — csv to gherkin
- `gherkin_to_test_data(gherkin_content:str) -> Dict[str, Any]` — gherkin to test data

---

#### `intent.py`

```yaml
path: code2logic/intent.py
lang: python | lines: 429/562
imports: [re, dataclasses.dataclass, dataclasses.field, enum.Enum, enum.auto... +7]
constants: [conditional:nltk, conditional:nltk.stem.WordNetLemmatizer, conditional:spacy]
```

> Enhanced Intent Generator with NLP support.

**enum `IntentType`**

**class `IntentType`(Enum)**

> Types of user intents for code analysis.


**class `Intent`**

> Represents a detected user intent.


**class `EnhancedIntentGenerator`**

> Generator intencji z NLP - lemmatyzacja, ekstrakcja z docstringów.

```yaml
methods:
  __init__(lang:str='en')  # creates
  generate(name:str, docstring:Optional[str]=None) -> str  # creates
  get_available_features() -> dict[str, bool]  # retrieves available features
```

**class `IntentAnalyzer`**

> Analyzes user queries to detect intent and provide suggestions.

```yaml
methods:
  __init__()  # creates
  analyze_intent(query:str, project:Any) -> List[Intent]  # processes intent
  detect_code_smells(project:Any) -> List[dict]  # detect code smells
  suggest_refactoring(target:str, project:Any) -> List[str]  # suggest refactoring
```

---

#### `llm.py`

```yaml
path: code2logic/llm.py
lang: python | lines: 287/374
imports: [json, os, dataclasses.dataclass, importlib.util.find_spec, typing.Any... +5]
```

> LLM Integration for Code2Logic

Provides integration with local Ollama and LiteL...

**class `LLMConfig`**

> Configuration for LLM backend.


**class `CodeAnalyzer`**

> LLM-powered code analysis for Code2Logic.

```yaml
methods:
  __init__(model:str=None, provider:str=None, base_url:str=None, api_key:str=None)  # creates
  is_available() -> bool  # is available
  suggest_refactoring(project) -> list[dict[str, Any]]  # suggest refactoring
  find_semantic_duplicates(project) -> list[dict[str, Any]]  # retrieves semantic duplicates
  generate_code(project, target_lang:str, module_filter:Optional[str]=None) -> dict[str, str]  # creates code
  translate_function(name:str, signature:str, intent:str, source_lang:str, ...+1) -> str  # converts function
```

**Functions:**

- `get_available_backends() -> dict[str, bool]` — retrieves available backends

---

#### `llm_clients.py`

```yaml
path: code2logic/llm_clients.py
lang: python | lines: 209/274
imports: [json, os, typing.Any, typing.Optional, lolm.DEFAULT_MODELS... +13]
constants: [conditional:lolm.DEFAULT_MODELS, conditional:lolm.DEFAULT_PROVIDER_PRIORITIES, conditional:lolm.RECOMMENDED_MODELS, conditional:lolm.BaseLLMClient, conditional:lolm.LiteLLMClient]
```

> LLM Client implementations for various providers.

**Functions:**

- `get_priority_mode() -> str` — retrieves priority mode
- `get_effective_provider_priorities() -> dict[str, int]` — retrieves effective provider priorities

---

#### `llm_profiler.py`

```yaml
path: code2logic/llm_profiler.py
lang: python | lines: 490/649
imports: [hashlib, json, time, dataclasses.asdict, dataclasses.dataclass... +7]
constants: [PROFILE_TEST_CASES]
```

> LLM Profiler for Adaptive Code Reproduction.

**class `LLMProfile`**

> Profile of LLM capabilities for code reproduction.

```yaml
methods:
  __post_init__()  # creates init
```

**class `ProfileTestResult`**

> Result of a single profile test.


**class `LLMProfiler`**

> Profile LLM capabilities for code reproduction.

```yaml
methods:
  __init__(client, verbose:bool=True)  # creates
  run_profile(quick:bool=False) -> LLMProfile  # starts profile
```

**class `AdaptiveChunker`**

> Adaptive chunking based on LLM profile.

```yaml
methods:
  __init__(profile:Optional[LLMProfile]=None)  # creates
  get_optimal_settings() -> dict[str, Any]  # retrieves optimal settings
  chunk_spec(spec:str, format:str='yaml') -> list[dict[str, Any]]  # chunk spec
  recommend_format(spec_size_tokens:int) -> str  # recommend format
  estimate_chunks_needed(spec_size_tokens:int) -> int  # estimate chunks needed
```

**Functions:**

- `load_profiles() -> dict[str, LLMProfile]` — retrieves profiles
- `save_profile(profile:LLMProfile) -> None` — caches profile
- `get_profile(provider:str, model:str) -> Optional[LLMProfile]` — retrieves profile
- `get_or_create_profile(provider:str, model:str) -> LLMProfile` — retrieves or create profile
- `profile_llm(client, quick:bool=False) -> LLMProfile` — profile llm
- `get_adaptive_chunker(provider:str, model:str) -> AdaptiveChunker` — retrieves adaptive chunker

---

#### `logicml.py`

```yaml
path: code2logic/logicml.py
lang: python | lines: 305/400
imports: [dataclasses.dataclass, pathlib.Path, typing.Dict, typing.List, typing.Optional... +8]
constants: [LOGICML_EXAMPLE]
```

> LogicML Format Generator for Code2Logic.

**class `LogicMLSpec`**

> LogicML specification output.


**class `LogicMLGenerator`**

> Generates LogicML format - optimized for LLM code reproduction.

```yaml
methods:
  __init__(verbose:bool=False) -> None  # creates
  generate(project:ProjectInfo, detail:str='standard', level:str='typed') -> LogicMLSpec  # creates
```

**Functions:**

- `generate_logicml(project:ProjectInfo, detail:str='standard') -> str` — creates logicml

---

#### `markdown_format.py`

```yaml
path: code2logic/markdown_format.py
lang: python | lines: 265/365
imports: [dataclasses.dataclass, pathlib.Path, typing.Dict, typing.List, generators.YAMLGenerator... +2]
```

> Markdown Hybrid Format Generator for Code2Logic.

**class `MarkdownSpec`**

> Markdown specification for a project.


**class `MarkdownHybridGenerator`**

> Generates optimized Markdown hybrid format.

```yaml
methods:
  __init__(verbose:bool=False)  # creates
  generate(project:ProjectInfo, detail:str='full') -> MarkdownSpec  # creates
```

**Functions:**

- `generate_markdown_hybrid(project:ProjectInfo, detail:str='full') -> str` — creates markdown hybrid
- `generate_file_markdown(file_path:str) -> str` — creates file markdown

---

#### `mcp_server.py`

```yaml
path: code2logic/mcp_server.py
lang: python | lines: 297/361
imports: [json, sys, __version__]
```

> MCP (Model Context Protocol) Server for Code2Logic

Provides Code2Logic function...

**Functions:**

- `handle_request(request:dict) -> dict` — handles request
- `call_tool(tool_name:str, arguments:dict) -> str` — call tool
- `run_server()` — starts server

---

#### `metrics.py`

```yaml
path: code2logic/metrics.py
lang: python | lines: 479/640
imports: [difflib, logging, re, collections.Counter, dataclasses.asdict... +6]
```

> Advanced Metrics for Code Reproduction Quality.

**class `TextMetrics`**

> Text-level similarity metrics.


**class `StructuralMetrics`**

> Structural code metrics.


**class `SemanticMetrics`**

> Semantic preservation metrics.


**class `FormatMetrics`**

> Format-specific efficiency metrics.


**class `ReproductionResult`**

> Complete reproduction analysis result.

```yaml
methods:
  to_dict() -> Dict[str, Any]  # converts dict
  to_report() -> str  # converts report
```

**class `ReproductionMetrics`**

> Analyze reproduction quality with multiple metrics.

```yaml
methods:
  __init__(verbose:bool=False)  # creates
  analyze(original:str, generated:str, spec:str='', format_name:str='', ...+1) -> ReproductionResult  # processes
```

**Functions:**

- `analyze_reproduction(original:str, generated:str, spec:str='', format_name:str='', ...+1) -> ReproductionResult` — processes reproduction
- `compare_formats(original:str, results:Dict[str,Tuple[str,str]], verbose:bool=False) -> Dict[str, Any]` — compare formats

---

#### `models.py`

```yaml
path: code2logic/models.py
lang: python | lines: 296/339
imports: [dataclasses.dataclass, dataclasses.field, typing.Dict, typing.List, typing.Optional]
```

> Data models for Code2Logic.

**class `FunctionInfo`**

> Information about a function or method.


**class `ClassInfo`**

> Information about a class or interface.


**class `TypeInfo`**

> Information about a type alias, interface, or enum.


**class `ModuleInfo`**

> Information about a source file/module.


**class `DependencyNode`**

> Node in the dependency graph with metrics.


**class `ProjectInfo`**

> Complete project analysis results.


**class `ConstantInfo`**

> Module-level constant information.


**class `FieldInfo`**

> Dataclass field information.


**class `AttributeInfo`**

> Instance attribute information (self.x = ...).


**class `PropertyInfo`**

> Property information (@property, @x.setter).


**class `OptionalImport`**

> Try/except import block information.


**class `ClassInfo`**

> Information about a class or interface.


**class `FunctionInfo`**

> Information about a function or method.


**class `ModuleInfo`**

> Information about a source file/module.


---

#### `parsers.py`

```yaml
path: code2logic/parsers.py
lang: python | lines: 2265/2679
imports: [ast, re, textwrap, typing.List, typing.Optional... +12]
constants: [TREE_SITTER_AVAILABLE, conditional:tree_sitter_javascript, conditional:tree_sitter_python, conditional:tree_sitter.Language, conditional:tree_sitter.Parser]
```

> Code parsers for multiple languages.

**class `_PyFunctionBodyAnalyzer`(ast.NodeVisitor)**

```yaml
methods:
  __init__()  # creates
  visit_Call(node)  # visit call
  visit_Raise(node)  # visit raise
  visit_If(node)  # visit if
  visit_For(node)  # visit for
  visit_AsyncFor(node)  # visit asyncfor
  visit_While(node)  # visit while
  visit_IfExp(node)  # visit ifexp
  visit_BoolOp(node)  # visit boolop
  visit_Try(node)  # visit try
  visit_comprehension(node)  # visit comprehension
  visit_Match(node)  # visit match
  # ... +4 more
```

**class `TreeSitterParser`**

> Parser using Tree-sitter for high-accuracy AST parsing.

```yaml
methods:
  __init__()  # creates
  is_available(language:str) -> bool  # is available
  get_supported_languages() -> List[str]  # retrieves supported languages
  parse(filepath:str, content:str, language:str) -> Optional[ModuleInfo]  # parses
```

**class `UniversalParser`**

> Fallback parser using Python AST and regex.

```yaml
methods:
  __init__()  # creates
  parse(filepath:str, content:str, language:str) -> Optional[ModuleInfo]  # parses
```

**Functions:**

- `is_tree_sitter_available() -> bool` — is tree sitter available

---

#### `project_reproducer.py`

```yaml
path: code2logic/project_reproducer.py
lang: python | lines: 318/397
imports: [json, concurrent.futures.ThreadPoolExecutor, concurrent.futures.as_completed, dataclasses.asdict, dataclasses.dataclass... +13]
constants: [SUPPORTED_EXTENSIONS, conditional:dotenv.load_dotenv]
```

> Project-level Code Reproduction.

**class `FileResult`**

> Result for a single file reproduction.


**class `ProjectResult`**

> Result for project reproduction.


**class `ProjectReproducer`**

> Multi-file project reproduction system.

```yaml
methods:
  __init__(client:BaseLLMClient=None, max_workers:int=4, target_lang:str=None, use_llm:bool=True)  # creates
  find_source_files(project_path:str, extensions:Set[str]=None, exclude_patterns:List[str]=None) -> List[Path]  # retrieves source files
  reproduce_file(file_path:Path, output_dir:Path) -> FileResult  # reproduce file
  reproduce_project(project_path:str, output_dir:str=None, parallel:bool=False) -> ProjectResult  # reproduce project
```

**Functions:**

- `reproduce_project(project_path:str, output_dir:str=None, target_lang:str=None, parallel:bool=False, ...+1) -> ProjectResult` — reproduce project

---

#### `prompts.py`

```yaml
path: code2logic/prompts.py
lang: python | lines: 120/157
imports: [typing.Dict]
constants: [FORMAT_HINTS]
```

> Prompt templates for code reproduction.

**Functions:**

- `get_reproduction_prompt(spec:str, fmt:str, file_name:str, language:str='python', ...+1) -> str` — retrieves reproduction prompt
- `get_review_prompt(code:str, spec:str, fmt:str) -> str` — retrieves review prompt
- `get_fix_prompt(code:str, issues:list, spec:str) -> str` — retrieves fix prompt

---

#### `quality.py`

```yaml
path: code2logic/quality.py
lang: python | lines: 212/266
imports: [dataclasses.dataclass, dataclasses.field, typing.Any, typing.Dict, typing.List... +2]
```

> Code quality analysis module.

**class `QualityIssue`**

> Represents a code quality issue.


**class `QualityReport`**

> Complete quality analysis report.

```yaml
methods:
  to_dict() -> Dict[str, Any]  # converts dict
```

**class `QualityAnalyzer`**

> Analyzes code quality and generates recommendations.

```yaml
methods:
  __init__(thresholds:Dict[str,int]=None)  # creates
  analyze(project:ProjectInfo) -> QualityReport  # processes
  analyze_modules(modules:List[ModuleInfo]) -> QualityReport  # processes modules
```

**Functions:**

- `analyze_quality(project:ProjectInfo, thresholds:Dict[str,int]=None) -> QualityReport` — processes quality
- `get_quality_summary(report:QualityReport) -> str` — retrieves quality summary

---

#### `refactor.py`

```yaml
path: code2logic/refactor.py
lang: python | lines: 308/384
imports: [dataclasses.asdict, dataclasses.dataclass, dataclasses.field, typing.Any, typing.Dict... +6]
```

> Refactoring utilities for code2logic.

**class `DuplicateGroup`**

> Group of duplicate functions.


**class `RefactoringSuggestion`**

> Single refactoring suggestion.


**class `RefactoringReport`**

> Complete refactoring analysis report.

```yaml
methods:
  to_dict() -> Dict[str, Any]  # converts dict
  to_markdown() -> str  # converts markdown
```

**Functions:**

- `find_duplicates(project_path:str, threshold:float=0.8) -> List[DuplicateGroup]` — retrieves duplicates
- `analyze_quality(project_path:str, include_security:bool=True, include_performance:bool=True) -> RefactoringReport` — processes quality
- `suggest_refactoring(project_path:str, use_llm:bool=False, client:BaseLLMClient=None) -> RefactoringReport` — suggest refactoring
- `compare_codebases(project1:str, project2:str) -> Dict[str, Any]` — compare codebases
- `quick_analyze(project_path:str) -> Dict[str, Any]` — quick analyze

---

#### `reproducer.py`

```yaml
path: code2logic/reproducer.py
lang: python | lines: 537/711
imports: [json, re, dataclasses.dataclass, dataclasses.field, enum.Enum... +5]
```

> Code Reproducer - Generate code files from logic specifications.

**enum `ReproductionStatus`**

**class `ReproductionStatus`(Enum)**

> Status of file reproduction.


**class `FileValidation`**

> Validation result for a single file.

```yaml
methods:
  score() -> float  # score
  to_dict() -> Dict[str, Any]  # converts dict
```

**class `ReproductionResult`**

> Result of reproduction process.

```yaml
methods:
  success_rate() -> float  # success rate
  average_score() -> float  # average score
  summary() -> str  # summary
```

**class `SpecReproducer`**

> Reproduces code structure from logic specifications.

```yaml
methods:
  __init__(verbose:bool=False)  # creates
  reproduce_from_yaml(spec_path:str, output_dir:str, filter_paths:Optional[List[str]]=None) -> ReproductionResult  # reproduce from yaml
  reproduce_from_json(spec_path:str, output_dir:str, filter_paths:Optional[List[str]]=None) -> ReproductionResult  # reproduce from json
```

**class `SpecValidator`**

> Validates generated files against logic specification.

```yaml
methods:
  __init__()  # creates
  validate(spec_path:str, generated_dir:str, filter_paths:Optional[List[str]]=None) -> List[FileValidation]  # validates
```

**Functions:**

- `reproduce_project(spec_path:str, output_dir:str, filter_paths:Optional[List[str]]=None, validate:bool=True, ...+1) -> ReproductionResult` — reproduce project
- `validate_files(spec_path:str, generated_dir:str, filter_paths:Optional[List[str]]=None) -> List[FileValidation]` — validates files

---

#### `reproduction.py`

```yaml
path: code2logic/reproduction.py
lang: python | lines: 333/441
imports: [difflib, re, datetime, pathlib.Path, typing.Any... +4]
```

> Code reproduction utilities.

**class `CodeReproducer`**

> Code reproduction workflow using LLM.

```yaml
methods:
  __init__(client:BaseLLMClient=None, provider:str=None)  # creates
  reproduce_file(source_path:str, output_dir:str=None) -> Dict[str, Any]  # reproduce file
  generate_from_gherkin(gherkin:str, language:str='python') -> str  # creates from gherkin
```

**Functions:**

- `generate_file_gherkin(file_path:Path) -> str` — creates file gherkin
- `compare_code(original:str, generated:str) -> Dict[str, Any]` — compare code
- `extract_code_block(text:str, language:str='python') -> str` — parses code block

---

#### `shared_utils.py`

```yaml
path: code2logic/shared_utils.py
lang: python | lines: 279/414
imports: [hashlib, re, typing.Dict, typing.List, typing.Optional... +1]
constants: [TYPE_ABBREVIATIONS, CATEGORY_PATTERNS, DOMAIN_KEYWORDS]
```

> Shared utilities for Code2Logic generators.

**Functions:**

- `compact_imports(imports:List[str], max_items:int=10) -> List[str]` — compact imports
- `deduplicate_imports(imports:List[str]) -> List[str]` — deduplicate imports
- `abbreviate_type(type_str:str) -> str` — abbreviate type
- `expand_type(abbrev:str) -> str` — expand type
- `build_signature(params:List[str], return_type:Optional[str]=None, include_self:bool=False, abbreviate:bool=False, ...+1) -> str` — creates signature
- `remove_self_from_params(params:List[str]) -> List[str]` — deletes self from params
- `categorize_function(name:str) -> str` — categorize function
- `extract_domain(path:str) -> str` — parses domain
- `compute_hash(name:str, signature:str, length:int=8) -> str` — processes hash
- `truncate_docstring(docstring:Optional[str], max_length:int=60) -> str` — truncate docstring
- `escape_for_yaml(text:str) -> str` — escape for yaml
- `clean_identifier(name:str) -> str` — clean identifier

---

#### `similarity.py`

```yaml
path: code2logic/similarity.py
lang: python | lines: 201/265
imports: [logging, time, collections.defaultdict, typing.Dict, typing.List... +3]
constants: [RAPIDFUZZ_AVAILABLE, conditional:rapidfuzz.fuzz, conditional:rapidfuzz.process]
```

> Similarity detector using Rapidfuzz.

**class `SimilarityDetector`**

> Detects similar functions using fuzzy string matching.

```yaml
methods:
  __init__(threshold:float=80.0)  # creates
  find_similar_functions(modules:List[ModuleInfo]) -> Dict[str, List[str]]  # retrieves similar functions
  find_duplicate_signatures(modules:List[ModuleInfo]) -> Dict[str, List[str]]  # retrieves duplicate signatures
```

**Functions:**

- `is_rapidfuzz_available() -> bool` — is rapidfuzz available
- `get_refactoring_suggestions(similar_functions:Dict[str,List[str]]) -> List[Dict[str, any]]` — retrieves refactoring suggestions

---

#### `terminal.py`

```yaml
path: code2logic/terminal.py
lang: python | lines: 496/665
imports: [os, re, sys, typing.Any, typing.List... +2]
constants: [COLORS]
```

> Terminal Renderer - Colorized Markdown Output in Shell.

**class `ShellRenderer`**

> Renders colorized markdown output in terminal.

```yaml
methods:
  __init__(use_colors:bool=True, verbose:bool=True)  # creates
  enable_log() -> None  # enable log
  get_log() -> str  # retrieves log
  clear_log() -> None  # deletes log
  heading(level:int, text:str) -> None  # heading
  codeblock(language:Language, content:str) -> None  # codeblock
  render_markdown(text:str) -> None  # formats markdown
  success(message:str) -> None  # success
  error(message:str) -> None  # error
  warning(message:str) -> None  # warning
  info(message:str) -> None  # info
  status(icon:str, message:str, type:Literal[info,success,warning,error]='info') -> None  # status
  # ... +9 more
```

**class `RenderAPI`**

> Convenience API for terminal rendering.

```yaml
methods:
  static heading(level:int, text:str) -> None  # heading
  static code(lang:Language, content:str) -> None  # code
  static codeblock(lang:Language, content:str) -> None  # codeblock
  static markdown(text:str) -> None  # markdown
  static success(message:str) -> None  # success
  static error(message:str) -> None  # error
  static warning(message:str) -> None  # warning
  static info(message:str) -> None  # info
  static status(icon:str, message:str, type:Literal[info,success,warning,error]='info') -> None  # status
  static kv(key:str, value:Any) -> None  # kv
  static progress(done:int, total:int, label:str='') -> None  # progress
  static separator(char:str='─', width:int=60) -> None  # separator
  # ... +5 more
```

**Functions:**

- `get_renderer(use_colors:bool=True, verbose:bool=True) -> ShellRenderer` — retrieves renderer
- `set_renderer(renderer:ShellRenderer) -> None` — updates renderer

---

#### `toon_format.py`

```yaml
path: code2logic/toon_format.py
lang: python | lines: 663/859
imports: [re, typing.Any, typing.Dict, typing.List, models.ClassInfo... +6]
```

> TOON Format Generator for Code2Logic.

**class `TOONGenerator`**

> Generates TOON format output from ProjectInfo.

```yaml
methods:
  __init__(delimiter:str=',', use_tabs:bool=False)  # creates
  generate(project:ProjectInfo, detail:str='standard', no_repeat_name:bool=False) -> str  # creates
  generate_hybrid(project:ProjectInfo, detail:str='full', no_repeat_name:bool=True, hub_top_n:int=5, ...+1) -> str  # creates hybrid
  generate_compact(project:ProjectInfo) -> str  # creates compact
  generate_full(project:ProjectInfo) -> str  # creates full
  generate_schema(format_type:str='standard') -> str  # creates schema
  generate_ultra_compact(project:ProjectInfo) -> str  # creates ultra compact
```

**class `TOONParser`**

> Parse TOON format back to Python dict.

```yaml
methods:
  __init__()  # creates
  parse(content:str) -> Dict[str, Any]  # parses
```

**Functions:**

- `generate_toon(project:ProjectInfo, detail:str='standard', use_tabs:bool=False) -> str` — creates toon
- `parse_toon(content:str) -> Dict[str, Any]` — parses toon

---

#### `universal.py`

```yaml
path: code2logic/universal.py
lang: python | lines: 957/1234
imports: [hashlib, json, re, dataclasses.asdict, dataclasses.dataclass... +12]
constants: [conditional:dotenv.load_dotenv]
```

> Universal Code Logic Representation (UCLR).

**enum `ElementType`**

**enum `Language`**

**class `ElementType`(Enum)**

> Types of code elements.


**class `Language`(Enum)**

> Supported languages.


**class `Parameter`**

> Function/method parameter.


**class `CodeElement`**

> Universal representation of a code element.


**class `CodeLogic`**

> Universal code logic representation for a single file.

```yaml
methods:
  to_dict() -> Dict[str, Any]  # converts dict
  to_compact() -> str  # converts compact
```

**class `UniversalParser`**

> Parse source code into universal CodeLogic format.

```yaml
methods:
  detect_language(content:str, file_ext:str) -> Language  # detect language
  parse(file_path:Union[str,Path]) -> CodeLogic  # parses
```

**class `CodeGenerator`**

> Generate code from CodeLogic in target language.

```yaml
methods:
  generate(logic:CodeLogic, target_lang:Language) -> str  # creates
```

**class `UniversalReproducer`**

> Universal code reproduction system.

```yaml
methods:
  __init__(client:BaseLLMClient=None)  # creates
  extract_logic(file_path:str) -> CodeLogic  # parses logic
  reproduce(source_path:str, target_lang:str=None, output_dir:str=None, use_llm:bool=True) -> Dict[str, Any]  # reproduce
```

**Functions:**

- `reproduce_file(source_path:str, target_lang:str=None, output_dir:str=None, use_llm:bool=True) -> Dict[str, Any]` — reproduce file

---

#### `utils.py`

```yaml
path: code2logic/utils.py
lang: python | lines: 16/25
imports: [shutil, pathlib.Path]
```

**Functions:**

- `estimate_tokens(text:str) -> int` — estimate tokens
- `write_text_atomic(path:Path, content:str) -> None` — logs text atomic
- `cleanup_generated_root(generated_root:Path, allowed_dirs:set[str]) -> None` — cleanup generated root

---

### 📂 code2logic/benchmarks

#### `__init__.py`

```yaml
path: code2logic/benchmarks/__init__.py
lang: python | lines: 34/36
imports: [common.create_single_project, common.generate_spec, common.generate_spec_token, common.get_async_reproduction_prompt, common.get_simple_reproduction_prompt... +8]
```

---

#### `common.py`

```yaml
path: code2logic/benchmarks/common.py
lang: python | lines: 327/389
imports: [json, datetime, pathlib.Path, generators.CSVGenerator, generators.JSONGenerator... +7]
```

**Functions:**

- `create_single_project(module_info, file_path:Path) -> ProjectInfo` — creates single project
- `generate_spec(project:ProjectInfo, fmt:str) -> str` — creates spec
- `generate_spec_token(project:ProjectInfo, fmt:str) -> str` — creates spec token
- `get_async_reproduction_prompt(spec:str, fmt:str, file_name:str, with_tests:bool=False) -> str` — retrieves async reproduction prompt
- `get_token_reproduction_prompt(spec:str, fmt:str, file_name:str, language:str='python') -> str` — retrieves token reproduction prompt
- `get_simple_reproduction_prompt(spec:str, fmt:str, file_name:str) -> str` — retrieves simple reproduction prompt

---

#### `results.py`

```yaml
path: code2logic/benchmarks/results.py
lang: python | lines: 162/246
imports: [json, dataclasses.asdict, dataclasses.dataclass, dataclasses.field, datetime... +5]
```

> Standardized Benchmark Result Dataclasses.

**class `FormatResult`**

> Result for a single format test.

```yaml
methods:
  to_dict() -> Dict[str, Any]  # converts dict
```

**class `FileResult`**

> Result for single file reproduction.

```yaml
methods:
  to_dict() -> Dict[str, Any]  # converts dict
```

**class `FunctionResult`**

> Result for single function reproduction.

```yaml
methods:
  to_dict() -> Dict[str, Any]  # converts dict
```

**class `BenchmarkResult`**

> Complete benchmark result.

```yaml
methods:
  __post_init__()  # creates init
  calculate_aggregates()  # processes aggregates
  to_dict() -> Dict[str, Any]  # converts dict
  to_json(indent:int=2) -> str  # converts json
  save(path:str)  # caches
  load(path:str) -> 'BenchmarkResult'  # retrieves
```

**class `BenchmarkConfig`**

> Configuration for benchmark runs.

```yaml
methods:
  to_dict() -> Dict[str, Any]  # converts dict
```

---

#### `runner.py`

```yaml
path: code2logic/benchmarks/runner.py
lang: python | lines: 883/1096
imports: [difflib, re, sys, time, pathlib.Path... +15]
```

> Unified Benchmark Runner for Code2Logic.

**class `BenchmarkRunner`**

> Unified benchmark runner for code2logic.

```yaml
methods:
  __init__(client:Optional[BaseLLMClient]=None, config:Optional[BenchmarkConfig]=None)  # creates
  run_format_benchmark(folder:str, formats:List[str]=None, limit:Optional[int]=None, verbose:bool=False) -> BenchmarkResult  # starts format benchmark
  run_file_benchmark(file_path:str, formats:List[str]=None, verbose:bool=False) -> BenchmarkResult  # starts file benchmark
  run_function_benchmark(file_path:str, function_names:List[str]=None, limit:Optional[int]=None, verbose:bool=False) -> BenchmarkResult  # starts function benchmark
  run_project_benchmark(project_path:str, formats:List[str]=None, limit:Optional[int]=None, verbose:bool=False) -> BenchmarkResult  # starts project benchmark
```

**Functions:**

- `run_benchmark(source:str, benchmark_type:str='format', formats:List[str]=None, limit:Optional[int]=None, ...+2) -> BenchmarkResult` — starts benchmark

---

### 📂 code2logic/core

#### `__init__.py`

```yaml
path: code2logic/core/__init__.py
lang: python | lines: 21/24
imports: [analyzer.ProjectAnalyzer, analyzer.analyze_project, dependency.DependencyAnalyzer, errors.AnalysisError, errors.AnalysisResult... +10]
```

> Core analysis components.

---

### 📂 code2logic/formats

#### `__init__.py`

```yaml
path: code2logic/formats/__init__.py
lang: python | lines: 31/40
imports: [file_formats.generate_file_csv, file_formats.generate_file_json, file_formats.generate_file_yaml, generators.CompactGenerator, generators.CSVGenerator... +15]
```

> Output format generators.

---

### 📂 code2logic/integrations

#### `__init__.py`

```yaml
path: code2logic/integrations/__init__.py
lang: python | lines: 5/8
imports: [mcp_server.call_tool, mcp_server.handle_request, mcp_server.run_server]
```

> External integrations.

---

### 📂 code2logic/llm

#### `__init__.py`

```yaml
path: code2logic/llm/__init__.py
lang: python | lines: 40/53
imports: [importlib.util.module_from_spec, importlib.util.spec_from_file_location, pathlib.Path, intent.EnhancedIntentGenerator, llm_clients.DEFAULT_MODELS... +7]
```

> LLM client integrations.

---

### 📂 code2logic/schemas

#### `__init__.py`

```yaml
path: code2logic/schemas/__init__.py
lang: python | lines: 25/30
imports: [json_schema.JSONSchema, json_schema.parse_json_spec, json_schema.validate_json, logicml_schema.LogicMLSchema, logicml_schema.validate_logicml... +4]
```

> Format Schemas for Code2Logic.

---

#### `json_schema.py`

```yaml
path: code2logic/schemas/json_schema.py
lang: python | lines: 206/258
imports: [json, dataclasses.dataclass, dataclasses.field, typing.Any, typing.Dict... +3]
```

> JSON Format Schema for Code2Logic.

**class `JSONMethodSchema`**

> Schema for JSON method definition.


**class `JSONClassSchema`**

> Schema for JSON class definition.


**class `JSONFunctionSchema`**

> Schema for JSON function definition.


**class `JSONModuleSchema`**

> Schema for JSON module definition.


**class `JSONSchema`**

> Complete JSON specification schema.


**Functions:**

- `validate_json(spec:str) -> Tuple[bool, List[str]]` — validates json
- `parse_json_spec(spec:str) -> Optional[JSONSchema]` — parses json spec

---

#### `logicml_schema.py`

```yaml
path: code2logic/schemas/logicml_schema.py
lang: python | lines: 184/243
imports: [re, dataclasses.dataclass, dataclasses.field, typing.Any, typing.Dict... +3]
```

> LogicML Format Schema for Code2Logic.

**class `LogicMLMethod`**

> Schema for LogicML method.


**class `LogicMLClass`**

> Schema for LogicML class.


**class `LogicMLModule`**

> Schema for LogicML module.


**class `LogicMLSchema`**

> Complete LogicML specification schema.

    Design Principles:
    1.


**Functions:**

- `validate_logicml(spec:str) -> Tuple[bool, List[str]]` — validates logicml
- `parse_logicml_header(line:str) -> Optional[Dict[str, Any]]` — parses logicml header
- `extract_logicml_signature(sig_line:str) -> Dict[str, Any]` — parses logicml signature

---

#### `markdown_schema.py`

```yaml
path: code2logic/schemas/markdown_schema.py
lang: python | lines: 118/172
imports: [re, dataclasses.dataclass, dataclasses.field, typing.Any, typing.Dict... +2]
```

> Markdown Hybrid Format Schema for Code2Logic.

**class `MarkdownMethod`**

> Schema for Markdown method.


**class `MarkdownClass`**

> Schema for Markdown class.


**class `MarkdownModule`**

> Schema for Markdown module.


**class `MarkdownSchema`**

> Complete Markdown specification schema.


**Functions:**

- `validate_markdown(spec:str) -> Tuple[bool, List[str]]` — validates markdown
- `extract_markdown_sections(spec:str) -> Dict[str, Any]` — parses markdown sections

---

#### `yaml_schema.py`

```yaml
path: code2logic/schemas/yaml_schema.py
lang: python | lines: 167/219
imports: [dataclasses.dataclass, dataclasses.field, typing.Any, typing.Dict, typing.List... +1]
```

> YAML Format Schema for Code2Logic.

**class `MethodSchema`**

> Schema for method definition.


**class `ClassSchema`**

> Schema for class definition.


**class `FunctionSchema`**

> Schema for function definition.


**class `ModuleSchema`**

> Schema for module definition.


**class `YAMLSchema`**

> Complete YAML specification schema.


**Functions:**

- `validate_yaml(spec:str) -> Tuple[bool, List[str]]` — validates yaml

---

### 📂 code2logic/tools

#### `__init__.py`

```yaml
path: code2logic/tools/__init__.py
lang: python | lines: 28/31
imports: [adaptive.LLM_CAPABILITIES, adaptive.AdaptiveReproducer, adaptive.AdaptiveResult, adaptive.get_llm_capabilities, benchmark.BenchmarkResult... +14]
```

> Development tools and utilities.

---

### 📂 examples

#### `01_quick_start.py`

```yaml
path: examples/01_quick_start.py
lang: python | lines: 47/72
imports: [sys, pathlib.Path, code2logic.analyze_project, code2logic.quick_analyze, code2logic.GherkinGenerator... +1]
```

> Quick Start Example - Basic code2logic usage.

**Functions:**

- `main()` — main

---

#### `02_refactoring.py`

```yaml
path: examples/02_refactoring.py
lang: python | lines: 45/69
imports: [sys, pathlib.Path, code2logic.find_duplicates, code2logic.analyze_quality, code2logic.suggest_refactoring]
```

> Refactoring Example - Find duplicates and quality issues.

**Functions:**

- `main()` — main

---

#### `03_reproduction.py`

```yaml
path: examples/03_reproduction.py
lang: python | lines: 57/76
imports: [sys, argparse, pathlib.Path, dotenv.load_dotenv, code2logic.UniversalReproducer... +2]
constants: [conditional:dotenv.load_dotenv]
```

> Code Reproduction Example - Generate code from logic.

**Functions:**

- `main()` — main

---

#### `04_project.py`

```yaml
path: examples/04_project.py
lang: python | lines: 55/75
imports: [sys, argparse, pathlib.Path, dotenv.load_dotenv, code2logic.reproduce_project... +1]
constants: [conditional:dotenv.load_dotenv]
```

> Project Reproduction Example - Multi-file projects.

**Functions:**

- `main()` — main

---

#### `05_llm_integration.py`

```yaml
path: examples/05_llm_integration.py
lang: python | lines: 87/110
imports: [sys, argparse, os, pathlib.Path, dotenv.load_dotenv... +5]
constants: [conditional:dotenv.load_dotenv]
```

> LLM Integration Example - Use with OpenRouter or Ollama.

**Functions:**

- `main()` — main

---

#### `06_metrics.py`

```yaml
path: examples/06_metrics.py
lang: python | lines: 85/118
imports: [sys, argparse, pathlib.Path, dotenv.load_dotenv, code2logic.ReproductionMetrics... +3]
constants: [conditional:dotenv.load_dotenv]
```

> Metrics Analysis Example - Detailed reproduction quality analysis.

**Functions:**

- `analyze_file(source_path:str, verbose:bool=False, no_llm:bool=False)` — processes file
- `main()` — main

---

#### `06_metrics_simple.py`

```yaml
path: examples/06_metrics_simple.py
lang: python | lines: 0/1
```

---

#### `08_format_benchmark.py`

```yaml
path: examples/08_format_benchmark.py
lang: python | lines: 90/120
imports: [argparse, sys, pathlib.Path, dotenv.load_dotenv, code2logic.benchmarks.BenchmarkRunner... +1]
```

> Format Benchmark - Simplified.

**Functions:**

- `print_format_comparison(result)` — logs format comparison
- `print_per_file_results(result)` — logs per file results
- `main()` — main

---

#### `09_async_benchmark.py`

```yaml
path: examples/09_async_benchmark.py
lang: python | lines: 60/80
imports: [argparse, sys, pathlib.Path, dotenv.load_dotenv, code2logic.benchmarks.BenchmarkRunner... +1]
```

> Async Benchmark - Simplified.

**Functions:**

- `print_results(result)` — logs results
- `main()` — main

---

#### `10_function_reproduction.py`

```yaml
path: examples/10_function_reproduction.py
lang: python | lines: 51/72
imports: [argparse, sys, pathlib.Path, dotenv.load_dotenv, code2logic.benchmarks.BenchmarkRunner... +1]
```

> Function-Level Reproduction - Simplified.

**Functions:**

- `print_results(result)` — logs results
- `main()` — main

---

#### `11_token_benchmark.py`

```yaml
path: examples/11_token_benchmark.py
lang: python | lines: 69/93
imports: [argparse, sys, pathlib.Path, dotenv.load_dotenv, code2logic.benchmarks.BenchmarkRunner... +1]
```

> Token-Aware Benchmark - Simplified.

**Functions:**

- `print_token_efficiency(result)` — logs token efficiency
- `main()` — main

---

#### `12_comprehensive_analysis.py`

```yaml
path: examples/12_comprehensive_analysis.py
lang: python | lines: 95/129
imports: [argparse, sys, pathlib.Path, dotenv.load_dotenv, code2logic.benchmarks.BenchmarkRunner... +1]
constants: [ALL_FORMATS]
```

> Comprehensive Analysis - Simplified.

**Functions:**

- `print_comprehensive_analysis(result)` — logs comprehensive analysis
- `main()` — main

---

#### `12_comprehensive_analysis_simple.py`

```yaml
path: examples/12_comprehensive_analysis_simple.py
lang: python | lines: 0/1
```

---

#### `13_project_benchmark.py`

```yaml
path: examples/13_project_benchmark.py
lang: python | lines: 70/95
imports: [argparse, sys, pathlib.Path, dotenv.load_dotenv, code2logic.benchmarks.BenchmarkRunner... +1]
```

> Project Benchmark - Simplified.

**Functions:**

- `print_project_results(result)` — logs project results
- `main()` — main

---

#### `14_repeatability_test.py`

```yaml
path: examples/14_repeatability_test.py
lang: python | lines: 331/440
imports: [argparse, difflib, json, sys, time... +15]
```

> Repeatability Test for Code Generation.

**class `RunResult`**

> Result of a single generation run.


**class `RepeatabilityResult`**

> Repeatability analysis for a format.


**Functions:**

- `generate_spec(project:ProjectInfo, fmt:str) -> str` — creates spec
- `get_reproduction_prompt(spec:str, fmt:str, file_name:str) -> str` — retrieves reproduction prompt
- `calculate_similarity(code1:str, code2:str) -> float` — processes similarity
- `get_diff(code1:str, code2:str, label1:str='Run 1', label2:str='Run 2') -> List[str]` — retrieves diff
- `test_syntax(code:str) -> bool` — checks syntax
- `run_repeatability_test(file_path:str, formats:List[str], num_runs:int=3, verbose:bool=False, ...+1) -> Dict[str, RepeatabilityResult]` — starts repeatability test
- `print_repeatability_summary(results:Dict[str,RepeatabilityResult])` — logs repeatability summary
- `save_repeatability_report(results:Dict[str,RepeatabilityResult], output:str)` — caches repeatability report
- `main()` — main

---

#### `15_unified_benchmark.py`

```yaml
path: examples/15_unified_benchmark.py
lang: python | lines: 143/182
imports: [argparse, os, sys, pathlib.Path, dotenv.load_dotenv... +4]
```

> Unified Benchmark Example.

**Functions:**

- `print_format_results(result)` — logs format results
- `print_function_results(result)` — logs function results
- `print_project_results(result)` — logs project results
- `main()` — main

---

#### `16_terminal_demo.py`

```yaml
path: examples/16_terminal_demo.py
lang: python | lines: 178/244
imports: [argparse, sys, pathlib.Path, code2logic.terminal.render, code2logic.terminal.ShellRenderer... +2]
```

> Terminal Rendering Demo - Colorized Markdown Output in Shell.

**Functions:**

- `demo_headings()` — demo headings
- `demo_codeblocks()` — demo codeblocks
- `demo_status_messages()` — demo status messages
- `demo_progress()` — demo progress
- `demo_tasks()` — demo tasks
- `demo_key_value()` — demo key value
- `demo_tables()` — demo tables
- `demo_markdown()` — demo markdown
- `demo_log_highlighting()` — demo log highlighting
- `main()` — main

---

#### `behavioral_benchmark.py`

```yaml
path: examples/behavioral_benchmark.py
lang: python | lines: 220/271
imports: [importlib.util, json, os, tempfile, dataclasses.asdict... +8]
```

> Behavioral benchmark: compare runtime behavior of reproduced functions vs origin...

**class `CaseResult`**


**class `FunctionBehaviorResult`**


**Functions:**

- `main() -> None` — main

---

#### `benchmark_report.py`

```yaml
path: examples/benchmark_report.py
lang: python | lines: 170/210
imports: [json, os, shlex, dataclasses.dataclass, datetime... +6]
```

> Generate a Markdown report linking benchmark artifacts and showing commands used...

**class `Artifact`**


**Functions:**

- `main() -> None` — main

---

#### `benchmark_summary.py`

```yaml
path: examples/benchmark_summary.py
lang: python | lines: 100/121
imports: [json, os, sys]
```

> Print benchmark summary from JSON result files.

**Functions:**

- `main()` — main

---

#### `duplicate_detection.py`

```yaml
path: examples/duplicate_detection.py
lang: python | lines: 0/1
```

---

#### `token_efficiency.py`

```yaml
path: examples/token_efficiency.py
lang: python | lines: 0/1
```

---

#### `windsurf_mcp_integration.py`

```yaml
path: examples/windsurf_mcp_integration.py
lang: python | lines: 0/1
```

---

### 📂 examples/code2logic/sample_project

#### `__init__.py`

```yaml
path: examples/code2logic/sample_project/__init__.py
lang: python | lines: 1/2
```

> Sample project used by examples/code2logic.

---

#### `calculator.py`

```yaml
path: examples/code2logic/sample_project/calculator.py
lang: python | lines: 22/30
```

**class `Calculator`**

> A tiny calculator used for Code2Logic examples.

```yaml
methods:
  add(a:float, b:float) -> float  # creates
  divide(a:float, b:float) -> float  # splits
```

**Functions:**

- `factorial(n:int) -> int` — factorial

---

### 📂 examples/code2logic/sample_project/api

#### `client.py`

```yaml
path: examples/code2logic/sample_project/api/client.py
lang: python | lines: 15/24
imports: [dataclasses.dataclass]
```

**class `Response`**

> Very small HTTP-like response placeholder.


**class `APIClient`**

> Example client with async methods.

```yaml
methods:
  async get(url:str) -> Response  # retrieves
  async post(url:str, data:dict) -> Response  # creates
```

---

### 📂 examples/code2logic/sample_project/models

#### `user.py`

```yaml
path: examples/code2logic/sample_project/models/user.py
lang: python | lines: 9/14
imports: [dataclasses.dataclass]
```

**class `User`**

> Simple user model.


---

### 📂 logic2code

#### `__init__.py`

```yaml
path: logic2code/__init__.py
lang: python | lines: 13/19
imports: [generator.CodeGenerator, generator.GeneratorConfig, generator.GenerationResult, renderers.PythonRenderer]
```

> Logic2Code - Generate source code from Code2Logic output files.

---

#### `__main__.py`

```yaml
path: logic2code/__main__.py
lang: python | lines: 8/12
imports: [cli.main]
```

> Entry point for running logic2code as a module.

---

#### `cli.py`

```yaml
path: logic2code/cli.py
lang: python | lines: 152/204
imports: [argparse, sys, pathlib.Path, generator.CodeGenerator, generator.GeneratorConfig... +1]
```

> CLI interface for logic2code.

**Functions:**

- `main()` — main

---

#### `generator.py`

```yaml
path: logic2code/generator.py
lang: python | lines: 233/303
imports: [re, dataclasses.dataclass, dataclasses.field, pathlib.Path, typing.Dict... +9]
```

> Main code generator that orchestrates parsing and code generation.

**class `GeneratorConfig`**

> Configuration for code generation.


**class `GenerationResult`**

> Result of code generation.


**class `CodeGenerator`**

> Main code generator class.

```yaml
methods:
  __init__(logic_file:Union[str,Path], config:Optional[GeneratorConfig]=None)  # creates
  project() -> ProjectSpec  # project
  generate(output_dir:Union[str,Path], modules:Optional[List[str]]=None) -> GenerationResult  # creates
  generate_module(module_path:str) -> str  # creates module
  generate_class(class_name:str, module_path:Optional[str]=None) -> str  # creates class
  generate_function(func_name:str, module_path:Optional[str]=None) -> str  # creates function
  summary() -> Dict  # summary
  list_modules() -> List[str]  # list modules
  list_classes() -> List[str]  # list classes
  list_functions() -> List[str]  # list functions
```

---

#### `renderers.py`

```yaml
path: logic2code/renderers.py
lang: python | lines: 297/417
imports: [re, abc.ABC, abc.abstractmethod, dataclasses.dataclass, dataclasses.field... +10]
```

> Code renderers for different programming languages.

**class `RenderConfig`**

> Configuration for code rendering.


**abstract class `BaseRenderer`(ABC)**

> Abstract base class for language-specific renderers.

```yaml
methods:
  __init__(config:Optional[RenderConfig]=None)  # creates
  render_module(module:ModuleSpec) -> str  # formats module
  render_class(cls:ClassSpec) -> str  # formats class
  render_function(func:FunctionSpec) -> str  # formats function
```

**class `PythonRenderer`(BaseRenderer)**

> Python code renderer.

```yaml
methods:
  render_module(module:ModuleSpec) -> str  # formats module
  render_class(cls:ClassSpec) -> str  # formats class
  render_function(func:FunctionSpec) -> str  # formats function
  render_init_file(modules:List[ModuleSpec]) -> str  # formats init file
```

---

### 📂 logic2code/examples

#### `01_quickstart.py`

```yaml
path: logic2code/examples/01_quickstart.py
lang: python | lines: 72/110
imports: [pathlib.Path, sys, logic2code.CodeGenerator, logic2code.GeneratorConfig]
```

> Logic2Code Quickstart Example

Generate source code from Code2Logic output files...

**Functions:**

- `example_basic_generation()` — example basic generation
- `example_with_config()` — example with config
- `example_stubs_only()` — example stubs only
- `example_single_module()` — example single module

---

#### `02_llm_enhanced.py`

```yaml
path: logic2code/examples/02_llm_enhanced.py
lang: python | lines: 77/114
imports: [pathlib.Path, sys, logic2code.CodeGenerator, logic2code.GeneratorConfig]
```

> Logic2Code LLM-Enhanced Generation Example

Using LLM to generate implementation...

**Functions:**

- `example_llm_generation()` — example llm generation
- `example_hybrid_generation()` — example hybrid generation
- `example_compare_outputs()` — example compare outputs

---

### 📂 logic2code/tests

#### `__init__.py`

```yaml
path: logic2code/tests/__init__.py
lang: python | lines: 1/2
```

> Tests for logic2code package.

---

#### `test_basic.py`

```yaml
path: logic2code/tests/test_basic.py
lang: python | lines: 40/55
imports: [pytest]
```

> Basic tests for logic2code package.

**Functions:**

- `test_import_logic2code()` — checks import logic2code
- `test_import_generator()` — checks import generator
- `test_config_defaults()` — checks config defaults
- `test_generator_config_custom()` — checks generator config custom
- `test_generator_config_llm()` — checks generator config llm

---

### 📂 logic2test

#### `__init__.py`

```yaml
path: logic2test/__init__.py
lang: python | lines: 14/20
imports: [generator.TestGenerator, generator.GeneratorConfig, generator.GenerationResult, parsers.LogicParser, templates.TestTemplate]
```

> Logic2Test - Generate tests from Code2Logic output files.

---

#### `__main__.py`

```yaml
path: logic2test/__main__.py
lang: python | lines: 8/12
imports: [cli.main]
```

> Entry point for running logic2test as a module.

---

#### `cli.py`

```yaml
path: logic2test/cli.py
lang: python | lines: 131/180
imports: [argparse, sys, pathlib.Path, generator.TestGenerator, generator.GeneratorConfig... +1]
```

> CLI interface for logic2test.

**Functions:**

- `main()` — main

---

#### `generator.py`

```yaml
path: logic2test/generator.py
lang: python | lines: 328/431
imports: [re, dataclasses.dataclass, dataclasses.field, pathlib.Path, typing.Dict... +9]
```

> Main test generator that orchestrates parsing and test generation.

**class `GeneratorConfig`**

> Configuration for test generation.


**class `GenerationResult`**

> Result of test generation.


**class `TestGenerator`**

> Main test generator class.

```yaml
methods:
  __init__(logic_file:Union[str,Path], config:Optional[GeneratorConfig]=None)  # creates
  project() -> ProjectSpec  # project
  generate_unit_tests(output_dir:Union[str,Path], modules:Optional[List[str]]=None) -> GenerationResult  # creates unit tests
  generate_integration_tests(output_dir:Union[str,Path], entry_points:Optional[List[str]]=None) -> GenerationResult  # creates integration tests
  generate_property_tests(output_dir:Union[str,Path]) -> GenerationResult  # creates property tests
  summary() -> Dict  # summary
```

---

#### `parsers.py`

```yaml
path: logic2test/parsers.py
lang: python | lines: 272/333
imports: [re, dataclasses.dataclass, dataclasses.field, pathlib.Path, typing.Any... +4]
```

> Parsers for Code2Logic output formats (YAML, Hybrid, TOON).

**class `FunctionSpec`**

> Specification of a function/method extracted from logic file.


**class `ClassSpec`**

> Specification of a class extracted from logic file.


**class `ModuleSpec`**

> Specification of a module extracted from logic file.


**class `ProjectSpec`**

> Full project specification from logic file.


**class `LogicParser`**

> Parser for Code2Logic output formats.

```yaml
methods:
  __init__(file_path:Union[str,Path])  # creates
  parse() -> ProjectSpec  # parses
```

---

#### `templates.py`

```yaml
path: logic2test/templates.py
lang: python | lines: 198/252
imports: [dataclasses.dataclass, typing.List, typing.Optional]
```

> Test templates for different test types and frameworks.

**class `TestTemplate`**

> Template for generating test code.

```yaml
methods:
  render_test_file_header(module_path:str, imports:List[str]=None) -> str  # formats test file header
  render_function_test(func_name:str, params:List[str], return_type:Optional[str]=None, docstring:Optional[str]=None, ...+2) -> str  # formats function test
  render_class_test(class_name:str, bases:List[str], is_dataclass:bool=False, fields:List[dict]=None, ...+1) -> str  # formats class test
  render_dataclass_test(class_name:str, fields:List[dict]) -> str  # formats dataclass test
```

---

### 📂 logic2test/examples

#### `01_quickstart.py`

```yaml
path: logic2test/examples/01_quickstart.py
lang: python | lines: 62/96
imports: [pathlib.Path, sys, logic2test.TestGenerator, logic2test.GeneratorConfig]
```

> Logic2Test Quickstart Example

Generate test scaffolds from Code2Logic output fi...

**Functions:**

- `example_basic_generation()` — example basic generation
- `example_with_config()` — example with config
- `example_generate_all_types()` — example generate all types

---

#### `02_custom_templates.py`

```yaml
path: logic2test/examples/02_custom_templates.py
lang: python | lines: 92/130
imports: [pathlib.Path, sys, logic2test.templates.TestTemplate, logic2test.parsers.FunctionSpec, logic2test.parsers.ClassSpec]
```

> Logic2Test Custom Templates Example

Customizing test generation templates.

**Functions:**

- `example_function_test()` — example function test
- `example_class_test()` — example class test
- `example_dataclass_test()` — example dataclass test
- `example_async_function_test()` — example async function test

---

### 📂 logic2test/tests

#### `__init__.py`

```yaml
path: logic2test/tests/__init__.py
lang: python | lines: 1/2
```

> Tests for logic2test package.

---

#### `test_basic.py`

```yaml
path: logic2test/tests/test_basic.py
lang: python | lines: 31/43
imports: [pytest]
```

> Basic tests for logic2test package.

**Functions:**

- `test_import_logic2test()` — checks import logic2test
- `test_import_generator()` — checks import generator
- `test_config_defaults()` — checks config defaults
- `test_generator_config_custom()` — checks generator config custom

---

### 📂 lolm

#### `__init__.py`

```yaml
path: lolm/__init__.py
lang: python | lines: 103/119
imports: [config.LLMConfig, config.load_config, config.save_config, config.get_config_path, config.get_provider_model... +15]
```

> LOLM - Lightweight Orchestrated LLM Manager

A reusable LLM configuration and ma...

---

#### `__main__.py`

```yaml
path: lolm/__main__.py
lang: python | lines: 10/14
imports: [cli.main]
```

> Entry point for running lolm as a module.

---

#### `cli.py`

```yaml
path: lolm/cli.py
lang: python | lines: 231/317
imports: [argparse, os, sys, pathlib.Path, config.DEFAULT_MODELS... +14]
```

> CLI interface for LOLM - LLM provider management.

**Functions:**

- `cmd_status(args) -> int` — cmd status
- `cmd_set_provider(args) -> int` — cmd set provider
- `cmd_set_model(args) -> int` — cmd set model
- `cmd_key_set(args) -> int` — cmd key set
- `cmd_key_show(args) -> int` — cmd key show
- `cmd_models(args) -> int` — cmd models
- `cmd_test(args) -> int` — cmd test
- `cmd_config_show(args) -> int` — cmd config show
- `cmd_priority_set_provider(args) -> int` — cmd priority set provider
- `cmd_priority_set_mode(args) -> int` — cmd priority set mode
- `main()` — main

---

#### `clients.py`

```yaml
path: lolm/clients.py
lang: python | lines: 274/342
imports: [os, time, typing.List, typing.Optional, config.DEFAULT_MODELS... +7]
constants: [conditional:httpx, conditional:litellm]
```

> LLM Client Implementations.

**class `LLMRateLimitError`(Exception)**

> Exception raised when a rate limit is hit.

```yaml
methods:
  __init__(message:str, provider:str='', status_code:int=429, headers:dict=None, ...+1)  # creates
  __str__()  # str
```

**class `OpenRouterClient`(BaseLLMClient)**

> OpenRouter API client for cloud LLM access.

```yaml
methods:
  __init__(api_key:str=None, model:str=None)  # creates
  generate(prompt:str, system:str=None, max_tokens:int=4000) -> str  # creates
  is_available() -> bool  # is available
  static list_recommended_models() -> List[tuple]  # list recommended models
```

**class `OllamaClient`(BaseLLMClient)**

> Ollama client for local LLM inference.

```yaml
methods:
  __init__(model:str=None, host:str=None)  # creates
  generate(prompt:str, system:str=None, max_tokens:int=4000) -> str  # creates
  is_available() -> bool  # is available
  list_models() -> List[str]  # list models
  static list_recommended_models() -> List[tuple]  # list recommended models
```

**class `LiteLLMClient`(BaseLLMClient)**

> LiteLLM client for universal LLM access.

```yaml
methods:
  __init__(model:str=None)  # creates
  generate(prompt:str, system:str=None, max_tokens:int=4000) -> str  # creates
  is_available() -> bool  # is available
```

**class `GroqClient`(BaseLLMClient)**

> Groq API client for fast inference.

```yaml
methods:
  __init__(api_key:str=None, model:str=None)  # creates
  generate(prompt:str, system:str=None, max_tokens:int=4000) -> str  # creates
  is_available() -> bool  # is available
```

**class `TogetherClient`(BaseLLMClient)**

> Together AI client.

```yaml
methods:
  __init__(api_key:str=None, model:str=None)  # creates
  generate(prompt:str, system:str=None, max_tokens:int=4000) -> str  # creates
  is_available() -> bool  # is available
```

---

#### `config.py`

```yaml
path: lolm/config.py
lang: python | lines: 271/335
imports: [json, os, dataclasses.dataclass, dataclasses.field, pathlib.Path... +6]
constants: [RECOMMENDED_MODELS, DEFAULT_MODELS, DEFAULT_PROVIDER_PRIORITIES, conditional:getv.EnvStore, conditional:yaml]
```

> LLM Configuration Management.

**class `LLMConfig`**

> LLM configuration container.

```yaml
methods:
  to_dict() -> Dict[str, Any]  # converts dict
  from_dict(data:Dict[str,Any]) -> 'LLMConfig'  # from dict
```

**Functions:**

- `get_config_dir() -> Path` — retrieves config dir
- `get_config_path() -> Path` — retrieves config path
- `load_config() -> LLMConfig` — retrieves config
- `save_config(config:LLMConfig) -> None` — caches config
- `load_env_file(search_paths:Optional[List[Path]]=None) -> None` — retrieves env file
- `load_litellm_config(search_paths:Optional[List[Path]]=None) -> Dict[str, Any]` — retrieves litellm config
- `save_litellm_config(config:Dict[str,Any], path:Optional[Path]=None) -> None` — caches litellm config
- `get_provider_model(provider:str) -> str` — retrieves provider model
- `set_provider_model(provider:str, model:str) -> None` — updates provider model
- `get_api_key(provider:str) -> Optional[str]` — retrieves api key
- `set_api_key(provider:str, key:str, env_path:Optional[Path]=None) -> None` — updates api key
- `get_provider_priorities_from_litellm() -> Dict[str, int]` — retrieves provider priorities from litellm

---

#### `manager.py`

```yaml
path: lolm/manager.py
lang: python | lines: 412/516
imports: [os, typing.Dict, typing.List, typing.Optional, config.DEFAULT_MODELS... +15]
```

> LLM Manager with Multi-Provider Support.

**class `ProviderInfo`**

> Information about a configured provider.

```yaml
methods:
  __init__(name:str, status:LLMProviderStatus, client:Optional[BaseLLMClient]=None, model:str='', ...+1)  # creates
```

**class `LLMManager`**

> LLM Manager with multi-provider support.

```yaml
methods:
  __init__(verbose:bool=False, enable_rotation:bool=True)  # creates
  is_available() -> bool  # is available
  is_ready() -> bool  # is ready
  primary_provider() -> Optional[BaseLLMClient]  # primary provider
  providers() -> Dict[str, ProviderInfo]  # providers
  initialize() -> None  # initializes
  get_client(provider:str=None) -> Optional[BaseLLMClient]  # retrieves client
  generate(prompt:str, system:str=None, max_tokens:int=4000, provider:str=None) -> str  # creates
  generate_with_fallback(prompt:str, system:str=None, max_tokens:int=4000, providers:Optional[List[str]]=None) -> str  # creates with fallback
  generate_with_rotation(prompt:str, system:str=None, max_tokens:int=4000, max_retries:int=3) -> str  # creates with rotation
  get_rotation_queue() -> Optional[RotationQueue]  # retrieves rotation queue
  get_provider_health(name:str=None) -> Dict  # retrieves provider health
  # ... +3 more
```

**Functions:**

- `get_client(provider:str=None, model:str=None) -> BaseLLMClient` — retrieves client
- `list_available_providers() -> List[str]` — list available providers

---

#### `provider.py`

```yaml
path: lolm/provider.py
lang: python | lines: 121/152
imports: [abc.ABC, abc.abstractmethod, dataclasses.dataclass, dataclasses.field, enum.Enum... +5]
```

> LLM Provider Base Classes and Types.

**enum `LLMProviderStatus`**

**class `LLMProviderStatus`(str, Enum)**

> Provider availability status.


**class `GenerateOptions`**

> Options for LLM generation.

```yaml
methods:
  to_messages() -> List[Dict[str, str]]  # converts messages
```

**class `LLMResponse`**

> Response from LLM generation.


**class `LLMModelInfo`**

> Information about an available model.


**abstract class `BaseLLMClient`(ABC)**

> Abstract base class for synchronous LLM clients.

```yaml
methods:
  generate(prompt:str, system:str=None, max_tokens:int=4000) -> str  # creates
  is_available() -> bool  # is available
  chat(messages:List[Dict[str,str]], max_tokens:int=4000) -> str  # chat
```

**abstract class `LLMProvider`(ABC)**

> Abstract base class for async LLM providers.

```yaml
methods:
  name() -> str  # name
  model() -> str  # model
  async is_available() -> bool  # is available
  async list_models() -> List[LLMModelInfo]  # list models
  async generate(options:GenerateOptions) -> LLMResponse  # creates
  async has_model(model_name:str) -> bool  # has model
  get_code_models(models:List[LLMModelInfo]) -> List[LLMModelInfo]  # retrieves code models
  async close() -> None  # stops
  async __aenter__()  # aenter
  async __aexit__()  # aexit
```

---

#### `rotation.py`

```yaml
path: lolm/rotation.py
lang: python | lines: 560/718
imports: [time, threading, dataclasses.dataclass, dataclasses.field, datetime... +10]
```

> LLM Provider Rotation with Rate Limit Detection and Dynamic Prioritization.

**enum `ProviderState`**

**enum `RateLimitType`**

**class `ProviderState`(str, Enum)**

> Provider availability state.


**class `RateLimitType`(str, Enum)**

> Type of rate limit encountered.


**class `RateLimitInfo`**

> Information about a rate limit event.

```yaml
methods:
  get_wait_seconds() -> float  # retrieves wait seconds
```

**class `ProviderHealth`**

> Health metrics for a provider.

```yaml
methods:
  success_rate() -> float  # success rate
  is_available() -> bool  # is available
  record_success(latency_ms:float=0) -> None  # record success
  record_failure(error:str, is_rate_limit:bool=False, rate_limit_info:Optional[RateLimitInfo]=None) -> None  # record failure
  check_cooldown() -> bool  # checks cooldown
  to_dict() -> Dict[str, Any]  # converts dict
```

**class `RotationQueue`**

> Priority queue for LLM provider rotation with automatic failover.

```yaml
methods:
  __init__(max_consecutive_failures:int=3, default_cooldown_seconds:float=60, enable_health_recovery:bool=True)  # creates
  add_provider(name:str, priority:int=100) -> None  # creates provider
  remove_provider(name:str) -> bool  # deletes provider
  set_priority(name:str, priority:int) -> bool  # updates priority
  get_priority_order() -> List[str]  # retrieves priority order
  get_next() -> Optional[str]  # retrieves next
  get_available() -> List[str]  # retrieves available
  record_success(name:str, latency_ms:float=0) -> None  # record success
  record_failure(name:str, error:str, is_rate_limit:bool=False, rate_limit_info:Optional[RateLimitInfo]=None) -> None  # record failure
  mark_rate_limited(name:str, rate_limit_info:Optional[RateLimitInfo]=None, cooldown_seconds:Optional[float]=None) -> None  # mark rate limited
  reset_provider(name:str) -> bool  # reset provider
  reset_all() -> None  # reset all
  # ... +6 more
```

**class `LLMRotationManager`**

> High-level manager for LLM rotation with generation capabilities.

```yaml
methods:
  __init__(max_retries:int=3, default_cooldown:float=60.0, verbose:bool=False)  # creates
  register(name:str, client:Any, priority:int=100) -> None  # registers
  unregister(name:str) -> bool  # unregister
  set_priority(name:str, priority:int) -> bool  # updates priority
  generate(prompt:str, system:str=None, max_tokens:int=4000, preferred_provider:str=None) -> str  # creates
  get_queue() -> RotationQueue  # retrieves queue
  get_status() -> Dict[str, Any]  # retrieves status
  reset() -> None  # reset
```

**Functions:**

- `parse_rate_limit_headers(headers:Dict[str,str]) -> Optional[RateLimitInfo]` — parses rate limit headers
- `is_rate_limit_error(status_code:int=None, error_message:str=None) -> bool` — is rate limit error
- `create_rotation_manager(providers:Dict[str,Tuple[Any,int]]=None, verbose:bool=False) -> LLMRotationManager` — creates rotation manager

---

### 📂 lolm/examples

#### `01_quickstart.py`

```yaml
path: lolm/examples/01_quickstart.py
lang: python | lines: 60/92
imports: [lolm.get_client, lolm.LLMManager]
```

> LOLM Quickstart Example

Basic usage of lolm for LLM interactions.

**Functions:**

- `example_simple_client()` — example simple client
- `example_specific_provider()` — example specific provider
- `example_manager()` — example manager
- `example_fallback()` — example fallback

---

#### `02_configuration.py`

```yaml
path: lolm/examples/02_configuration.py
lang: python | lines: 88/120
imports: [os, pathlib.Path, lolm.LLMConfig, lolm.load_config, lolm.save_config... +5]
```

> LOLM Configuration Example

Shows how to configure providers and manage settings...

**Functions:**

- `show_defaults()` — show defaults
- `show_recommended_models()` — show recommended models
- `show_current_config()` — show current config
- `example_modify_config()` — example modify config
- `show_environment_config()` — show environment config

---

#### `03_code_generation.py`

```yaml
path: lolm/examples/03_code_generation.py
lang: python | lines: 86/124
imports: [lolm.get_client, lolm.LLMManager]
constants: [SYSTEM_PROMPT]
```

> LOLM Code Generation Example

Using lolm for code generation tasks.

**Functions:**

- `generate_function(description:str) -> str` — creates function
- `generate_class(description:str) -> str` — creates class
- `explain_code(code:str) -> str` — explain code
- `review_code(code:str) -> str` — review code

---

### 📂 lolm/tests

#### `__init__.py`

```yaml
path: lolm/tests/__init__.py
lang: python | lines: 1/2
```

> Tests for lolm package.

---

#### `test_basic.py`

```yaml
path: lolm/tests/test_basic.py
lang: python | lines: 51/68
imports: [pytest]
```

> Basic tests for lolm package.

**Functions:**

- `test_import_lolm()` — checks import lolm
- `test_import_config()` — checks import config
- `test_import_clients()` — checks import clients
- `test_config_defaults()` — checks config defaults
- `test_manager_init()` — checks manager init
- `test_recommended_models()` — checks recommended models

---

### 📂 raport

#### `mermaid-init.js`

```yaml
path: raport/mermaid-init.js
lang: javascript | lines: 29/38
imports: [https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs]
```

**Functions:**

- `convertMermaidCodeBlocks()` — converts mermaid code blocks
- `async renderMermaid()` — formats mermaid

---

### 📂 scripts

#### `configure_llm.py`

```yaml
path: scripts/configure_llm.py
lang: python | lines: 360/467
imports: [sys, os, json, time, pathlib.Path... +7]
constants: [CONFIG_DIR, CONFIG_FILE, conditional:httpx, conditional:litellm]
```

> LLM Configuration Script for Code2Logic.

**Functions:**

- `log(msg:str, level:str='info')` — logs
- `check_ollama() -> Dict[str, Any]` — checks ollama
- `check_litellm() -> Dict[str, Any]` — checks litellm
- `check_env_keys() -> Dict[str, bool]` — checks env keys
- `categorize_models(models:List[Dict]) -> Dict[str, List[Dict]]` — categorize models
- `get_recommended_models(models:List[Dict]) -> Dict[str, str]` — retrieves recommended models
- `test_model(model:str, timeout:int=30) -> Dict[str, Any]` — checks model
- `save_config(config:Dict[str,Any])` — caches config
- `load_config() -> Dict[str, Any]` — retrieves config
- `format_size(size_bytes:int) -> str` — formats size
- `main()` — main

---

### 📂 tests

#### `__init__.py`

```yaml
path: tests/__init__.py
lang: python | lines: 3/4
```

> Tests for code2logic package.

---

#### `conftest.py`

```yaml
path: tests/conftest.py
lang: python | lines: 314/368
imports: [pytest, tempfile, pathlib.Path, typing.Dict, typing.Any... +4]
```

> Pytest configuration and fixtures for code2logic tests.

**Functions:**

- `sample_python_code() -> str` — sample python code
- `sample_javascript_code() -> str` — sample javascript code
- `sample_java_code() -> str` — sample java code
- `temp_project_dir()` — temp project dir
- `sample_project(temp_project_dir, sample_python_code)` — sample project
- `sample_module()` — sample module
- `sample_project_model()` — sample project model
- `mock_llm_config()` — mock llm config
- `sample_analysis_result()` — sample analysis result

---

#### `test_analyzer.py`

```yaml
path: tests/test_analyzer.py
lang: python | lines: 180/247
imports: [pytest, pathlib.Path, unittest.mock.Mock, unittest.mock.patch, code2logic.analyzer.ProjectAnalyzer... +4]
```

> Tests for the ProjectAnalyzer class.

**class `TestProjectAnalyzer`**

> Test cases for ProjectAnalyzer.

```yaml
methods:
  test_init(temp_project_dir)  # checks init
  test_init_with_verbose(temp_project_dir, capsys)  # checks init with verbose
  test_analyze_returns_project_info(sample_project)  # checks analyze returns project info
  test_analyze_finds_source_files(sample_project)  # checks analyze finds source files
  test_ignores_non_source_files(temp_project_dir)  # checks ignores non source files
  test_ignores_common_dirs(temp_project_dir)  # checks ignores common dirs
  test_analyze_extracts_functions(temp_project_dir)  # checks analyze extracts functions
  test_analyze_extracts_classes(temp_project_dir)  # checks analyze extracts classes
  test_analyze_extracts_imports(temp_project_dir)  # checks analyze extracts imports
  test_analyze_counts_lines(temp_project_dir)  # checks analyze counts lines
  test_analyze_empty_project(temp_project_dir)  # checks analyze empty project
  test_detect_entrypoints(temp_project_dir)  # checks detect entrypoints
  # ... +2 more
```

**class `TestAnalyzeProjectFunction`**

> Test the analyze_project convenience function.

```yaml
methods:
  test_analyze_project(sample_project)  # checks analyze project
```

**class `TestGetLibraryStatus`**

> Test the get_library_status function.

```yaml
methods:
  test_get_library_status()  # checks get library status
```

---

#### `test_e2e_projects.py`

```yaml
path: tests/test_e2e_projects.py
lang: python | lines: 82/106
imports: [shutil, pathlib.Path, code2logic.analyze_project, code2logic.generators.YAMLGenerator, logic2code.generator.Logic2CodeGenerator... +3]
```

**Functions:**

- `test_e2e_pipeline_code2logic_logic2test_logic2code(tmp_path:Path) -> None` — checks e2e pipeline code2logic logic2test logic2co
- `test_e2e_logic2test_on_examples_input(tmp_path:Path) -> None` — checks e2e logic2test on examples input
- `test_e2e_logic2code_on_examples_input(tmp_path:Path) -> None` — checks e2e logic2code on examples input

---

#### `test_error_handling.py`

```yaml
path: tests/test_error_handling.py
lang: python | lines: 311/490
imports: [pytest, tempfile, os, stat, pathlib.Path... +11]
```

> Comprehensive tests for error handling during project analysis.

**class `TestFilesystemErrors`**

> Tests for filesystem-related errors.

```yaml
methods:
  test_file_not_found(error_handler)  # checks file not found
  test_permission_denied(error_handler, tmp_path)  # checks permission denied
  test_file_too_large(tmp_path)  # checks file too large
  test_binary_file_detection(error_handler, tmp_path)  # checks binary file detection
  test_encoding_fallback(error_handler, tmp_path)  # checks encoding fallback
  test_empty_file(error_handler, tmp_path)  # checks empty file
```

**class `TestParsingErrors`**

> Tests for parsing-related errors.

```yaml
methods:
  test_syntax_error_in_python(error_handler, tmp_path)  # checks syntax error in python
  test_deeply_nested_code(error_handler)  # checks deeply nested code
  test_unsupported_language(error_handler)  # checks unsupported language
```

**class `TestGenerationErrors`**

> Tests for output generation errors.

```yaml
methods:
  test_yaml_with_special_characters(temp_project)  # checks yaml with special characters
  test_json_with_unicode(temp_project)  # checks json with unicode
  test_write_to_readonly_location(error_handler, tmp_path)  # checks write to readonly location
  test_write_creates_directories(error_handler, tmp_path)  # checks write creates directories
```

**class `TestErrorHandlerModes`**

> Tests for different error handler modes.

```yaml
methods:
  test_lenient_mode_continues(tmp_path)  # checks lenient mode continues
  test_strict_mode_stops(tmp_path)  # checks strict mode stops
  test_silent_mode_no_logging(tmp_path, caplog)  # checks silent mode no logging
  test_critical_error_stops_all_modes()  # checks critical error stops all modes
```

**class `TestIntegration`**

> Integration tests for error handling during full analysis.

```yaml
methods:
  test_mixed_valid_invalid_files(tmp_path)  # checks mixed valid invalid files
  test_nested_folders_with_errors(tmp_path)  # checks nested folders with errors
  test_large_project_resilience(tmp_path)  # checks large project resilience
```

**class `TestAnalysisResult`**

> Tests for AnalysisResult class.

```yaml
methods:
  test_result_summary()  # checks result summary
  test_error_to_dict()  # checks error to dict
```

**class `TestEdgeCases`**

> Tests for edge cases and unusual scenarios.

```yaml
methods:
  test_very_long_lines(tmp_path)  # checks very long lines
  test_many_functions(tmp_path)  # checks many functions
  test_deeply_nested_classes(tmp_path)  # checks deeply nested classes
  test_circular_imports_reference(tmp_path)  # checks circular imports reference
```

**Functions:**

- `error_handler()` — error handler
- `strict_handler()` — strict handler
- `temp_project(tmp_path)` — temp project

---

#### `test_formats.py`

```yaml
path: tests/test_formats.py
lang: python | lines: 393/559
imports: [pytest, json, pathlib.Path, code2logic.analyze_project, code2logic.YAMLGenerator... +12]
constants: [ALL_FORMATS]
```

> Consolidated Format Tests for Code2Logic.

**class `TestAllFormatsGeneration`**

> Test that all formats generate valid output.

```yaml
methods:
  test_format_generates_output(sample_project, fmt)  # checks format generates output
  test_format_contains_class_info(sample_project, fmt)  # checks format contains class info
  test_format_contains_function_info(sample_project, fmt)  # checks format contains function info
```

**class `TestRustSupport`**

```yaml
methods:
  test_rust_parsing_finds_top_level_and_impl_methods(rust_sample_project)  # checks rust parsing finds top level and 
  test_rust_shows_up_in_toon_and_function_logic(rust_sample_project)  # checks rust shows up in toon and functio
```

**class `TestFormatValidation`**

> Test format validation where applicable.

```yaml
methods:
  test_yaml_validation_valid(sample_project)  # checks yaml validation valid
  test_yaml_validation_invalid()  # checks yaml validation invalid
  test_json_validation_valid(sample_project)  # checks json validation valid
  test_json_validation_invalid()  # checks json validation invalid
  test_logicml_validation_valid(sample_project)  # checks logicml validation valid
  test_markdown_validation_valid(sample_project)  # checks markdown validation valid
```

**class `TestFormatEfficiency`**

> Test format size efficiency.

```yaml
methods:
  test_format_sizes(sample_project)  # checks format sizes
  test_logicml_compression(sample_project)  # checks logicml compression
```

**class `TestDetailLevels`**

> Test different detail levels.

```yaml
methods:
  test_detail_levels_ordering(sample_project, fmt)  # checks detail levels ordering
```

**class `TestYAMLSpecifics`**

> YAML-specific tests.

```yaml
methods:
  test_yaml_parseable(sample_project)  # checks yaml parseable
  test_yaml_includes_imports(sample_project)  # checks yaml includes imports
```

**class `TestJSONSpecifics`**

> JSON-specific tests.

```yaml
methods:
  test_json_parseable(sample_project)  # checks json parseable
  test_json_structure(sample_project)  # checks json structure
```

**class `TestLogicMLSpecifics`**

> LogicML-specific tests.

```yaml
methods:
  test_logicml_has_signatures(sample_project)  # checks logicml has signatures
  test_logicml_has_async_markers(samples_project)  # checks logicml has async markers
  test_logicml_token_estimate(sample_project)  # checks logicml token estimate
```

**class `TestGherkinSpecifics`**

> Gherkin-specific tests.

```yaml
methods:
  test_gherkin_has_feature(sample_project)  # checks gherkin has feature
  test_gherkin_has_scenarios(sample_project)  # checks gherkin has scenarios
  test_gherkin_has_steps(sample_project)  # checks gherkin has steps
```

**class `TestTOONSpecifics`**

> TOON-specific tests.

```yaml
methods:
  test_toon_array_syntax(sample_project)  # checks toon array syntax
  test_toon_tabular_syntax(sample_project)  # checks toon tabular syntax
  test_toon_minimal_quoting(sample_project)  # checks toon minimal quoting
  test_toon_with_tabs(sample_project)  # checks toon with tabs
```

**class `TestCSVSpecifics`**

> CSV-specific tests.

```yaml
methods:
  test_csv_has_header(sample_project)  # checks csv has header
  test_csv_consistent_columns(sample_project)  # checks csv consistent columns
```

**class `TestCompactSpecifics`**

> Compact format-specific tests.

```yaml
methods:
  test_compact_is_small(sample_project)  # checks compact is small
  test_compact_has_summary(sample_project)  # checks compact has summary
```

**class `TestCrossFormatEquivalence`**

> Test that all formats contain equivalent information.

```yaml
methods:
  test_all_formats_have_project_name(sample_project)  # checks all formats have project name
  test_all_formats_have_module_info(sample_project)  # checks all formats have module info
```

**Functions:**

- `sample_code()` — sample code
- `sample_project(tmp_path, sample_code)` — sample project
- `samples_project()` — samples project
- `rust_sample_project(tmp_path)` — rust sample project
- `get_generator(fmt:str)` — retrieves generator
- `generate_output(generator, project, detail='standard')` — creates output
- `test_function_logic_toon_js_does_not_default_return_type_to_none(tmp_path)` — checks function logic toon js does not default ret

---

#### `test_generators.py`

```yaml
path: tests/test_generators.py
lang: python | lines: 207/260
imports: [json, pytest, code2logic.MarkdownGenerator, code2logic.CompactGenerator, code2logic.JSONGenerator... +4]
```

> Tests for output generators.

**class `TestMarkdownGenerator`**

> Tests for MarkdownGenerator.

```yaml
methods:
  test_generate_basic(sample_project)  # checks generate basic
  test_generate_includes_modules(sample_project)  # checks generate includes modules
  test_generate_includes_classes(sample_project)  # checks generate includes classes
  test_generate_includes_functions(sample_project)  # checks generate includes functions
  test_generate_includes_entrypoints(sample_project)  # checks generate includes entrypoints
  test_detail_levels(sample_project)  # checks detail levels
```

**class `TestCompactGenerator`**

> Tests for CompactGenerator.

```yaml
methods:
  test_generate_basic(sample_project)  # checks generate basic
  test_generate_includes_hubs(sample_project)  # checks generate includes hubs
  test_compact_is_smaller(sample_project)  # checks compact is smaller
```

**class `TestJSONGenerator`**

> Tests for JSONGenerator.

```yaml
methods:
  test_generate_valid_json(sample_project)  # checks generate valid json
  test_generate_structure(sample_project)  # checks generate structure
  test_generate_modules(sample_project)  # checks generate modules
  test_generate_functions(sample_project)  # checks generate functions
  test_generate_classes(sample_project)  # checks generate classes
```

**Functions:**

- `sample_project()` — sample project

---

#### `test_intent.py`

```yaml
path: tests/test_intent.py
lang: python | lines: 353/504
imports: [pytest, unittest.mock.Mock, unittest.mock.patch, code2logic.intent.IntentAnalyzer, code2logic.intent.IntentType... +5]
```

> Tests for intent analysis functionality.

**class `TestIntentAnalyzer`**

> Test cases for IntentAnalyzer.

```yaml
methods:
  test_init()  # checks init
  test_extract_keywords()  # checks extract keywords
  test_calculate_intent_confidence()  # checks calculate intent confidence
  test_identify_target_module(sample_project_model)  # checks identify target module
  test_identify_target_function(sample_project_model)  # checks identify target function
  test_identify_target_class(sample_project_model)  # checks identify target class
  test_identify_target_project(sample_project_model)  # checks identify target project
  test_generate_description()  # checks generate description
  test_generate_suggestions_refactor()  # checks generate suggestions refactor
  test_generate_suggestions_analyze()  # checks generate suggestions analyze
  test_generate_suggestions_optimize()  # checks generate suggestions optimize
  test_analyze_intent_refactor(sample_project_model)  # checks analyze intent refactor
  # ... +18 more
```

**Functions:**

- `make_function(name, params=None, complexity=1, lines=5, ...+1)` — creates function
- `make_class(name, methods=None, bases=None)` — creates class
- `make_module(name, path, functions=None, classes=None, ...+2)` — creates module
- `make_project(name, modules)` — creates project

---

#### `test_llm_priority.py`

```yaml
path: tests/test_llm_priority.py
lang: python | lines: 44/69
imports: [json, os, pathlib.Path, pytest, code2logic.llm_clients.OpenRouterClient... +2]
```

**Functions:**

- `test_get_client_auto_prefers_override_provider_on_tie(tmp_path:Path, monkeypatch:Any)` — checks get client auto prefers override provider o
- `test_get_client_auto_model_first_uses_model_priority(tmp_path:Path, monkeypatch:Any)` — checks get client auto model first uses model prio

---

#### `test_llm_profiler.py`

```yaml
path: tests/test_llm_profiler.py
lang: python | lines: 416/556
imports: [json, tempfile, os, pathlib.Path, unittest.mock.Mock... +14]
```

> Tests for LLM Profiler module.

**class `TestLLMProfile`**

> Tests for LLMProfile dataclass.

```yaml
methods:
  test_profile_creation()  # checks profile creation
  test_profile_id_consistency()  # checks profile id consistency
  test_profile_defaults()  # checks profile defaults
  test_profile_custom_values()  # checks profile custom values
```

**class `TestDefaultProfiles`**

> Tests for default profile creation.

```yaml
methods:
  test_gpt4_profile()  # checks gpt4 profile
  test_gpt4_turbo_profile()  # checks gpt4 turbo profile
  test_claude_profile()  # checks claude profile
  test_qwen_coder_profile()  # checks qwen coder profile
  test_deepseek_profile()  # checks deepseek profile
  test_llama_70b_profile()  # checks llama 70b profile
  test_llama_7b_profile()  # checks llama 7b profile
  test_mistral_profile()  # checks mistral profile
  test_unknown_model_profile()  # checks unknown model profile
```

**class `TestProfileStorage`**

> Tests for profile storage (save/load).

```yaml
methods:
  test_save_and_load_profile(tmp_path)  # checks save and load profile
  test_load_empty_profiles(tmp_path)  # checks load empty profiles
  test_get_profile(tmp_path)  # checks get profile
  test_get_nonexistent_profile(tmp_path)  # checks get nonexistent profile
  test_get_or_create_profile_existing(tmp_path)  # checks get or create profile existing
  test_get_or_create_profile_new(tmp_path)  # checks get or create profile new
```

**class `TestAdaptiveChunker`**

> Tests for AdaptiveChunker.

```yaml
methods:
  test_chunker_creation()  # checks chunker creation
  test_chunker_default_profile()  # checks chunker default profile
  test_chunk_small_spec()  # checks chunk small spec
  test_chunk_large_spec()  # checks chunk large spec
  test_chunk_format_adjustment()  # checks chunk format adjustment
  test_recommend_format_small_spec()  # checks recommend format small spec
  test_recommend_format_large_spec()  # checks recommend format large spec
  test_estimate_chunks_needed()  # checks estimate chunks needed
```

**class `TestLLMProfiler`**

> Tests for LLMProfiler class.

```yaml
methods:
  test_profiler_creation()  # checks profiler creation
  test_profiler_with_unknown_client()  # checks profiler with unknown client
  test_code_to_spec()  # checks code to spec
  test_extract_code_with_block()  # checks extract code with block
  test_extract_code_without_block()  # checks extract code without block
  test_check_syntax_valid()  # checks check syntax valid
  test_check_syntax_invalid()  # checks check syntax invalid
  test_calculate_similarity()  # checks calculate similarity
  test_run_profile_quick(tmp_path)  # checks run profile quick
  test_metrics_calculation()  # checks metrics calculation
```

**class `TestProfileTestCases`**

> Tests for built-in test cases.

```yaml
methods:
  test_test_cases_exist()  # checks test cases exist
  test_test_cases_valid_python()  # checks test cases valid python
  test_test_cases_have_functions_or_classes()  # checks test cases have functions or clas
```

**class `TestConvenienceFunctions`**

> Tests for convenience functions.

```yaml
methods:
  test_get_adaptive_chunker()  # checks get adaptive chunker
  test_profile_llm_function(tmp_path)  # checks profile llm function
```

---

#### `test_parser_integrity.py`

```yaml
path: tests/test_parser_integrity.py
lang: python | lines: 361/454
imports: [pytest, code2logic.parsers.TreeSitterParser, code2logic.parsers.UniversalParser, code2logic.parsers.is_tree_sitter_available]
```

> Parser integrity tests for code2logic.

**class `TestFunctionNameExtraction`**

> Test 1: Verify complete function names are extracted.

```yaml
methods:
  test_function_name_not_truncated(parser)  # checks function name not truncated
  test_multiple_function_names(parser)  # checks multiple function names
```

**class `TestSignatureParsing`**

> Test 2: Verify complete signatures with types.

```yaml
methods:
  test_signature_with_types(parser)  # checks signature with types
  test_signature_with_defaults(parser)  # checks signature with defaults
```

**class `TestClassNameIntegrity`**

> Test 3: Verify class names have no embedded whitespace.

```yaml
methods:
  test_class_name_no_whitespace(parser)  # checks class name no whitespace
  test_class_with_bases(parser)  # checks class with bases
```

**class `TestImportParsing`**

> Test 4: Verify imports are correctly formatted.

```yaml
methods:
  test_import_from_statement(parser)  # checks import from statement
  test_no_duplicate_imports(parser)  # checks no duplicate imports
```

**class `TestExportsCompleteness`**

> Test 5: Verify exports contain full function names.

```yaml
methods:
  test_exports_complete(parser)  # checks exports complete
```

**class `TestDocstringTruncation`**

> Test 6: Verify docstrings are properly truncated.

```yaml
methods:
  test_long_docstring_truncated(parser)  # checks long docstring truncated
```

**class `TestUnicodeHandling`**

> Test 7: Verify Unicode characters don't break parsing.

```yaml
methods:
  test_unicode_in_docstring(parser)  # checks unicode in docstring
```

**class `TestNestedClassMethods`**

> Test 8: Verify methods in classes are parsed correctly.

```yaml
methods:
  test_class_methods(parser)  # checks class methods
```

**class `TestDecoratorCapture`**

> Test 9: Verify decorators are captured in metadata.

```yaml
methods:
  test_decorators_captured(parser)  # checks decorators captured
```

**class `TestLargeFileHandling`**

> Test 10: Verify large files don't cause truncation.

```yaml
methods:
  test_many_functions(parser)  # checks many functions
```

**class `TestJavaScriptFunctionExtraction`**

> Test JS function extraction: arrow fns, function expressions, IIFEs, nested, etc...

```yaml
methods:
  test_regular_function_declaration(parser)  # checks regular function declaration
  test_async_function_declaration(parser)  # checks async function declaration
  test_const_arrow_function(parser)  # checks const arrow function
  test_let_arrow_function(parser)  # checks let arrow function
  test_var_arrow_function(parser)  # checks var arrow function
  test_const_function_expression(parser)  # checks const function expression
  test_var_function_expression(parser)  # checks var function expression
  test_iife_named_function(parser)  # checks iife named function
  test_nested_function_in_body(parser)  # checks nested function in body
  test_deeply_nested_functions(parser)  # checks deeply nested functions
  test_module_exports_shorthand(parser)  # checks module exports shorthand
  test_commonjs_require_imports(parser)  # checks commonjs require imports
  # ... +3 more
```

**class `TestMethodSignatureIntegrity`**

> Additional tests for method signature integrity.

```yaml
methods:
  test_init_signature(parser)  # checks init signature
```

**Functions:**

- `parser()` — parser
- `parse_python(parser, code:str)` — parses python
- `parse_js(parser, code:str)` — parses js

---

#### `test_reproduction.py`

```yaml
path: tests/test_reproduction.py
lang: python | lines: 208/289
imports: [pytest, sys, pathlib.Path, code2logic.analyze_project, code2logic.ReproductionMetrics... +8]
```

> Tests for code reproduction functionality.

**class `TestYAMLGenerator`**

> Tests for YAML format generation.

```yaml
methods:
  test_yaml_basic()  # checks yaml basic
  test_yaml_includes_classes()  # checks yaml includes classes
  test_yaml_includes_functions()  # checks yaml includes functions
```

**class `TestGherkinGenerator`**

> Tests for Gherkin format generation.

```yaml
methods:
  test_gherkin_basic()  # checks gherkin basic
  test_gherkin_has_scenarios()  # checks gherkin has scenarios
```

**class `TestMarkdownGenerator`**

> Tests for Markdown hybrid format generation.

```yaml
methods:
  test_markdown_basic()  # checks markdown basic
  test_markdown_has_yaml_section()  # checks markdown has yaml section
```

**class `TestReproductionMetrics`**

> Tests for reproduction metrics calculation.

```yaml
methods:
  test_metrics_basic()  # checks metrics basic
  test_metrics_identical_code()  # checks metrics identical code
  test_metrics_different_code()  # checks metrics different code
```

**class `TestChunkedReproduction`**

> Tests for chunked reproduction functionality.

```yaml
methods:
  test_estimate_tokens()  # checks estimate tokens
  test_get_llm_limit()  # checks get llm limit
  test_chunk_yaml_spec()  # checks chunk yaml spec
  test_chunk_gherkin_spec()  # checks chunk gherkin spec
```

**class `TestProjectAnalysis`**

> Tests for project analysis.

```yaml
methods:
  test_analyze_samples()  # checks analyze samples
  test_analyze_detects_classes()  # checks analyze detects classes
  test_analyze_detects_functions()  # checks analyze detects functions
```

**class `TestLogicMLGenerator`**

> Tests for LogicML format generation.

```yaml
methods:
  test_logicml_basic()  # checks logicml basic
  test_logicml_includes_classes()  # checks logicml includes classes
  test_logicml_includes_signatures()  # checks logicml includes signatures
  test_logicml_convenience_function()  # checks logicml convenience function
```

**class `TestFormatComparison`**

> Tests comparing different formats.

```yaml
methods:
  test_yaml_compact()  # checks yaml compact
  test_all_formats_produce_output()  # checks all formats produce output
```

---

#### `test_shared_utils.py`

```yaml
path: tests/test_shared_utils.py
lang: python | lines: 249/326
imports: [pytest, code2logic.shared_utils.compact_imports, code2logic.shared_utils.deduplicate_imports, code2logic.shared_utils.abbreviate_type, code2logic.shared_utils.expand_type... +8]
```

> Tests for shared_utils module.

**class `TestCompactImports`**

> Tests for compact_imports function.

```yaml
methods:
  test_groups_submodules()  # checks groups submodules
  test_preserves_standalone()  # checks preserves standalone
  test_limits_output()  # checks limits output
  test_handles_empty()  # checks handles empty
  test_skips_module_module_duplicates()  # checks skips module module duplicates
```

**class `TestDeduplicateImports`**

> Tests for deduplicate_imports function.

```yaml
methods:
  test_removes_base_when_specific_exists()  # checks removes base when specific exists
  test_handles_empty()  # checks handles empty
```

**class `TestAbbreviateType`**

> Tests for abbreviate_type function.

```yaml
methods:
  test_simple_types()  # checks simple types
  test_complex_types()  # checks complex types
  test_optional_type()  # checks optional type
  test_preserves_unknown()  # checks preserves unknown
  test_handles_empty()  # checks handles empty
```

**class `TestExpandType`**

> Tests for expand_type function.

```yaml
methods:
  test_expands_abbreviated()  # checks expands abbreviated
  test_handles_empty()  # checks handles empty
```

**class `TestBuildSignature`**

> Tests for build_signature function.

```yaml
methods:
  test_removes_self_by_default()  # checks removes self by default
  test_includes_self_when_requested()  # checks includes self when requested
  test_removes_cls()  # checks removes cls
  test_abbreviates_types()  # checks abbreviates types
  test_truncates_params()  # checks truncates params
  test_includes_return_type()  # checks includes return type
  test_no_return_type()  # checks no return type
```

**class `TestRemoveSelfFromParams`**

> Tests for remove_self_from_params function.

```yaml
methods:
  test_removes_self()  # checks removes self
  test_removes_cls()  # checks removes cls
  test_removes_typed_self()  # checks removes typed self
```

**class `TestCategorizeFunction`**

> Tests for categorize_function function.

```yaml
methods:
  test_read_category()  # checks read category
  test_create_category()  # checks create category
  test_update_category()  # checks update category
  test_delete_category()  # checks delete category
  test_handles_method_names()  # checks handles method names
  test_returns_other_for_unknown()  # checks returns other for unknown
```

**class `TestExtractDomain`**

> Tests for extract_domain function.

```yaml
methods:
  test_extracts_known_domain()  # checks extracts known domain
  test_handles_windows_paths()  # checks handles windows paths
  test_returns_parent_for_unknown()  # checks returns parent for unknown
```

**class `TestComputeHash`**

> Tests for compute_hash function.

```yaml
methods:
  test_returns_hex_string()  # checks returns hex string
  test_respects_length()  # checks respects length
  test_same_input_same_hash()  # checks same input same hash
  test_different_input_different_hash()  # checks different input different hash
```

**class `TestTruncateDocstring`**

> Tests for truncate_docstring function.

```yaml
methods:
  test_truncates_long()  # checks truncates long
  test_preserves_short()  # checks preserves short
  test_removes_markers()  # checks removes markers
  test_stops_at_sentence_end()  # checks stops at sentence end
  test_handles_empty()  # checks handles empty
```

**class `TestEscapeForYaml`**

> Tests for escape_for_yaml function.

```yaml
methods:
  test_removes_newlines()  # checks removes newlines
  test_quotes_special_chars()  # checks quotes special chars
  test_handles_empty()  # checks handles empty
```

**class `TestCleanIdentifier`**

> Tests for clean_identifier function.

```yaml
methods:
  test_removes_whitespace()  # checks removes whitespace
  test_handles_empty()  # checks handles empty
```

---

#### `test_yaml_compact.py`

```yaml
path: tests/test_yaml_compact.py
lang: python | lines: 187/241
imports: [pytest, yaml, code2logic.analyze_project, code2logic.generators.YAMLGenerator, code2logic.models.ProjectInfo... +3]
```

> Tests for compact YAML format generation.

**class `TestYAMLShortKeys`**

> Test that YAML uses short keys.

```yaml
methods:
  test_short_keys_in_module(sample_project)  # checks short keys in module
  test_short_keys_in_class(sample_project)  # checks short keys in class
```

**class `TestSelfRemoval`**

> Test that 'self' is removed from method signatures.

```yaml
methods:
  test_no_self_in_signature(sample_project)  # checks no self in signature
```

**class `TestImportDeduplication`**

> Test that imports are deduplicated.

```yaml
methods:
  test_typing_grouped(sample_project)  # checks typing grouped
```

**class `TestEmptyFieldsOmitted`**

> Test that empty fields are omitted.

```yaml
methods:
  test_empty_bases_omitted(sample_project)  # checks empty bases omitted
  test_empty_decorators_omitted(sample_project)  # checks empty decorators omitted
```

**class `TestMetaLegend`**

> Test that meta.legend provides key transparency.

```yaml
methods:
  test_meta_legend_structure(sample_project)  # checks meta legend structure
  test_meta_legend_in_output(sample_project)  # checks meta legend in output
```

**class `TestCompactSizeReduction`**

> Test that compact format reduces output size for larger projects.

```yaml
methods:
  test_compact_smaller_for_large_projects()  # checks compact smaller for large project
```

**class `TestDocstringTruncation`**

> Test that docstrings are truncated.

```yaml
methods:
  test_class_docstring_truncated(sample_project)  # checks class docstring truncated
```

**Functions:**

- `sample_project()` — sample project

---

### 📂 tests/samples

#### `sample_algorithms.py`

```yaml
path: tests/samples/sample_algorithms.py
lang: python | lines: 184/252
imports: [typing.List, typing.Optional, typing.Tuple, typing.Generator, typing.TypeVar... +1]
constants: [T]
```

> Sample algorithms module for reproduction testing.

**Functions:**

- `binary_search(arr:List[int], target:int) -> int` — binary search
- `quicksort(arr:List[int]) -> List[int]` — quicksort
- `merge_sort(arr:List[int]) -> List[int]` — merges sort
- `fibonacci(n:int) -> int` — fibonacci
- `fibonacci_generator(limit:int) -> Generator[int, None, None]` — fibonacci generator
- `is_prime(n:int) -> bool` — is prime
- `sieve_of_eratosthenes(limit:int) -> List[int]` — sieve of eratosthenes
- `gcd(a:int, b:int) -> int` — gcd
- `lcm(a:int, b:int) -> int` — lcm
- `levenshtein_distance(s1:str, s2:str) -> int` — levenshtein distance
- `knapsack_01(weights:List[int], values:List[int], capacity:int) -> int` — knapsack 01

---

#### `sample_api.py`

```yaml
path: tests/samples/sample_api.py
lang: python | lines: 106/139
imports: [typing.Dict, typing.List, typing.Optional, typing.Any, dataclasses.dataclass... +3]
```

> Sample API module for reproduction testing.

**class `APIResponse`**

> Standard API response structure.


**class `User`**

> User model for API.


**class `APIError`(Exception)**

> Custom API error.

```yaml
methods:
  __init__(message:str, code:int=400)  # creates
```

**class `UserAPI`**

> User management API.

```yaml
methods:
  __init__()  # creates
  create_user(username:str, email:str, roles:List[str]=None) -> APIResponse  # creates user
  get_user(user_id:int) -> APIResponse  # retrieves user
  update_user(user_id:int) -> APIResponse  # updates user
  delete_user(user_id:int) -> APIResponse  # deletes user
  list_users(limit:int=10, offset:int=0) -> APIResponse  # list users
  search_users(query:str) -> APIResponse  # filters users
```

**Functions:**

- `async fetch_user_async(api:UserAPI, user_id:int) -> APIResponse` — retrieves user async
- `async create_user_async(api:UserAPI, username:str, email:str) -> APIResponse` — creates user async
- `handle_api_error(func)` — handles api error

---

#### `sample_async.py`

```yaml
path: tests/samples/sample_async.py
lang: python | lines: 163/211
imports: [asyncio, typing.List, typing.Dict, typing.Optional, typing.Any... +5]
constants: [T]
```

> Sample async Python module for reproduction testing.

**class `Task`**

> Async task with status tracking.


**class `TaskResult`**

> Result of task execution.


**class `AsyncTaskQueue`**

> Async task queue with concurrency control.

```yaml
methods:
  __init__(max_concurrent:int=5)  # creates
  async add_task(task:Task) -> None  # creates task
  async get_task(task_id:str) -> Optional[Task]  # retrieves task
  async process_task(task_id:str, handler) -> TaskResult  # processes task
  async process_all(handler) -> List[TaskResult]  # processes all
```

**class `AsyncCache`**

> Simple async cache with TTL.

```yaml
methods:
  __init__(ttl_seconds:int=300)  # creates
  async get(key:str) -> Optional[Any]  # retrieves
  async set(key:str, value:Any) -> None  # updates
  async delete(key:str) -> bool  # deletes
  async clear() -> None  # deletes
```

**Functions:**

- `async async_timer(name:str='operation')` — async timer
- `async fetch_with_retry(url:str, max_retries:int=3, delay:float=1.0) -> Dict[str, Any]` — retrieves with retry
- `async parallel_map(items:List[T], handler, max_concurrent:int=5) -> List[Any]` — parallel map
- `async race() -> Any` — race
- `async timeout_wrapper(coro, timeout_seconds:float, default=None)` — timeout wrapper

---

#### `sample_class.py`

```yaml
path: tests/samples/sample_class.py
lang: python | lines: 74/97
imports: [typing.List, typing.Dict, typing.Optional, typing.Any]
```

> Sample file with a class for reproduction testing.

**class `Calculator`**

> Simple calculator with history.

```yaml
methods:
  __init__(precision:int=2)  # creates
  add(a:float, b:float) -> float  # creates
  subtract(a:float, b:float) -> float  # subtract
  multiply(a:float, b:float) -> float  # multiply
  divide(a:float, b:float) -> Optional[float]  # splits
  clear_history() -> None  # deletes history
  get_history() -> List[str]  # retrieves history
```

---

#### `sample_csharp.cs`

```yaml
path: tests/samples/sample_csharp.cs
lang: csharp | lines: 28/37
imports: [System, System.Collections.Generic, System.Linq]
```

**interface `IHasId`**

**record `User`**

**interface `IHasId`**


**class `User`**


**Functions:**

- `static FilterActive(IEnumerable<User> users) -> List<User>` — filters active
- `static FindById(IEnumerable<User> users, string id) -> User` — retrieves by id

---

#### `sample_dataclasses.py`

```yaml
path: tests/samples/sample_dataclasses.py
lang: python | lines: 68/83
imports: [dataclasses.dataclass, dataclasses.field, typing.Optional, typing.List, typing.Dict... +1]
```

> Sample file with dataclasses for reproduction testing.

**class `User`**

> Represents a user in the system.


**class `Product`**

> Represents a product in the catalog.


**class `Order`**

> Represents a customer order.


**class `Address`**

> Represents a shipping address.


---

#### `sample_enum.py`

```yaml
path: tests/samples/sample_enum.py
lang: python | lines: 78/96
imports: [enum.Enum, enum.IntEnum, enum.auto, typing.List]
```

> Sample enum types for testing.

**enum `Status`**

**enum `Priority`**

**enum `Color`**

**enum `HttpStatus`**

**enum `TaskType`**

**class `Status`(Enum)**

> Status enumeration with auto values.


**class `Priority`(IntEnum)**

> Priority levels as integers.


**class `Color`(Enum)**

> Color enumeration with string values.

```yaml
methods:
  from_hex(hex_code:str) -> "Color"  # from hex
```

**class `HttpStatus`(IntEnum)**

> HTTP status codes.

```yaml
methods:
  is_success() -> bool  # is success
  is_error() -> bool  # is error
```

**class `TaskType`(Enum)**

> Task type enumeration.

```yaml
methods:
  get_timeout() -> int  # retrieves timeout
  get_allowed_statuses() -> List[Status]  # retrieves allowed statuses
```

---

#### `sample_functions.py`

```yaml
path: tests/samples/sample_functions.py
lang: python | lines: 89/124
imports: [typing.List, typing.Dict, typing.Optional, typing.Any, json... +1]
```

> Sample file with functions for reproduction testing.

**Functions:**

- `calculate_total(items:List[int], tax_rate:float=0.1) -> float` — processes total
- `filter_by_status(records:List[Dict], status:str) -> List[Dict]` — filters by status
- `merge_configs(base:Dict[str,Any], override:Dict[str,Any]) -> Dict[str, Any]` — merges configs
- `validate_email(email:str) -> bool` — validates email
- `load_json_file(path:str) -> Optional[Dict]` — retrieves json file
- `get_env_or_default(key:str, default:str='') -> str` — retrieves env or default
- `chunk_list(items:List[Any], chunk_size:int) -> List[List[Any]]` — chunk list
- `format_currency(amount:int, currency:str='USD') -> str` — formats currency

---

#### `sample_go.go`

```yaml
path: tests/samples/sample_go.go
lang: go | lines: 90/120
imports: [errors]
```

**struct `User`**

**struct `Product`**

**struct `Order`**

**struct `OrderItem`**

**struct `UserService`**

**class `User`**


**class `Product`**


**class `Order`**


**class `OrderItem`**


**class `UserService`**


**interface `Repository`**


**Functions:**

- `NewUserService()` — creates user service
- `GetUser(id int) -> (*User, error)` — retrieves user
- `CreateUser(name, email string) -> (*User, error)` — creates user
- `CalculateTotal(items []OrderItem, taxRate float64) -> int64` — processes total
- `FilterProducts(products []Product, predicate func(Product) -> bool` — filters products
- `FormatPrice(cents int64, currency string) -> string` — formats price

---

#### `sample_java.java`

```yaml
path: tests/samples/sample_java.java
lang: java | lines: 43/54
imports: [java.util.ArrayList, java.util.Collections, java.util.Comparator, java.util.List, java.util.Objects]
constants: [DEFAULT_LIMIT]
```

**interface `Identifiable`**

**enum `Status`**

**record `User`**

**class `SampleJava`**


**interface `Identifiable`**


**class `User`**


**Functions:**

- `getId() -> String` — retrieves id
- `static filterActive(List<User> users) -> List<User>` — filters active
- `static sortByName(List<User> users) -> List<User>` — sorts by name

---

#### `sample_javascript.js`

```yaml
path: tests/samples/sample_javascript.js
lang: javascript | lines: 154/168
```

> Sample JavaScript file for reproduction testing. Tests classes, functions, and async patterns.

**class `User`**

```yaml
methods:
  constructor(id, name, email)  # constructor
  getDisplayName()  # retrieves display name
  deactivate()  # deactivate
```

**class `Product`**

```yaml
methods:
  constructor(sku, name, price)  # constructor
  addTags()  # creates tags
  isInStock()  # is in stock
  formatPrice()  # formats price
```

**Functions:**

- `calculateTotal(items)` — processes total
- `filterBy(arr, predicate)` — filters by
- `async fetchData(url)` — retrieves data
- `debounce(func, wait)` — debounce
- `deepClone(obj)` — deep clone

---

#### `sample_javascript_advanced.js`

```yaml
path: tests/samples/sample_javascript_advanced.js
lang: javascript | lines: 111/144
imports: [fs, path, events]
```

> Advanced JavaScript sample for testing TOON parser coverage. Covers: IIFEs, nested functions, var de

**class `FileProcessor`(EventEmitter)**

```yaml
methods:
  constructor(rootDir)  # constructor
  async analyze(filePath)  # processes
  static fromConfig(configPath)  # from config
  getResults()  # retrieves results
```

**Functions:**

- `getArg(name, def)` — retrieves arg
- `processItem(item)` — processes item
- `validate(input)` — validates
- `async fetchResource(url, options)` — retrieves resource
- `shouldIgnore(filePath)` — checks ignore
- `formatOutput(data, indent)` — formats output
- `walk(dir, onFile)` — walk
- `findFiles(dir, extensions)` — retrieves files
- `async processFiles(pattern)` — processes files
- `outerFunction(data)` — outer function
- `main()` — main
- `traverse(currentDir)` — traverse
- `middleHelper(items)` — middle helper
- `innerSort(a, b)` — inner sort

---

#### `sample_pydantic.py`

```yaml
path: tests/samples/sample_pydantic.py
lang: python | lines: 62/79
imports: [pydantic.BaseModel, pydantic.Field, pydantic.field_validator, typing.List, typing.Optional... +2]
```

> Sample Pydantic models for testing.

**enum `TaskStatus`**

**class `TaskStatus`(str, Enum)**

> Task status enumeration.


**class `Task`(BaseModel)**

> Task model with Pydantic features.

```yaml
methods:
  name_not_empty(v:str) -> str  # name not empty
```

**class `TaskQueue`(BaseModel)**

> Task queue container.

```yaml
methods:
  add_task(task:Task) -> bool  # creates task
  get_pending() -> List[Task]  # retrieves pending
  get_by_status(status:TaskStatus) -> List[Task]  # retrieves by status
```

**class `Project`(BaseModel)**

> Project model with nested models.

```yaml
methods:
  total_tasks() -> int  # total tasks
```

---

#### `sample_rust.rs`

```yaml
path: tests/samples/sample_rust.rs
lang: rust | lines: 168/210
imports: [std::collections::HashMap, std::fmt]
```

**struct `User`**

**struct `Product`**

**struct `Order`**

**struct `OrderItem`**

**struct `Repository`**

**class `User`**


**class `Product`**


**class `Order`**


**class `OrderItem`**


**class `Repository`**


**class `AppError`**


**Functions:**

- `new() -> Self` — creates
- `add(&mut self, item: T)` — creates
- `get(&self, id: &str) -> Option<&T>` — retrieves
- `get_all(&self) -> Vec<&T>` — retrieves all
- `delete(&mut self, id: &str) -> bool` — deletes
- `count(&self) -> usize` — count
- `create_user(id: u64, name: &str, email: &str) -> User` — creates user
- `calculate_order_total(items: &[OrderItem]) -> f64` — processes order total
- `validate_email(email: &str) -> Result<()>` — validates email
- `process_order(mut order: Order) -> Result<Order>` — processes order

---

#### `sample_sql.sql`

```yaml
path: tests/samples/sample_sql.sql
lang: sql | lines: 73/94
```

**table `users`**

**table `products`**

**table `orders`**

**table `order_items`**

**view `order_summary`**

**class `users`**


**class `products`**


**class `orders`**


**class `order_items`**


**class `order_summary`**


**Functions:**

- `calculate_order_total()` — processes order total
- `update_order_total()` — updates order total

---

#### `sample_sql_dsl.py`

```yaml
path: tests/samples/sample_sql_dsl.py
lang: python | lines: 177/232
imports: [typing.List, typing.Optional, typing.Dict, typing.Any, typing.Union... +3]
```

> Sample SQL DSL module for reproduction testing.

**enum `SQLOperator`**

**enum `JoinType`**

**class `SQLOperator`(Enum)**

> SQL comparison operators.


**class `JoinType`(Enum)**

> SQL join types.


**class `Condition`**

> SQL WHERE condition.

```yaml
methods:
  to_sql() -> str  # converts sql
```

**class `Join`**

> SQL JOIN clause.

```yaml
methods:
  to_sql() -> str  # converts sql
```

**class `QueryBuilder`**

> Fluent SQL query builder.

```yaml
methods:
  __init__(table:str)  # creates
  select() -> "QueryBuilder"  # retrieves
  where(column:str, operator:Union[SQLOperator,str], value:Any=None) -> "QueryBuilder"  # where
  join(table:str, on_left:str, on_right:str, join_type:JoinType=JoinType.INNER) -> "QueryBuilder"  # merges
  order_by() -> "QueryBuilder"  # sorts by
  group_by() -> "QueryBuilder"  # group by
  limit(count:int) -> "QueryBuilder"  # limit
  offset(count:int) -> "QueryBuilder"  # offset
  build() -> str  # creates
```

**class `InsertBuilder`**

> SQL INSERT query builder.

```yaml
methods:
  __init__(table:str)  # creates
  columns() -> "InsertBuilder"  # columns
  values() -> "InsertBuilder"  # values
  build() -> str  # creates
```

**class `UpdateBuilder`**

> SQL UPDATE query builder.

```yaml
methods:
  __init__(table:str)  # creates
  set(column:str, value:Any) -> "UpdateBuilder"  # updates
  where(column:str, operator:SQLOperator, value:Any=None) -> "UpdateBuilder"  # where
  build() -> str  # creates
```

**Functions:**

- `select(table:str) -> QueryBuilder` — retrieves
- `insert(table:str) -> InsertBuilder` — creates
- `update(table:str) -> UpdateBuilder` — updates

---

#### `sample_typescript.ts`

```yaml
path: tests/samples/sample_typescript.ts
lang: typescript | lines: 114/142
```

> Sample TypeScript file for reproduction testing. Tests interfaces, generics, and type annotations.

**interface `User`**

**interface `Product`**

**interface `OrderItem`**

**interface `Order`**

**type `Result`**

**class `Repository`**

```yaml
methods:
  add(item: T) -> void  # creates
  get(id: T['id']) -> Nullable<T>  # retrieves
  getAll() -> T[]  # retrieves all
  delete(id: T['id']) -> boolean  # deletes
  count() -> number  # count
```

**Functions:**

- `createUser(id: number, name: string, email: string) -> User` — creates user
- `calculateOrderTotal(items: OrderItem[]) -> number` — processes order total
- `filterByStatus(items: T[], status: string) -> T[]` — filters by status
- `groupBy(items: T[], key: K) -> Map<T[K], T[]>` — group by
- `async fetchUser(id: number) -> Promise<Result<User>>` — retrieves user
- `async processOrder(order: Order) -> Promise<Result<Order>>` — processes order

---

### 📂 tests/samples/sample_reexport

#### `__init__.py`

```yaml
path: tests/samples/sample_reexport/__init__.py
lang: python | lines: 17/22
imports: [models.User, models.Order, models.Product, utils.process_data, utils.validate_input... +2]
```

> Sample re-export module for testing.

---

#### `exceptions.py`

```yaml
path: tests/samples/sample_reexport/exceptions.py
lang: python | lines: 7/12
```

> Sample exceptions for re-export testing.

**class `ValidationError`(Exception)**

> Validation error.


**class `ProcessingError`(Exception)**

> Processing error.


---

#### `models.py`

```yaml
path: tests/samples/sample_reexport/models.py
lang: python | lines: 21/29
imports: [dataclasses.dataclass, typing.List, typing.Optional]
```

> Sample models for re-export testing.

**class `User`**

> User model.


**class `Order`**

> Order model.


**class `Product`**

> Product model.


---

#### `utils.py`

```yaml
path: tests/samples/sample_reexport/utils.py
lang: python | lines: 8/14
imports: [typing.Any, typing.Dict]
```

> Sample utilities for re-export testing.

**Functions:**

- `process_data(data:Dict[str,Any]) -> Dict[str, Any]` — processes data
- `validate_input(data:Dict[str,Any]) -> bool` — validates input

---

### 📂 tests/samples/sample_ts_reexport

#### `index.ts`

```yaml
path: tests/samples/sample_ts_reexport/index.ts
lang: typescript | lines: 2/6
```

> Sample TypeScript re-export index for reproduction tests.

---

#### `math.ts`

```yaml
path: tests/samples/sample_ts_reexport/math.ts
lang: typescript | lines: 6/10
```

> Math helpers for re-export tests

**Functions:**

- `add(a: number, b: number) -> number` — creates
- `multiply(a: number, b: number) -> number` — multiply

---

#### `types.ts`

```yaml
path: tests/samples/sample_ts_reexport/types.ts
lang: typescript | lines: 5/9
```

> Types for re-export tests

**type `Result`**

**interface `User`**

---
