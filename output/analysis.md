# 📦 code2logic

```yaml
generated: 2026-01-03T12:50:50.955139
files: 24
lines: 8219
languages: {"python": 24}
entrypoints: ["code2logic/cli.py", "code2logic/llm.py", "code2logic/analyzer.py", "code2logic/parsers.py", "code2logic/intent.py"]
```

## 📁 Structure

```
├── code2logic/
│   ├── __init__.py: [python]
│   ├── analyzer.py: [python] ProjectAnalyzer, 
    path: str,, us() -> Dict[str, 
│   ├── cli.py: [python] ensure_dependencies, main
│   ├── dependency.py: [python] DependencyAnalyzer, is_networkx_available
│   ├── generators.py: [python] MarkdownGenerator, ltra-compact out,  output for m +2
│   ├── gherkin.py: [python] GherkinGenerator, nitionGenerator:
    "", YAMLGenerator:
    "" +1
│   ├── intent.py: [python] EnhancedIntentGenerator
│   ├── llm.py: [python] OllamaClient, LiteLLMClient, CodeAnalyzer +1
│   ├── mcp_server.py: [python] handle_request, call_tool, run_server
│   ├── models.py: [python]
│   ├── parsers.py: [python] TreeSitterParser, UniversalParser, is_tree_sitter_available
│   └── similarity.py: [python] SimilarityDetector, is_rapidfuzz_available
├── examples/
│   ├── bdd_workflow.py: [python] lculate_token_savings(c, in()
│   ├── compare_projects.py: [python] compute_hash, extract_elements, compare_projects +2
│   ├── duplicate_detection.py: [python] compute_signature_hash, compute_name_hash, normalize_intent +5
│   ├── generate_code.py: [python] check_ollama, generate_with_ollama, generate_code_from_csv +1
│   ├── quick_start.py: [python] main
│   ├── refactor_suggestions.py: [python] check_ollama, generate_with_ollama, find_issues +2
│   └── token_efficiency.py: [python] count_tokens, estimate_cost, analyze_format +1
└── tests/
    ├── __init__.py: [python]
    ├── conftest.py: [python] sample_python_code, sample_javascript_code, sample_java_code +6
    ├── test_analyzer.py: [python] TestProjectAnalyzer
    ├── test_generators.py: [python] sample_project, TestMarkdownGenerator, tCompactGenerator:
  +1
    └── test_intent.py: [python] TestIntentAnalyzer
```

## 📄 Modules

### 📂 code2logic

#### `__init__.py`

```yaml
path: code2logic/__init__.py
lang: python | lines: 76/88
```

> Code2Logic - Convert source code to logical representation for LLM analysis.

A Python library that 

---

#### `analyzer.py`

```yaml
path: code2logic/analyzer.py
lang: python | lines: 221/287
imports: [sys]
```

> Main project analyzer orchestrating all analysis components.

Provides the high-level API for analyz

**class `ProjectAnalyzer`**

> Main class for analyzing software projects.
    
    Orchestrates:
    - File scanning and language 

```yaml
methods:
  __init__(self, root_path:str, use_treesitter:bool, verbose:bool, ...+1)  # Initialize the project analyzer.
   Projec(Info) ->      """
    # yze the project.
  ):
        (""Sc)  # e all source files."""
  nts(self) -> List[s(r]:
) ->    """Det  # t entry points."""
  elf) -> Dict:
(    ) -> "
    # analysis statistics.
```

**Functions:**

- `
    path: str,(se_t:esi, bool = True,
 : ver, False,
:-> P) ->  """
    Co` — nce function to analyze a project.
- `us() -> Dict[str, () -> 
    """
    Ge` — lability status of optional libraries.

---

#### `cli.py`

```yaml
path: code2logic/cli.py
lang: python | lines: 187/223
imports: [argparse, os, sys, subprocess]
```

> Command-line interface for Code2Logic.

Usage:
    code2logic /path/to/project
    code2logic /path/

**Functions:**

- `ensure_dependencies()` — Auto-install optional dependencies for best result
- `main()` — Main CLI entry point.

---

#### `dependency.py`

```yaml
path: code2logic/dependency.py
lang: python | lines: 187/247
constants: [NETWORKX_AVAILABLE]
```

> Dependency graph analyzer using NetworkX.

Builds and analyzes dependency graphs from module imports

**class `DependencyAnalyzer`**

> Analyzes dependency graphs using NetworkX.
    
    Computes:
    - PageRank: Importance metric for 

```yaml
methods:
  __init__(self)  # Initialize the dependency analyzer.
  build_graph(self, modules:List[ModuleInfo]) -> Dict[str, List[str]]  # Build dependency graph from modules.
  analyze_metrics(self) -> Dict[str, DependencyNode]  # Compute metrics for each node in the gra
  get_entrypoints(self) -> List[str]  # entry points (nodes with no incoming edg
  get_hubs(self) -> List[str]  # hub modules (high centrality).
  detect_cycles(self) -> List[List[str]]  # Detect dependency cycles.
  get_strongly_connected_components(self) -> List[List[str]]  # strongly connected components.
  get_dependency_depth(self, module_path:str) -> int  # the maximum depth of dependencies for a 
```

**Functions:**

- `is_networkx_available() -> bool` — Check if NetworkX is available.

---

#### `generators.py`

```yaml
path: code2logic/generators.py
lang: python | lines: 827/1018
imports: [json]
```

> Output generators for Code2Logic analysis results.

Includes:
- MarkdownGenerator: Human-readable Ma

**class `MarkdownGenerator`**

> Generates Markdown output for project analysis.
    
    Produces human-readable documentation with:

```yaml
methods:
  generate(self, project:ProjectInfo, detail_level:str) -> str  # Generate Markdown output.
  : List[st(], p, ject::rojectInf, :
     : """Generat)  # tructure tree."""
  : List[str]( tre,  dict:prefix: s, , de:h: i,  = 0)::   , ...+1)  # ee structure."""
  : ModuleInf(, 
 ,      :         , e:il: str, p,     "":ene, ...+1)  # fn = Path(m.path).name
  lassInfo, (etai,  str):        ", Gen:ate class, ocumen:tio)  # kind = "interface" if cls.is_int
     "("Gen, a: function si) -> re.  # "
```

**class `ltra-compact out`**

> r token efficiency.
    
    Optimized for minimal token usage while preserving
    essential inform

```yaml
methods:
  tr:
    (   ", 
      :Generate co) ->  ou  # tr:
```

**class ` output for m`**

>  processing.
    
    Suitable for:
    - RAG (Retrieval-Augmented Generation) systems
    - Databas

```yaml
methods:
  : bool =(Fals,  
     :          d, ail::tr =,  """
 :   ) -> tpu  # : bool =
  fo, detail: str)(-> s, :
     : """Generat, nested:SON) -> ctu  # def ser_func(f: FunctionInfo) ->
  , detail: str)(-> s, :
     : """Generat, flat J:N l) -> or   # , detail: str)
  lem_type: str, nam(: st, 
:          ,          : si, atur: st, ...+4) -> ow f  # lem type: str, nam
  > str:
        "("Bui,  :mpact signat) -> ""
  # rams = ','.join(f.params[:4])
      """Cate(oriz, by n:e p) -> n."  # name_lower = name.lower().split(
          """Extr(ct d, ain :om ) -> """  # arts = path.lower().replace('\\
   str) -> str:(    ,   "":omp, e short h:h.") ->      #  str)  > str:
```

**class ` output for h`**

> eadable representation.
    
    Supports both nested (hierarchical) and flat (table-like) formats.


```yaml
methods:
  : bool =(Fals,  
     :          d, ail::tr =,  """
 :   ) -> tpu  # : bool =
  fo, detail: str)(-> d, t:
    :  """Build , at dat:str) -> e op  # r comparisons."""
  Info, detail: str)(-> d, t:
    :  """Build , sted h:rar) -> l da  # info, detail: str)
  r, name: s(r, s, natu:: s, ,
       :   ,     :ang, ...+4) -> lat   # r, name: s
  unctionInfo, langua(e: s, ,
  :   ,  :            ,  detail::tr,, ...+3) ->  fun  # sig = self._build_signature(f)
  name: str, f: Fun(tion, fo,
:   ,           :   ,  :nguage: str,, ...+4) -> od."  # sig = self._build_signature(f)
  detail: str) -> d(ct:
,  :   """Conver, functi: to) ->  for  # detail: str)  > d
  tail: str) -> d(ct:
,  :   """Conver, method:o d) -> or n  # tail: str)  > d
  > str:
        "("Bui,  :mpact signat) -> tri  # params = ','.join(f.params[:4])
      """Cate(oriz, func:on ) -> me   # name_lower = name.lower()
          """Extr(ct d, ain :om ) -> pat  # parts = path.lower().replace('\\
   str) -> str:(    ,   "":omp, e short h:h f) -> ick  #  str)  > str:
  # ... +1 more
```

**class `utput optimi`**

> r LLM processing.
    
    CSV is the most token-efficient format (~50% smaller than JSON).
    Each

```yaml
methods:
  il: str ( 'st, dard') : str:
     ,  """
 :   ) -> put  # il: str
  : str, nam(: st,  :          ,       sig:tur,  str:cal, ...+4) -> ""
   # : str, nam
  elem_type: str, nam(: st, 
:          ,          :  f, Func:onI, ...+4) -> on/m  # sig = self._build_signature(f)
  > str:
        "("Bui,  :mpact signat) -> ""
  # rams = ','.join(p.split(':')[0
      """Cate(oriz, func:on ) -> me   # name_lower = name.lower().split
          """Extr(ct d, ain :om ) -> pat  # parts = path.lower().replace('\\
   str) -> str:(    ,   "":omp, e short h:h f) -> ick  #  str)  > str:
      """Esca(e te,  for:SV ) -> ve   # limit commas)."""
```

---

#### `gherkin.py`

```yaml
path: code2logic/gherkin.py
lang: python | lines: 766/981
imports: [re, hashlib]
```

> Gherkin/BDD Generator for Code2Logic.

Generates Gherkin feature files from code analysis for:
- Ult

**class `GherkinGenerator`**

> Generates Gherkin feature files from code analysis.
    
    Achieves ~50x token compression compare

```yaml
methods:
  (self, l(ngua, : str = :n'))  # Initialize GherkinGenerator.
  (self, p(ojec,  Projec:nfo, detail, str = :tan, : str = :oma) ->      # Generate Gherkin feature files from proj
  feature(self, g(oup_, me: str, i:ms:, ist[d:t], 
     ,  Projec:nfo, group_, ...+1) -> eature:
        # e a Gherkin feature from grouped items."
  scenario(self, c(tego, : str, i:ms:, ist[d:t], 
     , str) -:Ghe) -> cenario:
        # e a scenario from category items."""
  edge_case_scenarios(self, c(tego, : str, 
:   , ist[d:t]) -> Lis) -> rkinScenario]:
        # e edge case scenarios for thorough testi
  when_step(self, f(nc: , ncti:Info, verb: , r) -:str) ->      # e a When step from function info."""
  background(self, d(main, str, 
:   , ist[d:t]) -> Opt) -> [List[str]]:
        # e background steps for common setup."""
  examples_table(self, i(ems:, ist[d:t]) -> Lis) -> t[str, str]]:
        # e Examples table for Scenario Outline.""
  r_step(self, s(ep_t, e: str, p:ter,  str, f:c: , ncti:Info):
     )  # ter a step definition for later generati
  features(self, f(atur, : List[G:rkinFeature], 
     , str) -:str) ->      # r features to Gherkin text."""
  feature(self, f(atur,  Gherki:eature, detail, str) -:str) ->      # r a single feature."""
  scenario(self, s(enar, : Gherki:cenario, detail, str) -:str) ->      # r a single scenario."""
  # ... +1 more
```

**class `nitionGenerator:
    ""`**

> Generates step definition stubs from Gherkin features.
    
    Supports multiple frameworks:
    - 


**class `YAMLGenerator:
    ""`**

> Generates Cucumber YAML configuration and test data.
    
    YAML format provides ~5x token compres

```yaml
methods:
  (self, p(ojec,  Projec:nfo, detail, str = :tan) ->      # ate Cucumber YAML configuration."""
  ize(self, n(me: , r) -:str) ->      # orize by name pattern."""
```

**Functions:**

- `herkin(csv_con(ent: str, l:gua, : str = :n')) ->  ""` — Convert CSV analysis directly to Gherkin.

---

#### `intent.py`

```yaml
path: code2logic/intent.py
lang: python | lines: 180/247
imports: [re]
```

> Enhanced Intent Generator with NLP support.

Uses lemmatization, pattern matching, and docstring ext

**class `EnhancedIntentGenerator`**

> Generator intencji z NLP - lemmatyzacja, ekstrakcja z docstringów.
    
    Supports both English an

```yaml
methods:
  __(self,(lang, str :'en)  # Initialize the intent generator.
  te(self,(name, str,:ocs, ing: Opti:al[str] = Non) ->      # Generate intent from function name and o
  ct_from_docstring(self,(docs, ing: str):> O) -> al[str]:
      # ract intent from docstring's first line.
  ailable_features(cls) (> d) -> tr, bool]:
      # dictionary of available NLP features.
```

---

#### `llm.py`

```yaml
path: code2logic/llm.py
lang: python | lines: 355/450
imports: [json]
```

> LLM Integration for Code2Logic

Provides integration with local Ollama and LiteLLM for:
- Code gener

**class `OllamaClient`**

> Direct Ollama API client.

```yaml
methods:
  __init__(self, config:LLMConfig)  # creates
  generate(self, prompt:str, system:Optional[str]) -> str  # Generate completion from Ollama.
  chat(self, messages:List[Dict[str, str]]) -> str  # Chat completion from Ollama.
  is_available(self) -> bool  # Check if Ollama is running.
  list_models(self) -> List[str]  # List available models.
```

**class `LiteLLMClient`**

> LiteLLM client for unified API access.

```yaml
methods:
  __init__(self, config:LLMConfig)  # creates
  generate(self, prompt:str, system:Optional[str]) -> str  # Generate completion via LiteLLM.
  chat(self, messages:List[Dict[str, str]]) -> str  # Chat completion via LiteLLM.
  is_available(self) -> bool  # Check if LiteLLM backend is available.
```

**class `CodeAnalyzer`**

> LLM-powered code analysis for Code2Logic.
    
    Example:
        >>> from code2logic import analy

```yaml
methods:
  __init__(self, model:str, provider:str, base_url:str)  # Initialize CodeAnalyzer.
  is_available(self) -> bool  # Check if LLM backend is available.
  suggest_refactoring(self, project) -> List[Dict[str, Any]]  # Analyze project and suggest refactoring 
  find_semantic_duplicates(self, project) -> List[Dict[str, Any]]  # Find semantically similar functions usin
  generate_code(self, project, target_lang:str, module_filter:Optional[str]) -> Dict[str, str]  # Generate code in target language from pr
  translate_function(self, name:str, signature:str, intent:str, ...+2) -> str  # Translate a single function to another l
```

**Functions:**

- `get_available_backends() -> Dict[str, bool]` — availability status of LLM backends.

---

#### `mcp_server.py`

```yaml
path: code2logic/mcp_server.py
lang: python | lines: 292/355
imports: [json, sys]
```

> MCP (Model Context Protocol) Server for Code2Logic

Provides Code2Logic functionality as an MCP serv

**Functions:**

- `handle_request(request:dict) -> dict` — Handle incoming MCP request.
- `call_tool(tool_name:str, arguments:dict) -> str` — Execute a tool and return result.
- `run_server()` — Run the MCP server.

---

#### `models.py`

```yaml
path: code2logic/models.py
lang: python | lines: 150/171
```

> Data models for Code2Logic.

Contains dataclasses representing the analyzed code structure:
- Functi

---

#### `parsers.py`

```yaml
path: code2logic/parsers.py
lang: python | lines: 772/909
imports: [ast, re]
constants: [TREE_SITTER_AVAILABLE]
```

> Code parsers for multiple languages.

Includes:
- TreeSitterParser: High-accuracy AST parsing using 

**class `TreeSitterParser`**

> Parser using Tree-sitter for high-accuracy AST parsing.
    
    Supports Python, JavaScript, and Ty

```yaml
methods:
  __init__(self)  # Initialize Tree-sitter parsers for avail
  is_available(self, language:str) -> bool  # Check if Tree-sitter parser is available
  get_supported_languages(cls) -> List[str]  # list of potentially supported languages.
  parse(self, filepath:str, content:str, language:str) -> Optional[ModuleInfo]  # Parse a source file using Tree-sitter.
```

**class `UniversalParser`**

> Fallback parser using Python AST and regex.
    
    Used when Tree-sitter is not available. Provide

```yaml
methods:
  __init__(self)  # Initialize the universal parser.
  parse(self, filepath:str, content:str, language:str) -> Optional[ModuleInfo]  # Parse a source file using AST or regex.
```

**Functions:**

- `is_tree_sitter_available() -> bool` — Check if Tree-sitter is available.

---

#### `similarity.py`

```yaml
path: code2logic/similarity.py
lang: python | lines: 123/166
constants: [RAPIDFUZZ_AVAILABLE]
```

> Similarity detector using Rapidfuzz.

Detects similar functions across modules to identify
potential

**class `SimilarityDetector`**

> Detects similar functions using fuzzy string matching.
    
    Uses Rapidfuzz for fast similarity c

```yaml
methods:
  __init__(self, threshold:float)  # Initialize the similarity detector.
  find_similar_functions(self, modules:List[ModuleInfo]) -> Dict[str, List[str]]  # Find similar functions across all module
  find_duplicate_signatures(self, modules:List[ModuleInfo]) -> Dict[str, List[str]]  # Find functions with identical signatures
```

**Functions:**

- `is_rapidfuzz_available() -> bool` — Check if Rapidfuzz is available.

---

### 📂 examples

#### `bdd_workflow.py`

```yaml
path: examples/bdd_workflow.py
lang: python | lines: 169/239
imports: [s
i, on
f]
```

> Example: Complete BDD Workflow with code2logic.

Demonstrates the full BDD workflow:
1. Analyze code

**Functions:**

- `lculate_token_savings(c(v_content: :r, , erkin_content: :r) ) -> ct:
` — Calculate token savings between formats."""
- `in()()` — Run BDD workflow."""

---

#### `compare_projects.py`

```yaml
path: examples/compare_projects.py
lang: python | lines: 176/228
imports: [sys, json]
```

> Example: Compare two projects for duplicates and similarities.

This script demonstrates how to:
1. 

**Functions:**

- `compute_hash(name:str, signature:str) -> str` — Compute hash for function comparison.
- `extract_elements(project) -> dict` — Extract all elements from project with hashes.
- `compare_projects(project1, project2) -> dict` — Compare two projects.
- `print_comparison(result:dict)` — Print comparison results.
- `"Mai()` — "mai

---

#### `duplicate_detection.py`

```yaml
path: examples/duplicate_detection.py
lang: python | lines: 286/371
imports: [sys, json, hashlib]
```

> Example: Duplicate Detection and Deduplication Report.

Finds duplicates in codebases using multiple

**Functions:**

- `compute_signature_hash(params:List[str], return_type:str) -> str` — Compute hash of function signature (ignoring name)
- `compute_name_hash(name:str, params:List[str]) -> str` — Compute hash for exact duplicate detection.
- `normalize_intent(intent:str) -> str` — Normalize intent for comparison.
- `find_duplicates(project) -> Dict[str, Any]` — Find all types of duplicates in project.
- `categorize(name:str) -> str` — Categorize function by name.
- `extract_domain(path:str) -> str` — Extract domain from path.
- `generate_report(results:Dict[str, Any], project_name:str) -> str` — Generate markdown deduplication report.
- `main()` — Run duplicate detection.

---

#### `generate_code.py`

```yaml
path: examples/generate_code.py
lang: python | lines: 111/161
imports: [sys, json]
constants: [OLLAMA_HOST, MODEL]
```

> Example: Generate code from CSV analysis using local Ollama.

This script demonstrates how to:
1. An

**Functions:**

- `check_ollama()` — Check if Ollama is running.
- `generate_with_ollama(prompt:str, system:str) -> str` — Generate text using Ollama API.
- `generate_code_from_csv(csv_content:str, target_lang:str) -> dict` — Generate code from CSV analysis.
- `main()` — Main example.

---

#### `quick_start.py`

```yaml
path: examples/quick_start.py
lang: python | lines: 151/239
imports: [sys]
```

> Quick Start Guide for code2logic.

This script demonstrates the most common use cases in 5 minutes.


**Functions:**

- `main()` — main

---

#### `refactor_suggestions.py`

```yaml
path: examples/refactor_suggestions.py
lang: python | lines: 203/273
imports: [sys, json]
constants: [OLLAMA_HOST, MODEL]
```

> Example: LLM-powered refactoring suggestions using Ollama.

This script demonstrates how to:
1. Anal

**Functions:**

- `check_ollama() -> bool` — Check if Ollama is running.
- `generate_with_ollama(prompt:str, system:str) -> str` — Generate text using Ollama.
- `find_issues(project) -> list` — Find potential code issues.
- `get_llm_suggestions(issues:list, project) -> str` — detailed suggestions from LLM.
- `main()` — Main refactoring analysis.

---

#### `token_efficiency.py`

```yaml
path: examples/token_efficiency.py
lang: python | lines: 168/240
imports: [sys]
```

> Example: Token Efficiency Analysis for LLM Cost Optimization.

Analyzes and compares token usage acr

**Functions:**

- `count_tokens(text:str) -> int` — Approximate token count (~4 chars per token for En
- `estimate_cost(tokens:int, model:str) -> float` — Estimate API cost based on token count.
- `analyze_format(name:str, content:str, baseline_tokens:int) -> Dict[str, Any]` — Analyze a single format's efficiency.
- `main()` — Run token efficiency analysis.

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
lang: python | lines: 233/288
imports: [pytest, tempfile]
```

> Pytest configuration and fixtures for code2logic tests.

**Functions:**

- `sample_python_code() -> str` — Sample Python code for testing.
- `sample_javascript_code() -> str` — Sample JavaScript code for testing.
- `sample_java_code() -> str` — Sample Java code for testing.
- `temp_project_dir()` — a temporary project directory.
- `sample_project(temp_project_dir, sample_python_code)` — a sample project with Python files.
- `sample_module()` — a sample module for testing.
- `sample_project_model()` — a sample project model for testing.
- `mock_llm_config()` — Mock LLM configuration for testing.
- `sample_analysis_result()` — Sample analysis result for testing.

---

#### `test_analyzer.py`

```yaml
path: tests/test_analyzer.py
lang: python | lines: 193/278
imports: [pytest]
```

> Tests for the ProjectAnalyzer class.

**class `TestProjectAnalyzer`**

> Test cases for ProjectAnalyzer.

```yaml
methods:
  test_init(self, temp_project_dir)  # Test ProjectAnalyzer initialization.
  test_init_with_config(self, temp_project_dir)  # Test ProjectAnalyzer initialization with
  test_discover_source_files(self, sample_project)  # Test source file discovery.
  test_discover_source_files_filters_non_source(self, temp_project_dir)  # Test that non-source files are filtered 
  test_discover_source_files_ignores_common_dirs(self, temp_project_dir)  # Test that common directories are ignored
  test_parse_file_tree_sitter_success(self, mock_fallback, mock_tree_sitter, sample_project)  # Test successful parsing with Tree-sitter
  test_parse_file_fallback_success(self, mock_fallback, mock_tree_sitter, sample_project)  # Test fallback parsing when Tree-sitter f
  test_parse_file_both_fail(self, mock_fallback, mock_tree_sitter, sample_project)  # Test when both parsers fail.
  test_extract_metadata(self, sample_project)  # Test metadata extraction.
  test_analyze_complete(self, mock_similarity, mock_dependency, sample_project)  # Test complete analysis process.
  test_analyze_without_files(self, temp_project_dir)  # Test analysis with no source files.
  test_generate_output(self, mock_generator_class, sample_project_model)  # Test output generation.
  # ... +3 more
```

---

#### `test_generators.py`

```yaml
path: tests/test_generators.py
lang: python | lines: 207/260
imports: [json, pytest]
```

> Tests for output generators.

**class `TestMarkdownGenerator`**

> Tests for MarkdownGenerator.

```yaml
methods:
  test_generate_basic(self, sample_project)  # Test basic Markdown generation.
  t_generate_includes_modules(se(f, s, ple_project):
)  # Test that modules are included."""
  t_generate_includes_classes(se(f, s, ple_project):
)  # Test that classes are included."""
  t_generate_includes_functions(se(f, s, ple_project):
)  # Test that functions are included."""
  t_generate_includes_entrypoints(se(f, s, ple_project):
)  # Test that entrypoints are included."""
  t_detail_levels(se(f, s, ple_project):
)  # Test different detail levels."""
```

**class `tCompactGenerator:
 `**

> Tests for CompactGenerator."""

 

```yaml
methods:
  t_generate_basic(se(f, s, ple_project):
)  # Test basic compact generation."""
  t_generate_includes_hubs(se(f, s, ple_project):
)  # Test that ENTRY is included."""
  t_compact_is_smaller(se(f, s, ple_project):
)  # Test that compact output is smaller than
```

**class `tJSONGenerator:
 `**

> Tests for JSONGenerator."""

 

```yaml
methods:
  t_generate_valid_json(se(f, s, ple_project):
)  # Test that output is valid JSON."""
  t_generate_structure(se(f, s, ple_project):
)  # Test JSON structure."""
  t_generate_modules(se(f, s, ple_project):
)  # Test module structure in JSON."""
  t_generate_functions(se(f, s, ple_project):
)  # Test function structure in JSON."""
  t_generate_classes(se(f, s, ple_project):
)  # Test class structure in JSON."""
```

**Functions:**

- `sample_project()` — a sample project for testing generators.

---

#### `test_intent.py`

```yaml
path: tests/test_intent.py
lang: python | lines: 356/496
imports: [pytest]
```

> Tests for intent analysis functionality.

**class `TestIntentAnalyzer`**

> Test cases for IntentAnalyzer.

```yaml
methods:
  test_init(self)  # Test IntentAnalyzer initialization.
  test_extract_keywords(self)  # Test keyword extraction from queries.
  test_calculate_intent_confidence(self)  # Test intent confidence calculation.
  test_identify_target_module(self, sample_project_model)  # Test target identification for modules.
  test_identify_target_function(self, sample_project_model)  # Test target identification for functions
  test_identify_target_class(self, sample_project_model)  # Test target identification for classes.
  test_identify_target_project(self, sample_project_model)  # Test target identification for project-l
  test_generate_description(self)  # Test description generation for intents.
  test_generate_suggestions_refactor(self)  # Test suggestion generation for refactor 
  test_generate_suggestions_analyze(self)  # Test suggestion generation for analyze i
  test_generate_suggestions_optimize(self)  # Test suggestion generation for optimize 
  test_analyze_intent_refactor(self, sample_project_model)  # Test intent analysis for refactoring.
  # ... +18 more
```

---
