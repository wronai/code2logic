# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) + [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.43] - 2026-02-25

### Added
- **`--function-logic-context`** CLI flag (`none`/`minimal`/`full`) — adds class/module context headers to function.toon output
- **`--hybrid`** flag for TOON format — combines project structure with function-logic details for hub modules
- **LogicML `level` parameter** (`compact`/`typed`/`full`) — controls signature richness and type preservation
- **AST-based structural scoring** in `metrics.py` — uses Python `ast` module instead of regex, with regex fallback
- **`failure_rate` metric** in `BenchmarkResult` — tracks percentage of files scoring 0%
- **`generate_hybrid()`** method in `TOONGenerator` — project TOON + selective function details for top-N hub modules

### Changed
- **Benchmark aggregation** now includes ALL scores (including zeros) instead of filtering `score > 0`
- **`_structural_score`** in benchmark runner uses ratio-based scoring (`min/max`) instead of binary exact-match
- **`_extract_code`** supports 12+ language-specific code block markers (js, ts, go, rust, java, etc.)
- **Reproduction prompts** rewritten with detailed parsing instructions per format (gherkin, function.toon, csv, markdown, logicml)
- **Spec truncation limit** increased from 8000 to 12000 chars
- **Function benchmark** prompt enriched with calls/raises/complexity info, `max_tokens` increased 2000→3000
- **LogicML default `level`** changed from `compact` to `typed` (10 params with full types)
- **`BenchmarkResult.load()`** properly reconstructs `FormatResult` objects from JSON

### Fixed
- Project benchmark merge-score: `FileResult.score` recalculated as average across all format_results after merge
- Format/project benchmark score loops now include zero scores in per-format averages
- Function benchmark similarity calculation includes all functions (not just non-zero)

### Removed
- Dead code: `llm_clients_new.py` stub (unused)
- Stale generated artifacts from root directory

## [1.0.34] - 2026-02-24

### Changed
- CLI improvements and config management updates
- Generator interface unification (`.generate()` signatures standardized)

## [1.0.33] - 2026-02-23

### Added
- Deep code analysis engine with 6 supporting modules
- Updated pyproject.toml build configuration

## [1.0.31] - 2026-02-15

### Added
- Goal-driven analysis engine with 7 supporting modules
- `code2logic llm` command group for LLM provider/model/key management
- Provider auto-selection with priority modes (`provider-first`, `model-first`, `mixed`)

## [1.0.1] - 2026-01-03

### Added
- Initial release of code2logic
- Multi-language code analysis (Python, JavaScript, Java, C/C++)
- Tree-sitter AST parsing with fallback parsers
- NetworkX dependency graph analysis (PageRank, hub detection)
- Code similarity detection (Rapidfuzz)
- LLM integration (Ollama, LiteLLM, OpenRouter)
- Output formats: JSON, YAML, CSV, Markdown, Compact, TOON, LogicML, Gherkin
- MCP server for Claude Desktop integration
- CLI with auto-dependency installation
- Docker support
- 286 tests

## [1.0.0] - 2024-01-01

### Added
- Core project analysis engine
- Multi-language parsing, dependency graphs, similarity detection
- LLM integration framework
- CLI interface and Docker deployment
