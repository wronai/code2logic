# Refactoring Report: code2logic/

## Summary
- **Files:** 53
- **Functions:** 646
- **Duplicates:** 23
- **Quality issues:** 151
- **Security issues:** 26

## Duplicates

### c0e8002f
**Suggestion:** Extract to shared utility function

- `llm_profiler.py::LLMProfile.__post_init__`
- `benchmarks/results.py::BenchmarkResult.__post_init__`

### 30ac9c08
**Suggestion:** Extract to shared utility function

- `llm_profiler.py::LLMProfiler._extract_code`
- `chunked_reproduction.py::ChunkedReproducer._extract_code`

### 72f8dd24
**Suggestion:** Extract to shared utility function

- `config.py::Config.to_dict`
- `errors.py::AnalysisError.to_dict`
- `quality.py::QualityReport.to_dict`
- `reproducer.py::FileValidation.to_dict`
- `metrics.py::ReproductionResult.to_dict`

### 813543f7
**Suggestion:** Extract to shared utility function

- `project_reproducer.py::ProjectReproducer._get_client`
- `universal.py::UniversalReproducer._get_client`
- `benchmarks/runner.py::BenchmarkRunner._get_client`

### 5d30077a
**Suggestion:** Extract to shared utility function

- `base.py::VerboseMixin.__init__`
- `base.py::BaseParser.__init__`
- `base.py::BaseGenerator.__init__`
- `reproducer.py::SpecReproducer.__init__`
- `metrics.py::ReproductionMetrics.__init__`

### 20345973
**Suggestion:** Extract to shared utility function

- `base.py::VerboseMixin.info`
- `cli.py::Logger.info`

### 7f5a82eb
**Suggestion:** Extract to shared utility function

- `base.py::VerboseMixin.error`
- `cli.py::Logger.error`

### 9ec0e68a
**Suggestion:** Extract to shared utility function

- `cli.py::_get_user_llm_config_path`
- `llm_clients.py::_get_user_llm_config_path`

### 2979010b
**Suggestion:** Extract to shared utility function

- `cli.py::_load_user_llm_config`
- `llm_clients.py::_load_user_llm_config`

### 1e2a2e55
**Suggestion:** Extract to shared utility function

- `errors.py::AnalysisResult.summary`
- `reproducer.py::ReproductionResult.summary`

## Quality Issues

- 🟡 **high_complexity** at `llm_profiler.py::_create_default_profile`
  - high_complexity: 11
- 🔴 **high_complexity** at `file_formats.py::generate_file_yaml`
  - high_complexity: 22
- 🔴 **high_complexity** at `file_formats.py::_parse_file_elements`
  - high_complexity: 45
- 🔴 **high_complexity** at `project_reproducer.py::ProjectReproducer._aggregate_results`
  - high_complexity: 17
- 🔴 **high_complexity** at `cli.py::_code2logic_llm_cli`
  - high_complexity: 34
- 🔴 **high_complexity** at `cli.py::main`
  - high_complexity: 93
- 🟡 **high_complexity** at `llm.py::CodeAnalyzer.find_semantic_duplicates`
  - high_complexity: 11
- 🟡 **high_complexity** at `llm.py::CodeAnalyzer.generate_code`
  - high_complexity: 11
- 🟡 **high_complexity** at `errors.py::ErrorHandler.safe_read_file`
  - high_complexity: 14
- 🟡 **high_complexity** at `code_review.py::analyze_code_quality`
  - high_complexity: 15

## Security Issues

- 🔒 **hardcoded_secrets** at `llm_profiler.py::`
- 🔒 **hardcoded_secrets** at `config.py::get_api_key`
- 🔒 **hardcoded_secrets** at `cli.py::_code2logic_llm_cli`
- 🔒 **hardcoded_secrets** at `chunked_reproduction.py::`
- 🔒 **hardcoded_secrets** at `chunked_reproduction.py::chunk_yaml_spec`
- 🔒 **hardcoded_secrets** at `chunked_reproduction.py::chunk_gherkin_spec`
- 🔒 **hardcoded_secrets** at `chunked_reproduction.py::chunk_markdown_spec`
- 🔒 **hardcoded_secrets** at `benchmarks/common.py::_generate_token_json_compact`
- 🔒 **hardcoded_secrets** at `benchmarks/common.py::generate_spec_token`
- 🔒 **hardcoded_secrets** at `benchmarks/common.py::generate_spec_token`
