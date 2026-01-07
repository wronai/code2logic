# Changelog

All notable changes to the **logic2code** package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-07

### Added

- Initial release of logic2code
- Generate source code from Code2Logic output files
- Support for multiple input formats:
  - YAML (`.yaml`)
  - Hybrid YAML (`.hybrid.yaml`)
  - TOON (`.toon`)
- `CodeGenerator` class for programmatic code generation
- `PythonRenderer` class for Python code rendering
- LLM-enhanced generation via `lolm` integration
- CLI interface:
  - `logic2code FILE -o OUTPUT` - Generate code
  - `logic2code FILE --summary` - Show summary
  - `logic2code FILE --stubs-only` - Generate stubs only
  - `logic2code FILE --modules LIST` - Generate specific modules
  - `--no-docstrings` - Skip docstrings
  - `--no-type-hints` - Skip type hints
- Configuration options:
  - Language selection (python)
  - Stubs-only mode
  - Docstring and type hint control
  - Structure preservation
  - LLM provider selection
- Generated code features:
  - Full type annotations
  - Docstrings with Args/Returns
  - Dataclass support
  - Async function support
  - `__init__.py` generation
- Full type hints and documentation

### Dependencies

- `pyyaml>=6.0` - YAML parsing
- `logic2test>=0.1.0` - Shared parsers
- Optional: `lolm>=0.1.0` - LLM-enhanced generation

[Unreleased]: https://github.com/wronai/code2logic/compare/logic2code-v0.1.0...HEAD
[0.1.0]: https://github.com/wronai/code2logic/releases/tag/logic2code-v0.1.0
