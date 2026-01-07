# Changelog

All notable changes to the **logic2test** package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-07

### Added

- Initial release of logic2test
- Generate test scaffolds from Code2Logic output files
- Support for multiple input formats:
  - YAML (`.yaml`)
  - Hybrid YAML (`.hybrid.yaml`)
  - TOON (`.toon`)
- Test types:
  - Unit tests - Individual class/function tests
  - Integration tests - Cross-module interaction tests
  - Property tests - Hypothesis-based property testing
- `TestGenerator` class for programmatic test generation
- `LogicParser` class for parsing Code2Logic output
- `TestTemplate` class for customizable test templates
- CLI interface:
  - `logic2test FILE -o OUTPUT` - Generate tests
  - `logic2test FILE --summary` - Show testable items
  - `logic2test FILE --type TYPE` - Generate specific test type
  - `--include-private` - Include private methods
  - `--include-dunder` - Include dunder methods
- Configuration options:
  - Framework selection (pytest, unittest)
  - Max tests per file
  - Test prefix customization
- Smart mocking based on type hints
- pytest-style test generation
- Full type hints and documentation

### Dependencies

- `pyyaml>=6.0` - YAML parsing
- Optional: `hypothesis>=6.0.0` - Property testing

[Unreleased]: https://github.com/wronai/code2logic/compare/logic2test-v0.1.0...HEAD
[0.1.0]: https://github.com/wronai/code2logic/releases/tag/logic2test-v0.1.0
