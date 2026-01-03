# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026-01-03

### Changed
- Improved benchmark robustness and artifact hygiene in example scripts.
- Added packaging tools to development extras.

### Fixed
- Cleaned and hardened benchmark output generation (atomic writes, cleanup of generated artifacts).
- Improved error reporting in benchmarking and function reproduction examples.

### Added
- Initial release of code2logic
- Multi-language code analysis support (Python, JavaScript, Java, C/C++)
- Tree-sitter based AST parsing with fallback parsers
- NetworkX dependency graph analysis
- Code similarity detection algorithms
- LLM integration with Ollama and LiteLLM
- Intent analysis from natural language queries
- Multiple output formats (JSON, YAML, CSV, Markdown, Compact)
- MCP server for Claude Desktop integration
- CLI with auto-dependency installation
- Docker support with multi-service orchestration
- Comprehensive test suite with 46 tests
- Example scripts and documentation

### Changed
- N/A (initial release)

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [1.0.0] - 2024-01-XX

### Added
- Core project analysis engine
- Multi-language parsing support
- Dependency graph analysis
- Code similarity detection
- LLM integration framework
- Intent analysis system
- Multiple output generators
- MCP server implementation
- CLI interface
- Docker deployment support
- Comprehensive documentation
- Full test coverage

---

## Version History

### Development Phase
- **v0.1.0** - Initial project structure and basic parsing
- **v0.2.0** - Added dependency analysis and similarity detection
- **v0.3.0** - Integrated LLM support and intent analysis
- **v0.4.0** - Added MCP server and CLI improvements
- **v0.5.0** - Enhanced output formats and Docker support
- **v1.0.0** - Production-ready release with full feature set

### Key Milestones
- ✅ Multi-language AST parsing
- ✅ Dependency graph analysis
- ✅ Code similarity detection
- ✅ LLM integration (Ollama/LiteLLM)
- ✅ Intent analysis
- ✅ Multiple output formats
- ✅ MCP server for Claude Desktop
- ✅ CLI with auto-installation
- ✅ Docker deployment
- ✅ Comprehensive testing
- ✅ Documentation and examples

---

## Breaking Changes

### v1.0.0
No breaking changes - this is the initial stable release.

---

## Dependencies

### Core Dependencies
- `networkx>=3.0` - Graph analysis
- `pyyaml>=6.0` - YAML support
- `tree-sitter>=0.20.0` - AST parsing
- `litellm>=1.0.0` - LLM integration
- `click>=8.0.0` - CLI framework
- `rich>=13.0.0` - Terminal output

### Optional Dependencies
- `mcp>=1.0.0` - MCP server support
- `pytest>=7.0.0` - Testing framework
- `black>=23.0.0` - Code formatting
- `mypy>=1.0.0` - Type checking

---

## Platform Support

### Supported Python Versions
- Python 3.8+
- Tested on 3.8, 3.9, 3.10, 3.11, 3.12

### Supported Operating Systems
- Linux (primary development platform)
- macOS
- Windows (limited testing)

### Container Support
- Docker Hub images available
- Multi-architecture support (amd64, arm64)

---

## Performance Notes

### Large Project Analysis
- Tested on projects up to 1000+ files
- Memory usage scales with project size
- LLM processing may be rate-limited by provider

### Optimization Tips
- Use `--no-llm` flag for faster analysis without AI features
- Consider using compact format for large projects
- Enable parallel processing for multiple projects

---

## Known Issues

### Current Limitations
- Tree-sitter parsers require language-specific grammars
- LLM integration requires external services (Ollama/LiteLLM)
- Large projects may exceed LLM context limits
- Some language features may not be fully supported

### Planned Fixes
- Improved error handling for malformed code
- Better memory management for large projects
- Enhanced LLM prompt optimization
- Additional language support

---

## Security Considerations

### LLM Integration
- Code is sent to external LLM services
- Consider using local Ollama for sensitive code
- Review LLM provider privacy policies

### File System Access
- Tool reads project files for analysis
- No write operations except output generation
- Respects .gitignore and common exclude patterns

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to code2logic.

### Development Setup
```bash
git clone https://github.com/wronai/code2logic.git
cd code2logic
pip install -e .[dev]
pre-commit install
```

### Running Tests
```bash
pytest --cov=code2logic
```

---

## Support

- **Documentation**: https://code2logic.readthedocs.io
- **Issues**: https://github.com/wronai/code2logic/issues
- **Discussions**: https://github.com/wronai/code2logic/discussions
- **Email**: team@code2logic.dev

---

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file for details.
