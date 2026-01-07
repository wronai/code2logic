# Changelog

All notable changes to the **lolm** package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-07

### Added

- Initial release of LOLM (Lightweight Orchestrated LLM Manager)
- Multi-provider support: OpenRouter, Ollama, Groq, Together, LiteLLM
- `LLMManager` class for managing multiple providers with fallback
- `get_client()` function for simple provider access
- CLI interface similar to `reclapp llm`:
  - `lolm status` - Show provider status
  - `lolm set-provider` - Set default provider
  - `lolm set-model` - Set model for provider
  - `lolm key set/show` - Manage API keys
  - `lolm models` - List recommended models
  - `lolm test` - Test LLM generation
  - `lolm config show` - Show configuration
  - `lolm priority set-provider/set-mode` - Manage priorities
- Configuration via:
  - Environment variables (`.env`)
  - `litellm_config.yaml`
  - User config (`~/.lolm/config.json`)
- Priority routing with three modes: `provider-first`, `model-first`, `mixed`
- Automatic provider fallback on failure
- Recommended models list for each provider
- Full type hints and documentation

### Dependencies

- `httpx>=0.24.0` - HTTP client
- `pyyaml>=6.0` - YAML configuration
- Optional: `litellm>=1.0.0` - Universal LLM interface

[Unreleased]: https://github.com/wronai/code2logic/compare/lolm-v0.1.0...HEAD
[0.1.0]: https://github.com/wronai/code2logic/releases/tag/lolm-v0.1.0
