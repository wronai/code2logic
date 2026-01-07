.PHONY: help install install-dev install-full clean build test lint format typecheck publish publish-test docs docker

POETRY := $(shell command -v poetry 2>/dev/null)
ifeq ($(POETRY),)
RUN :=
PYTHON := python3
PIP := pip
else
RUN := poetry run
PYTHON := $(RUN) python
PIP :=
endif

# Colors for terminal output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

# Sub-packages
SUBPACKAGES := lolm logic2test logic2code

help: ## Show this help message
	@echo "$(BLUE)Code2Logic - Build Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ============================================================================
# Installation
# ============================================================================

install: ## Install package (minimal)
	@if [ -n "$(POETRY)" ]; then \
		poetry install; \
	else \
		$(PIP) install -e .; \
	fi

install-dev: ## Install with development dependencies
	@if [ -n "$(POETRY)" ]; then \
		poetry install --with dev; \
	else \
		$(PIP) install -e ".[dev]"; \
	fi

install-full: ## Install with all features
	@if [ -n "$(POETRY)" ]; then \
		poetry install --with dev -E full; \
	else \
		$(PIP) install -e ".[full,dev]"; \
	fi

install-docs: ## Install documentation dependencies
	@if [ -n "$(POETRY)" ]; then \
		poetry install --with docs; \
	else \
		$(PIP) install -e ".[docs]"; \
	fi

install-llm: ## Install LLM integration dependencies
	@if [ -n "$(POETRY)" ]; then \
		poetry install -E llm; \
	else \
		$(PIP) install httpx litellm python-dotenv; \
	fi

# ============================================================================
# Configuration
# ============================================================================

config: ## Show current configuration
	@$(PYTHON) -c "from code2logic.config import Config; import json; print(json.dumps(Config().to_dict(), indent=2))"

config-env: ## Show shell commands to configure API keys
	@echo "$(BLUE)API Configuration Commands:$(NC)"
	@echo ""
	@echo "# OpenRouter (cloud LLM)"
	@echo 'export OPENROUTER_API_KEY="sk-or-v1-your-key"'
	@echo 'export OPENROUTER_MODEL="qwen/qwen-2.5-coder-32b-instruct"'
	@echo ""
	@echo "# Ollama (local LLM)"
	@echo 'export OLLAMA_HOST="http://localhost:11434"'
	@echo 'export OLLAMA_MODEL="qwen2.5-coder:14b"'
	@echo ""
	@echo "# Or create .env file:"
	@echo "cp .env.example .env"
	@echo "# Then edit .env with your keys"

config-check: ## Check which providers are configured
	@echo "$(BLUE)Provider Status:$(NC)"
	@$(PYTHON) -c "from code2logic.config import Config; c=Config(); [print(f'  {k}: ✓' if v else f'  {k}: ✗') for k,v in c.list_configured_providers().items()]"

# ============================================================================
# Testing
# ============================================================================

test: ## Run tests
	$(RUN) pytest tests/ -v -p no:aiohttp

test-cov: ## Run tests with coverage
	$(RUN) pytest tests/ -v -p no:aiohttp --cov=code2logic --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage (faster)
	$(RUN) pytest tests/ -v -p no:aiohttp --no-cov

test-all: ## Run all tests including subpackages
	@echo "$(BLUE)Running code2logic tests...$(NC)"
	$(RUN) pytest tests/ -v -p no:aiohttp --cov=code2logic
	@echo "$(BLUE)Running subpackage tests...$(NC)"
	@for pkg in $(SUBPACKAGES); do \
		echo "$(YELLOW)Testing $$pkg...$(NC)"; \
		(cd $$pkg && $(MAKE) test); \
	done
	@echo "$(GREEN)All tests passed!$(NC)"

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run linters
	$(RUN) ruff check code2logic tests
	
lint-fix: ## Run linters and fix issues
	$(RUN) ruff check code2logic tests --fix

format: ## Format code with black
	$(RUN) black code2logic tests

format-check: ## Check code formatting
	$(RUN) black code2logic tests --check

typecheck: ## Run type checking
	$(RUN) mypy code2logic --ignore-missing-imports

quality: lint format-check typecheck ## Run all quality checks

# ============================================================================
# Building
# ============================================================================

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf output/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	@if [ -n "$(POETRY)" ]; then \
		poetry build; \
	else \
		$(PYTHON) -m build; \
	fi
	@echo "$(GREEN)Build complete!$(NC)"
	@ls -lh dist/

# ============================================================================
# Publishing
# ============================================================================

publish-test: build ## Publish to TestPyPI
	@echo "$(YELLOW)Publishing to TestPyPI...$(NC)"
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)Published to TestPyPI!$(NC)"
	@echo "Install with: pip install -i https://test.pypi.org/simple/ code2logic"


git-clean: ## Ensure git working tree is clean
	@if [ "$(FORCE)" = "1" ]; then \
		echo "FORCE=1 set: skipping git clean check."; \
		exit 0; \
	fi
	@git diff --quiet || (echo "Working tree has unstaged changes. Commit or stash before publishing."; exit 1)
	@git diff --cached --quiet || (echo "Index has staged but uncommitted changes. Commit before publishing."; exit 1)

check-bumpver: ## Ensure bumpver is installed
	@$(PYTHON) -c "import bumpver" >/dev/null 2>&1 || ( \
		echo "Missing bumpver. Installing bumpver..."; \
		$(PYTHON) -m pip install "bumpver>=2023.1129" >/dev/null; \
		$(PYTHON) -c "import bumpver" >/dev/null 2>&1 || ( \
			echo "bumpver still missing. Installing project dev dependencies..."; \
			$(PIP) install -e \".[dev]\"; \
			$(PYTHON) -c "import bumpver" >/dev/null 2>&1 || (echo "Failed to install bumpver."; exit 1); \
		); \
	)


bump-patch: check-bumpver ## Bump patch version (updates pyproject.toml and code2logic/__init__.py)
	$(PYTHON) -m bumpver update --patch


bump-minor: check-bumpver ## Bump minor version
	$(PYTHON) -m bumpver update --minor


bump-major: check-bumpver ## Bump major version
	$(PYTHON) -m bumpver update --major

publish: bump-patch build ## Publish to PyPI (production)
	@echo "$(YELLOW)Publishing to PyPI...$(NC)"
	$(PYTHON) -m twine upload dist/*
	@echo "$(GREEN)Published to PyPI!$(NC)"
	@echo "Install with: pip install code2logic"

publish-dirty: bump-patch build ## Publish to PyPI without git-clean (dangerous)
	@echo "$(YELLOW)Publishing to PyPI (skipping git clean check)...$(NC)"
	$(PYTHON) -m twine upload dist/*
	@echo "$(GREEN)Published to PyPI!$(NC)"
	@echo "Install with: pip install code2logic"

# ============================================================================
# Docker
# ============================================================================

docker-build: ## Build Docker image
	docker build -t code2logic:latest .
	@echo "$(GREEN)Docker image built: code2logic:latest$(NC)"

docker-build-dev: ## Build development Docker image
	docker build -f Dockerfile.dev -t code2logic:dev .
	@echo "$(GREEN)Docker dev image built: code2logic:dev$(NC)"

docker-run: ## Run code2logic in Docker (usage: make docker-run PATH=/project)
	docker run -v $(PATH):/project -v $(PWD)/output:/output code2logic:latest /project -f csv -o /output/analysis.csv

docker-shell: ## Open shell in development container
	docker run -it -v $(PWD):/app code2logic:dev bash

docker-compose-up: ## Start all services (code2logic + ollama + litellm)
	docker-compose up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo "  Ollama: http://localhost:11434"
	@echo "  LiteLLM: http://localhost:4000"

docker-compose-down: ## Stop all services
	docker-compose down

# ============================================================================
# Documentation
# ============================================================================

docs: ## Build documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

# ============================================================================
# Development
# ============================================================================

dev-setup: ## Setup development environment
	$(PIP) install -e ".[full,dev,docs]"
	$(PIP) install httpx litellm
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

# ============================================================================
# Examples & Analysis
# ============================================================================

run: ## Run code2logic on current directory
	code2logic . -f csv -d standard

run-all-formats: ## Generate all formats for current directory
	@mkdir -p output
	code2logic . -f csv -d minimal -o output/analysis_minimal.csv
	code2logic . -f csv -d standard -o output/analysis_standard.csv
	code2logic . -f csv -d full -o output/analysis_full.csv
	code2logic . -f json -d standard -o output/analysis.json
	code2logic . -f json --flat -d standard -o output/analysis_flat.json
	code2logic . -f yaml -d standard -o output/analysis.yaml
	code2logic . -f yaml --flat -d standard -o output/analysis_flat.yaml
	code2logic . -f compact -o output/analysis_compact.txt
	code2logic . -f markdown -d standard -o output/analysis.md
	@echo "$(GREEN)All formats generated in output/$(NC)"
	@ls -la output/

run-compare: ## Compare sizes of all formats
	@echo "$(BLUE)Format comparison:$(NC)"
	@echo -n "CSV minimal:  " && code2logic . -f csv -d minimal 2>/dev/null | wc -c
	@echo -n "CSV standard: " && code2logic . -f csv -d standard 2>/dev/null | wc -c  
	@echo -n "CSV full:     " && code2logic . -f csv -d full 2>/dev/null | wc -c
	@echo -n "JSON nested:  " && code2logic . -f json 2>/dev/null | wc -c
	@echo -n "JSON flat:    " && code2logic . -f json --flat 2>/dev/null | wc -c
	@echo -n "YAML nested:  " && code2logic . -f yaml 2>/dev/null | wc -c
	@echo -n "YAML flat:    " && code2logic . -f yaml --flat 2>/dev/null | wc -c
	@echo -n "Compact:      " && code2logic . -f compact 2>/dev/null | wc -c
	@echo -n "Markdown:     " && code2logic . -f markdown 2>/dev/null | wc -c

status: ## Show library status
	code2logic --status

# ============================================================================
# LLM Integration
# ============================================================================

ollama-start: ## Start Ollama server
	@echo "$(YELLOW)Starting Ollama...$(NC)"
	ollama serve &
	@sleep 2
	@echo "$(GREEN)Ollama started at http://localhost:11434$(NC)"

ollama-pull: ## Pull recommended models for code analysis
	ollama pull qwen2.5-coder:7b
	ollama pull qwen2.5-coder:14b
	ollama pull deepseek-coder:6.7b
	@echo "$(GREEN)Models pulled!$(NC)"

ollama-list: ## List available Ollama models
	@echo "$(BLUE)Available Ollama models:$(NC)"
	@ollama list 2>/dev/null || echo "Ollama not running"

mcp-server: ## Start MCP server for Claude Desktop
	$(PYTHON) -m code2logic.mcp_server

# ============================================================================
# LLM Configuration
# ============================================================================

llm: ## Configure LLM providers (Ollama, LiteLLM)
	@echo "$(BLUE)Configuring LLM providers...$(NC)"
	$(PYTHON) scripts/configure_llm.py

llm-list: ## List all available LLM models
	$(PYTHON) scripts/configure_llm.py --list

llm-test: ## Test configured LLM models
	$(PYTHON) scripts/configure_llm.py --test

llm-status: ## Show LLM configuration status
	@echo "$(BLUE)LLM Status:$(NC)"
	@echo "Ollama:"
	@curl -s http://localhost:11434/api/version 2>/dev/null && echo " ✓ Running" || echo " ✗ Not running"
	@echo "\nModels:"
	@ollama list 2>/dev/null | head -10 || echo "  None available"
	@echo "\nConfig:"
	@cat ~/.code2logic/llm_config.json 2>/dev/null | head -20 || echo "  Not configured (run: make llm)"

# ============================================================================
# Sub-packages (lolm, logic2test, logic2code)
# ============================================================================

publish-lolm: ## Publish lolm package to PyPI
	@echo "$(YELLOW)Publishing lolm to PyPI...$(NC)"
	cd lolm && $(MAKE) build && $(MAKE) publish
	@echo "$(GREEN)lolm published!$(NC)"

publish-logic2test: ## Publish logic2test package to PyPI
	@echo "$(YELLOW)Publishing logic2test to PyPI...$(NC)"
	cd logic2test && $(MAKE) build && $(MAKE) publish
	@echo "$(GREEN)logic2test published!$(NC)"

publish-logic2code: ## Publish logic2code package to PyPI
	@echo "$(YELLOW)Publishing logic2code to PyPI...$(NC)"
	cd logic2code && $(MAKE) build && $(MAKE) publish
	@echo "$(GREEN)logic2code published!$(NC)"

publish-all: ## Publish all packages (code2logic + sub-packages)
	@echo "$(BLUE)Publishing all packages...$(NC)"
	@for pkg in $(SUBPACKAGES); do \
		echo "$(YELLOW)Building and publishing $$pkg...$(NC)"; \
		cd $$pkg && $(MAKE) build && $(MAKE) publish && cd ..; \
	done
	@echo "$(YELLOW)Publishing code2logic...$(NC)"
	$(MAKE) publish
	@echo "$(GREEN)All packages published!$(NC)"

publish-all-test: ## Publish all packages to TestPyPI
	@echo "$(BLUE)Publishing all packages to TestPyPI...$(NC)"
	@for pkg in $(SUBPACKAGES); do \
		echo "$(YELLOW)Building and publishing $$pkg to TestPyPI...$(NC)"; \
		cd $$pkg && $(MAKE) build && $(MAKE) publish-test && cd ..; \
	done
	$(MAKE) publish-test
	@echo "$(GREEN)All packages published to TestPyPI!$(NC)"

build-subpackages: ## Build all sub-packages
	@echo "$(BLUE)Building sub-packages...$(NC)"
	@for pkg in $(SUBPACKAGES); do \
		echo "$(YELLOW)Building $$pkg...$(NC)"; \
		cd $$pkg && $(MAKE) build && cd ..; \
	done
	@echo "$(GREEN)All sub-packages built!$(NC)"

test-subpackages: ## Run tests for all sub-packages
	@echo "$(BLUE)Testing sub-packages...$(NC)"
	@for pkg in $(SUBPACKAGES); do \
		echo "$(YELLOW)Testing $$pkg...$(NC)"; \
		cd $$pkg && $(MAKE) test && cd ..; \
	done
	@echo "$(GREEN)All sub-package tests passed!$(NC)"

lint-subpackages: ## Lint all sub-packages
	@echo "$(BLUE)Linting sub-packages...$(NC)"
	@for pkg in $(SUBPACKAGES); do \
		echo "$(YELLOW)Linting $$pkg...$(NC)"; \
		cd $$pkg && $(MAKE) lint && cd .. || true; \
	done

clean-subpackages: ## Clean all sub-packages
	@for pkg in $(SUBPACKAGES); do \
		cd $$pkg && $(MAKE) clean && cd ..; \
	done

install-subpackages: ## Install all sub-packages in dev mode
	@for pkg in $(SUBPACKAGES); do \
		echo "$(YELLOW)Installing $$pkg...$(NC)"; \
		cd $$pkg && $(MAKE) install-dev && cd ..; \
	done
	@echo "$(GREEN)All sub-packages installed!$(NC)"

# ============================================================================
# Release
# ============================================================================

version: ## Show current version
	@$(PYTHON) -c "from code2logic import __version__; print(__version__)"

version-all: ## Show versions of all packages
	@echo "$(BLUE)Package versions:$(NC)"
	@echo -n "  code2logic: " && $(PYTHON) -c "from code2logic import __version__; print(__version__)"
	@echo -n "  lolm: " && $(PYTHON) -c "from lolm import __version__; print(__version__)" 2>/dev/null || echo "not installed"
	@echo -n "  logic2test: " && $(PYTHON) -c "from logic2test import __version__; print(__version__)" 2>/dev/null || echo "not installed"
	@echo -n "  logic2code: " && $(PYTHON) -c "from logic2code import __version__; print(__version__)" 2>/dev/null || echo "not installed"

check-release: clean test lint build ## Full release check
	@echo "$(GREEN)Release check passed!$(NC)"
	@echo "Ready to publish with: make publish"

check-release-all: clean test lint build test-subpackages build-subpackages ## Full release check for all packages
	@echo "$(GREEN)Release check for all packages passed!$(NC)"
	@echo "Ready to publish all with: make publish-all"
