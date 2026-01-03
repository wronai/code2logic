.PHONY: help install install-dev install-full clean build test lint format typecheck publish publish-test docs docker

PYTHON := python3
PIP := pip

# Colors for terminal output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Code2Logic - Build Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ============================================================================
# Installation
# ============================================================================

install: ## Install package (minimal)
	$(PIP) install -e .

install-dev: ## Install with development dependencies
	$(PIP) install -e ".[dev]"

install-full: ## Install with all features
	$(PIP) install -e ".[full,dev]"

install-docs: ## Install documentation dependencies
	$(PIP) install -e ".[docs]"

install-llm: ## Install LLM integration dependencies
	$(PIP) install httpx litellm

# ============================================================================
# Testing
# ============================================================================

test: ## Run tests
	pytest tests/ -v -p no:aiohttp

test-cov: ## Run tests with coverage
	pytest tests/ -v -p no:aiohttp --cov=code2logic --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage (faster)
	pytest tests/ -v -p no:aiohttp --no-cov

test-all: ## Run all tests including integration
	pytest tests/ -v -p no:aiohttp --cov=code2logic
	@echo "$(GREEN)All tests passed!$(NC)"

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run linters
	ruff check code2logic tests
	
lint-fix: ## Run linters and fix issues
	ruff check code2logic tests --fix

format: ## Format code with black
	black code2logic tests

format-check: ## Check code formatting
	black code2logic tests --check

typecheck: ## Run type checking
	mypy code2logic --ignore-missing-imports

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
	$(PYTHON) -m build
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

publish: build ## Publish to PyPI (production)
	@echo "$(YELLOW)Publishing to PyPI...$(NC)"
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
	ollama pull qwen2.5:1.5b
	@echo "$(GREEN)Models pulled!$(NC)"

mcp-server: ## Start MCP server for Claude Desktop
	$(PYTHON) -m code2logic.mcp_server

# ============================================================================
# Release
# ============================================================================

version: ## Show current version
	@$(PYTHON) -c "from code2logic import __version__; print(__version__)"

check-release: clean test lint build ## Full release check
	@echo "$(GREEN)Release check passed!$(NC)"
	@echo "Ready to publish with: make publish"
