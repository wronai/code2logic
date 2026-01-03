# Makefile for code2logic
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev test lint format clean build docker docker-dev docker-compose up down logs shell

# Default target
help:
	@echo "code2logic - Convert codebase structure to logical representations"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "  install      Install code2logic in production mode"
	@echo "  install-dev  Install code2logic in development mode"
	@echo "  test         Run tests with coverage"
	@echo "  lint         Run code linting and formatting checks"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Clean build artifacts and cache"
	@echo "  build        Build distribution packages"
	@echo "  docker       Build Docker image"
	@echo "  docker-dev   Build development Docker image"
	@echo "  docker-compose Run docker-compose services"
	@echo "  up           Start all services with docker-compose"
	@echo "  down         Stop all services with docker-compose"
	@echo "  logs         Show logs from running services"
	@echo "  shell        Open shell in development container"
	@echo "  examples     Run example scripts"
	@echo "  docs         Generate documentation"
	@echo "  release     Create a new release"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .[dev,all]
	pre-commit install

# Testing
test:
	pytest --cov=code2logic --cov-report=html --cov-report=term --cov-report=xml

test-fast:
	pytest -m "not slow" --cov=code2logic --cov-report=term

test-unit:
	pytest -m unit --cov=code2logic --cov-report=term

test-integration:
	pytest -m integration --cov=code2logic --cov-report=term

test-llm:
	pytest -m llm --cov=code2logic --cov-report=term

# Code quality
lint:
	flake8 code2logic tests examples
	mypy code2logic
	bandit -r code2logic
	pydocstyle code2logic

format:
	black code2logic tests examples
	isort code2logic tests examples

format-check:
	black --check code2logic tests examples
	isort --check-only code2logic tests examples

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean
	rm -rf .venv/
	rm -rf venv/
	rm -rf env/

# Building
build:
	python -m build

build-wheel:
	python -m build --wheel

build-sdist:
	python -m build --sdist

# Docker
docker:
	docker build -t code2logic:latest .

docker-dev:
	docker build -f Dockerfile.dev -t code2logic:dev .

docker-compose:
	docker-compose build

# Docker Compose operations
up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

shell:
	docker-compose exec code2logic bash

shell-dev:
	docker-compose --profile dev up -d
	docker-compose exec code2logic-dev bash

# Development workflows
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation."

dev-test: format lint test
	@echo "Development tests completed!"

# Examples
examples:
	python examples/run_examples.sh --all

examples-basic:
	python examples/run_examples.sh --basic

examples-llm:
	python examples/run_examples.sh --llm

# Documentation
docs:
	@echo "Documentation is available in the docs/ directory"
	@echo "View FORMAT_ANALYSIS.md for LLM integration details"

docs-serve:
	@echo "Starting local documentation server..."
	@echo "Documentation available at http://localhost:8000"
	cd docs && python -m http.server 8000

# Release management
version:
	@python -c "import code2logic; print(code2logic.__version__)"

check-version:
	@echo "Checking version consistency..."
	@python -c "import code2logic; version = code2logic.__version__; print(f'Current version: {version}')"

tag:
	@echo "Creating git tag for version $(shell python -c 'import code2logic; print(code2logic.__version__)')"
	git tag -a v$(shell python -c 'import code2logic; print(code2logic.__version__)') -m "Release version $(shell python -c 'import code2logic; print(code2logic.__version__)')"

release: clean build test
	@echo "Release package ready in dist/"
	@echo "Run 'make tag' to create git tag"
	@echo "Then run 'twine upload dist/*' to publish"

# Quality gates
quality: format-check lint test
	@echo "All quality checks passed!"

ci: install-dev quality
	@echo "CI pipeline completed successfully!"

# Performance
benchmark:
	python -m pytest tests/test_performance.py -v

profile:
	python -m cProfile -o profile.stats examples/run_examples.sh --basic

# Security
security:
	bandit -r code2logic
	safety check

# Database (if using PostgreSQL)
db-init:
	docker-compose exec postgres psql -U code2logic -d code2logic -c "CREATE TABLE IF NOT EXISTS analysis_results (id SERIAL PRIMARY KEY, project_name VARCHAR(255), analysis_data JSONB, created_at TIMESTAMP DEFAULT NOW());"

db-migrate:
	@echo "Database migrations would be implemented here"

# Monitoring
monitoring-up:
	docker-compose --profile monitoring up -d

monitoring-down:
	docker-compose --profile monitoring down

# Backup and restore
backup:
	docker-compose exec postgres pg_dump -U code2logic code2logic > backup_$(shell date +%Y%m%d_%H%M%S).sql

restore:
	@echo "Usage: make restore BACKUP_FILE=<backup_file>"
	@echo "Example: make restore BACKUP_FILE=backup_20240101_120000.sql"

# Development helpers
watch:
	@echo "Watching for file changes..."
	@echo "Install watchdog: pip install watchdog"
	watchdog --patterns="*.py" --recursive --command="make test-fast" code2logic/

serve:
	python -m code2logic.mcp_server --mcp --mcp-port 8080

# Project information
info:
	@echo "code2logic Project Information"
	@echo "============================="
	@echo "Version: $(shell python -c 'import code2logic; print(code2logic.__version__)')"
	@echo "Python: $(shell python --version)"
	@echo "Pip: $(shell pip --version)"
	@echo "Git: $(shell git --version)"
	@echo "Docker: $(shell docker --version)"
	@echo "Docker Compose: $(shell docker-compose --version)"

# Quick start
quick-start: install-dev examples
	@echo "Quick start completed!"
	@echo "Check the examples_output/ directory for results"

# Advanced workflows
full-ci: clean install-dev quality security
	@echo "Full CI pipeline completed!"

production-build: clean build docker
	@echo "Production build completed!"

development-setup: install-dev docker-dev
	@echo "Development environment ready!"
	@echo "Run 'make shell-dev' to enter development container"
