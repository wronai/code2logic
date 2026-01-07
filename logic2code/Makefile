# Logic2Code - Code Generation from Logic Files
# Makefile for package management

.PHONY: help install install-dev test lint format typecheck build publish clean

PACKAGE_NAME = logic2code
PYTHON = python3
PIP = pip3

help:
	@echo "Logic2Code - Code Generation from Logic Files"
	@echo ""
	@echo "Usage:"
	@echo "  make install       Install package"
	@echo "  make install-dev   Install with dev dependencies"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linter"
	@echo "  make format        Format code"
	@echo "  make typecheck     Run type checker"
	@echo "  make build         Build package"
	@echo "  make publish       Publish to PyPI"
	@echo "  make publish-test  Publish to Test PyPI"
	@echo "  make clean         Clean build artifacts"
	@echo "  make demo          Run demo generation"

# Installation
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

install-full:
	$(PIP) install -e ".[llm,dev]"

# Testing
test:
	$(PYTHON) -m pytest tests/ -v

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=$(PACKAGE_NAME) --cov-report=html

# Code quality
lint:
	$(PYTHON) -m ruff check $(PACKAGE_NAME)/
	$(PYTHON) -m ruff check tests/ || true

format:
	$(PYTHON) -m black $(PACKAGE_NAME)/
	$(PYTHON) -m black tests/ || true
	$(PYTHON) -m isort $(PACKAGE_NAME)/
	$(PYTHON) -m isort tests/ || true

typecheck:
	$(PYTHON) -m mypy $(PACKAGE_NAME)/ --ignore-missing-imports

# Build & Publish
build: clean
	$(PYTHON) -m build

publish: build
	$(PYTHON) -m twine upload dist/*

publish-test: build
	$(PYTHON) -m twine upload --repository testpypi dist/*

# Clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf generated_code/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Development helpers
demo:
	@if [ -f ../project.c2l.yaml ]; then \
		$(PYTHON) -m $(PACKAGE_NAME) ../project.c2l.yaml -o generated_code/ -v; \
	else \
		echo "No project.c2l.yaml found. Run 'code2logic .' first."; \
	fi

summary:
	@if [ -f ../project.c2l.yaml ]; then \
		$(PYTHON) -m $(PACKAGE_NAME) ../project.c2l.yaml --summary; \
	else \
		echo "No project.c2l.yaml found. Run 'code2logic .' first."; \
	fi

# Version
version:
	@$(PYTHON) -c "from $(PACKAGE_NAME) import __version__; print(__version__)"

# Documentation
docs:
	@echo "Documentation: https://code2logic.readthedocs.io/en/latest/logic2code/"
