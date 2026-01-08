# LOLM - Lightweight Orchestrated LLM Manager
# Makefile for package management

.PHONY: help install install-dev test lint format typecheck build publish clean

PACKAGE_NAME = lolm
PYTHON = python3
PIP = pip3

help:
	@echo "LOLM - Lightweight Orchestrated LLM Manager"
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
	@echo "  make status        Show LLM status"

# Installation
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

install-full:
	$(PIP) install -e ".[full,dev]"

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

bump-patch:
	@$(PYTHON) -c 'import re; from pathlib import Path; pp=Path("pyproject.toml"); txt=pp.read_text(encoding="utf-8"); m=re.search(r"(?m)^version\s*=\s*\"(\d+)\.(\d+)\.(\d+)\"\s*$$", txt); (m is not None) or (_ for _ in ()).throw(SystemExit("Cannot find version in pyproject.toml")); maj, minor, patch = map(int, m.groups()); new_version=f"{maj}.{minor}.{patch+1}"; txt=re.sub(r"(?m)^version\s*=\s*\"\d+\.\d+\.\d+\"\s*$$", f"version = \"{new_version}\"", txt, count=1); pp.write_text(txt, encoding="utf-8"); init_py=Path("__init__.py"); init_txt=init_py.read_text(encoding="utf-8"); init_txt, n=re.subn(r"(?m)^__version__\s*=\s*[\x27\x22]\d+\.\d+\.\d+[\x27\x22]\s*$$", f"__version__ = \x27{new_version}\x27", init_txt, count=1); n and init_py.write_text(init_txt, encoding="utf-8"); print(new_version)'

publish: bump-patch build
	$(PYTHON) -m twine upload dist/*

publish-test: bump-patch build
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
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Development helpers
status:
	$(PYTHON) -m $(PACKAGE_NAME) status

test-llm:
	$(PYTHON) -m $(PACKAGE_NAME) test

models:
	$(PYTHON) -m $(PACKAGE_NAME) models

# Version
version:
	@$(PYTHON) -c "from $(PACKAGE_NAME) import __version__; print(__version__)"

# Documentation
docs:
	@echo "Documentation: https://code2logic.readthedocs.io/en/latest/lolm/"
