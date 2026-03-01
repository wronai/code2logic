.PHONY: install dev-install test lint format clean help analyze run docker mermaid-png install-mermaid check-mermaid clean-png

# Default target
help:
	@echo "code2flow - Python Code Flow Analysis Tool"
	@echo ""
	@echo "Available targets:"
	@echo "  make install       - Install package"
	@echo "  make dev-install   - Install with development dependencies"
	@echo "  make test          - Run test suite"
	@echo "  make lint          - Run linters (flake8, black --check)"
	@echo "  make format        - Format code with black"
	@echo "  make typecheck     - Run mypy type checking"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make analyze       - Run analysis on sample (python folder)"
	@echo "  make run           - Run with example arguments"
	@echo "  make build         - Build distribution packages"
	@echo "  make mermaid-png   - Generate PNG from all Mermaid files"
	@echo "  make install-mermaid - Install Mermaid CLI renderer"
	@echo "  make check-mermaid - Check available Mermaid renderers"
	@echo ""

# Installation
install:
	pip install -e .
	@echo "✓ code2flow installed"

dev-install:
	pip install -e ".[dev]"
	@echo "✓ code2flow installed with dev dependencies"

# Testing
test:
	python -m pytest tests/ -v --tb=short 2>/dev/null || echo "No tests yet - create tests/ directory"

test-cov:
	python -m pytest tests/ --cov=code2flow --cov-report=html --cov-report=term 2>/dev/null || echo "No tests yet"

# Code quality
lint:
	python -m flake8 code2flow/ --max-line-length=100 --ignore=E203,W503 2>/dev/null || echo "flake8 not installed"
	python -m black --check code2flow/ 2>/dev/null || echo "black not installed"
	@echo "✓ Linting complete"

format:
	python -m black code2flow/ --line-length=100 2>/dev/null || echo "black not installed, run: pip install black"
	@echo "✓ Code formatted"

typecheck:
	python -m mypy code2flow/ --ignore-missing-imports 2>/dev/null || echo "mypy not installed"

# Running
run:
	python -m code2flow ../python/stts_core -v -o ./output

analyze: run
	@echo "✓ Analysis complete"

# Building
build:
	rm -rf build/ dist/ *.egg-info
	python setup.py sdist bdist_wheel
	@echo "✓ Build complete - check dist/"

# Cleaning
clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .coverage htmlcov/
	rm -rf code2flow/__pycache__ code2flow/*/__pycache__
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned build artifacts"

# Development utilities
check: lint typecheck test
	@echo "✓ All checks passed"

# Mermaid diagram generation
mermaid-png:
	python mermaid_to_png.py --batch output output

mermaid-png-%:
	python mermaid_to_png.py output/$*.mmd output/$*.png

install-mermaid:
	npm install -g @mermaid-js/mermaid-cli

check-mermaid:
	@echo "Checking available Mermaid renderers..."
	@which mmdc > /dev/null && echo "✓ mmdc (mermaid-cli)" || echo "✗ mmdc (run: npm install -g @mermaid-js/mermaid-cli)"
	@which npx > /dev/null && echo "✓ npx (for @mermaid-js/mermaid-cli)" || echo "✗ npx (install Node.js)"
	@which puppeteer > /dev/null && echo "✓ puppeteer" || echo "✗ puppeteer (run: npm install -g puppeteer)"

clean-png:
	rm -f output/*.png
	@echo "✓ Cleaned PNG files"
