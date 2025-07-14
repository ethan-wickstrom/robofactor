.PHONY: help install install-dev clean test lint format type-check check readme

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install the package in production mode"
	@echo "  install-dev   Install the package in development mode with all groups"
	@echo "  clean         Remove build artifacts and caches"
	@echo "  test          Run all tests with pytest"
	@echo "  lint          Run ruff linting checks with auto-fix"
	@echo "  format        Format code with ruff"
	@echo "  type-check    Run basedpyright type checking"
	@echo "  check         Run all checks (type-check, lint, test)"
	@echo "  readme        Generate README.md using DSPy"

# Installation targets
install:
	uv sync --no-dev

install-dev:
	uv sync --all-groups

# Cleaning
clean:
	rm -rf build dist .eggs *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Testing
test:
	@if [ -n "$$(find tests -name '*.py' 2>/dev/null)" ]; then \
		uv run pytest; \
	else \
		echo "No tests found in tests directory"; \
	fi

# Code quality
lint:
	uv run ruff check src tests --fix

format:
	uv run ruff format src tests

type-check:
	uv run basedpyright --pythonversion 3.13 src

# Combined checks
check: type-check lint test

# Documentation
readme:
	uv run python scripts/generate_readme.py
