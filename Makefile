.PHONY: help install install-dev clean test test-unit test-integration lint format type-check check build docs serve-docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install the package in production mode"
	@echo "  install-dev   Install the package in development mode"
	@echo "  clean         Remove build artifacts and caches"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and isort"
	@echo "  type-check    Run mypy type checking"
	@echo "  check         Run all checks (lint, type-check, test)"
	@echo "  build         Build distribution packages"
	@echo "  docs          Build documentation"
	@echo "  serve-docs    Serve documentation locally"

# Installation targets
install:
	uv sync --no-dev

install-dev:
	uv sync --all-groups

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Testing
test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration

test-coverage:
	uv run pytest --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality
lint:
	uv run ruff check src tests --fix

format:
	uv run ruff format src tests
	uv run isort src tests

type-check:
	uv run mypy src

# Combined checks
check: lint type-check test

# Building
build: clean
	uv build

# Documentation
docs:
	uv run --group docs mkdocs build

serve-docs:
	uv run --group docs mkdocs serve

# Development helpers
dev-shell:
	uv run python

example-simple:
	uv run python examples/simple_api.py

example-blog:
	uv run python examples/blog_api.py
