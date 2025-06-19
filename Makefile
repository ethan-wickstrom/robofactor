.PHONY: help install install-dev clean test test-unit test-integration lint format type-check check build docs serve-docs readme

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
	@echo "  readme        Generate README.md using DSPy"

# Installation targets
install:
	uv sync --no-dev

install-dev:
	uv sync --all-groups

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

# Documentation
readme:
	uv run scripts/generate_readme.py
