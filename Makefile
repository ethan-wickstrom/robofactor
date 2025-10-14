.PHONY: help install install-dev clean test lint format type-check check readme

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install the package in production mode"
	@echo "  install-dev   Install the package in development mode"
	@echo "  test          Run all tests with coverage"
	@echo "  lint          Run linting checks with Ruff"
	@echo "  format        Format code with Ruff"
	@echo "  type-check    Run type checking with Ty"
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

# Code quality
lint:
	uv run ruff check src tests --fix

format:
	uv run ruff format src tests

type-check:
	uv run ty check

# Combined checks
check: lint type-check test

# Documentation
readme:
	uv run scripts/generate_readme.py
