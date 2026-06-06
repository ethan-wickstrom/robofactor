.PHONY: help install install-dev clean test test-coverage domain-language lint lint-fix format format-check type-check architecture mutation lock-check audit build check readme

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install the package in production mode"
	@echo "  install-dev   Install the package in development mode"
	@echo "  clean         Remove generated local artifacts"
	@echo "  test          Run all tests"
	@echo "  test-coverage Run tests with coverage reports"
	@echo "  domain-language Check forbidden domain vocabulary"
	@echo "  lint          Run Ruff lint checks"
	@echo "  lint-fix      Run safe Ruff lint fixes"
	@echo "  format        Format code with Ruff"
	@echo "  format-check  Check Ruff formatting"
	@echo "  type-check    Run ty type checking"
	@echo "  architecture  Check package dependency direction"
	@echo "  mutation      Run mutation testing"
	@echo "  lock-check    Check uv lockfile freshness"
	@echo "  audit         Audit dependencies"
	@echo "  build         Build sdist and wheel"
	@echo "  check         Run the full project gate"
	@echo "  readme        Generate README.md using DSPy"

# Installation targets
install:
	uv sync --no-dev

install-dev:
	uv sync --all-groups

clean:
	rm -rf .coverage coverage.xml htmlcov optimized
	rm -rf .pytest_cache .ruff_cache build dist mutants
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# Testing
test:
	uv run pytest

test-coverage:
	uv run pytest --cov=robofactor --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "Coverage report generated in htmlcov/index.html"

domain-language:
	uv run pytest tests/test_domain_language.py --no-cov

# Code quality
lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

type-check:
	uv run ty check

architecture:
	uv run tach check --exact
	uv run tach check-external

mutation:
	uv run mutmut run
	uv run mutmut export-cicd-stats

lock-check:
	uv lock --check

audit:
	uv audit --locked --ignore-until-fixed GHSA-w8v5-vhqr-4h9v

build:
	uv build --clear

# Combined checks
check: lock-check format-check lint type-check architecture test audit build

# Documentation
readme:
	uv run scripts/generate_readme.py
