# Robofactor Agent Guide

## Commands
- **Run all checks**: `make check` (runs lint, type-check, test)
- **Test all**: `uv run pytest`
- **Test single file**: `uv run pytest tests/path/to/test_file.py`
- **Test single function**: `uv run pytest tests/path/to/test_file.py::test_function_name`
- **Lint**: `uv run ruff check src tests --fix`
- **Format**: `uv run ruff format src tests`
- **Type-check**: `uv run ty check`

## Architecture
- **Package**: Python CLI tool using DSPy for LLM-powered code refactoring
- **Core modules**: `src/robofactor/{main.py, modules/, evaluation.py, analysis.py, ui.py}`
- **Key components**: DSPy modules (`modules/`), evaluation pipeline (railway-oriented with `returns`), AST-based analysis
- **CLI**: Built with `typer`, rich formatting with `rich` library
- **No database/API**: Stateless CLI tool, optional MLflow integration for experiment tracking

## Code Style
- **Python**: 3.12+, line length 100, follow Ruff config (`pyproject.toml`)
- **Imports**: Standard library → third-party → local (sorted by isort/Ruff)
- **Types**: Use type hints everywhere; check with `ty`
- **Error handling**: Railway-oriented programming with `returns.result` (Success/Failure), beartype for runtime validation
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Testing**: pytest with coverage, test paths mirror `src/` structure
