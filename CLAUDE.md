# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
- `make install-dev` - Install project with development dependencies using UV
- `uv sync` - Sync dependencies if pyproject.toml changes

### Running the Application
- `uv run python -m resting_agent` - Run the CLI application
- `uv run resting-agent` - Alternative way to run after installation

### Testing
- `make test` - Run full test suite with pytest
- `make test-cov` - Run tests with coverage report
- `uv run pytest tests/test_specific.py::test_function` - Run specific test

### Code Quality
- `make check` - Run all quality checks (lint, format check, type check)
- `make lint` - Run ruff linting
- `make format` - Auto-format code with ruff
- `make type-check` - Run pyright type checking

### Build and Distribution
- `make build` - Build distribution packages
- `make clean` - Clean build artifacts and caches

## Architecture Overview

Resting-Agent is an autonomous agent that generates complete RESTful APIs for Laravel applications from natural language descriptions. It uses DSPy for structured AI interactions.

### Core Components

1. **ApiAgent** (agent.py) - Main orchestrator that:
   - Accepts natural language intent
   - Creates execution plans via DSPy
   - Executes actions to generate Laravel code
   - Verifies results

2. **Action System** - Task decomposition:
   - **Action** (models.py): Individual task (create file, run command)
   - **Plan** (models.py): Ordered actions with dependencies
   - **Executors** (executors.py): Execute specific action types

3. **DSPy Signatures** (signatures.py):
   - **GeneratePlan**: Natural language → structured plan
   - **GenerateCode**: File spec → Laravel PHP code

4. **Services**:
   - **FileSystemHandler**: Safe file operations
   - **CommandExecutor**: Shell command execution with Laravel context

### Workflow
1. User provides natural language API description
2. GeneratePlan creates structured action plan
3. ApiAgent executes actions in dependency order
4. Each action generates Laravel artifacts (models, controllers, tests)
5. Verification ensures generated code works

### Laravel Code Generation Targets
- Database migrations
- Eloquent models
- REST controllers
- Form requests (validation)
- API resources
- Routes
- Feature tests

### Key Conventions
- Python 3.12+ with strict type hints
- Pydantic models for data validation
- DSPy for LLM interactions
- Laravel 10+ and PSR-12 for generated PHP code