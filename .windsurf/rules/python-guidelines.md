---
trigger: glob
globs: **/*.py, src/**/*.py, tests/**/*.py
---

# Python Best Practices

## Project Structure

- Use src-layout with `src/your_package_name/`
- Place tests in `tests/` directory parallel to `src/`
- Keep configuration in `config/`
- Store requirements in `pyproject.toml`
- Place static files in `static/` directory
- Use `templates/` for Jinja2 templates

## Code Style

- Follow Black code formatting
- Use isort for import sorting
- Follow PEP 8 naming conventions:
  - snake_case for functions and variables
  - PascalCase for classes
  - UPPER_CASE for constants
- Maximum line length of 88 characters (Black default)
- Use absolute imports over relative imports

## Type Hints

- Use type hints for all function parameters and returns
- Import types from `typing` module
- Use `Optional[Type]` instead of `Type | None`
- Define custom types in `types.py`
- Use `Protocol` for duck typing

## Documentation

- Use Google-style docstrings
- Document all public APIs
- Use proper inline comments
- Generate API documentation
- Document environment setup

## Development Workflow

- Use virtual environments (venv)
- Implement pre-commit hooks
- Use proper Git workflow
- Follow semantic versioning
- Use proper CI/CD practices
- Implement proper logging

## Dependencies

- Pin dependency versions
- Separate dev dependencies
- Use proper package versions
- Regularly update dependencies
- Check for security vulnerabilities
