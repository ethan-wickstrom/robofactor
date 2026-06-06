# Testing

Use this file when a task creates, changes, or runs tests.

## Pytest

- Use pytest as the only test runner.
- Write tests as pytest functions or pytest-style classes.
- Put tests in `tests/` for packages and applications unless package-internal tests are required for distribution.
- Use pytest discovery names: `test_*.py`, `*_test.py`, `test_*` functions, and `Test*` classes without `__init__`.
- Configure pytest in `[tool.pytest.ini_options]`.
- Set `testpaths = ["tests"]` when tests live under `tests/`.
- Use `--import-mode=importlib`.
- Enable strict pytest behavior with `--strict-config` and `--strict-markers`.
- Run a focused test while iterating: `uv run pytest path/to/test_file.py::test_name`.
- Run a changed test file with `uv run pytest path/to/test_file.py`.
- Run a selected subset with `uv run pytest -k "<expression>"`.
- Run the full suite with `uv run pytest`.
- Prefer fixtures for reusable setup and cleanup.
- Keep fixture scope as narrow as practical.
- Prefer `pytest.mark.parametrize` for repeated input/output cases.
- Use `tmp_path`, `monkeypatch`, and local fixtures instead of persistent global state, real user files, or permanent environment changes.
