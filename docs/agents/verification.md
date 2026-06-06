# Verification

Use this file when choosing checks for a completed change or updating CI.

## Local Gates

Run the smallest relevant command first, then broaden to the full gate before completion when the change affects code, packaging, or CI.

- Full gate: `make check`
- Tests: `make test` or `uv run pytest`
- Coverage: `make test-coverage`
- Lint: `make lint` or `uv run ruff check .`
- Safe lint fixes: `make lint-fix` or `uv run ruff check . --fix`
- Format: `make format` or `uv run ruff format .`
- Format check: `make format-check` or `uv run ruff format --check .`
- Type check: `make type-check` or `uv run ty check`
- Architecture: `make architecture`
- Domain language: `make domain-language`
- Dependency lock freshness: `make lock-check` or `uv lock --check`
- Vulnerability audit: `make audit`
- Package build: `make build` or `uv build`

`make audit` ignores `GHSA-w8v5-vhqr-4h9v` only with `--ignore-until-fixed` because it is currently a transitive `dspy -> diskcache` advisory with no fixed release.

## Change-Specific Checks

- For dependency changes, run `uv lock`, `uv lock --check`, `uv sync --all-extras --all-groups`, and at least one test or import command that exercises the dependency.
- For lint-only changes, run `uv run ruff check .` and `uv run ruff format --check .`.
- For format-only changes, run `uv run ruff format .` and `uv run ruff format --check .`.
- For typing changes, run `uv run ty check`.
- For test changes, run the focused pytest command and then `uv run pytest`.
- For packaging changes, run `uv build` and a package smoke test.

## CI

CI for applications and workspaces must include:

- `uv python install`
- Deno `2.x` setup when checks run DSPy's Pyodide-backed Python interpreter
- `uv sync --locked --all-extras --all-groups`
- `uv lock --check`
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run ty check`
- `uv run tach check --exact`
- `uv run tach check-external`
- `uv run pytest`

CI for distributable packages must also include `uv build`.

CI must not use pip installs, Poetry installs, Pipenv installs, pyenv setup, virtualenv setup, tox, nox, Black, isort, Flake8, mypy, pyright, or bare unittest commands.
