# Code Quality

Use this file when a task changes Python code, public types, linting, formatting, type checking, or architecture rules.

## Working Principles

- Leave the codebase better than you found it.
- Write simple, skimmable code with no clever indirection.
- Use precise domain language; rename or split overloaded concepts.
- Minimize states with required arguments, narrow types, and discriminated unions.
- Prefer data-and-function flow over class hierarchies and dependency injection.
- Push effects to outer layers; keep core logic pure where practical.
- Handle variants exhaustively, fail on unknown variants, and keep edits scoped.

## Ruff

- Use Ruff as the only linter, formatter, import sorter, syntax upgrader, unused-code checker, and Flake8-style checker.
- Run lint checks with `uv run ruff check .`.
- Run safe lint fixes with `uv run ruff check . --fix`.
- Format code with `uv run ruff format .`.
- Verify formatting with `uv run ruff format --check .`.
- Scope Ruff commands to changed paths during iteration, then run the full Ruff gate before completion.
- Keep generated, vendored, and external snapshot paths excluded explicitly.
- Do not add Black, isort, Flake8, autoflake, pyupgrade, or related config.
- Do not weaken Ruff rules to hide errors. Fix the code or add targeted per-file ignores with a reason.

## ty and Typing

- Use ty as the only final type checker.
- Run type checks with `uv run ty check`.
- Add ty to the appropriate development dependency group.
- Configure ty in `[tool.ty]` or `ty.toml`.
- Start the language server with `uv run ty server` when editor integration is needed.
- Do not add mypy or pyright config.
- Add explicit annotations for public functions, methods, class attributes, fixtures, callbacks, and boundary code.
- Annotate return types explicitly, including `-> None`.
- Prefer dataclasses, typed domain models, `TypedDict`, `Protocol`, enums, and narrow type aliases over loose `dict`, `object`, tuple payloads, or stringly typed values.
- Use `Any` only at untyped external boundaries. Convert to typed data immediately after the boundary.
- Prefer `collections.abc` interfaces such as `Mapping`, `Sequence`, `Iterable`, `Callable`, and `Iterator` for inputs.
- Use modern built-in generics and union syntax whenever supported by `project.requires-python`.
- Use `typing.override` for intentional overrides when supported by the project Python range.
- For distributed typed libraries, include `py.typed` and ensure it is included in built artifacts.
- Do not weaken type configuration to pass checks. Use local fixes or narrow, documented suppressions.

## Architecture

- Run `uv run tach check --exact` and `uv run tach check-external` when import boundaries or dependency directions change.
- Keep `tach.toml` exact: remove unused dependencies instead of leaving historical permissions in place.
- Keep boundary-crossing effects in adapters or entrypoints.
- Keep core scoring and check logic pure where practical.
