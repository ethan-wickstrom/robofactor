---
title: Unused Code Audit
tags:
  - code-quality
  - unused-code
  - dependencies
  - research-notes
category: compound-engineering
status: active
updated: 2026-06-06
---

# Unused Code Audit

Worker 3/8 audited Robofactor for code, exports, tests, dependency declarations, config, and generated leftovers that were unused enough to remove.

## Scope

- Read `AGENTS.md`, `Makefile`, `pyproject.toml`, `tach.toml`, `CONTEXT-MAP.md`, `src/robofactor/CONTEXT.md`, and the relevant agent docs.
- Treated `robofactor = robofactor.main:app`, `robofactor.checks` Tach interface entries, and package `__all__` facades as public boundaries.
- Used Python-appropriate tooling. Knip was not applicable because this repo is a Python package, not a JavaScript/TypeScript package-export graph.

## Tool Findings

- `uv run ruff check . --output-format concise`: clean.
- `uv run ruff check . --select F401,F841,F821,F822,F823,ARG --output-format concise`: clean for unused imports, locals, unresolved names, and unused arguments.
- `uv run ty check --output-format concise`: clean.
- `uv run tach check --exact --output json`: `[]`; no stale internal dependency declarations.
- `uv run tach check-external`: clean.
- `uv run ruff analyze graph src tests scripts --direction dependents --detect-string-imports`: no unreferenced internal source module that was safe to delete; public/package init files were left intact.
- `uvx --from vulture vulture src tests scripts --min-confidence 60 --sort-by-size`: useful low-confidence signals, but remaining reports are public model fields, enum members, Pydantic fields, Typer commands, and DSPy hooks.
- `uvx --from deptry deptry ...`: initially reported `gepa`, `ruff`, and `ty`; after cleanup it reports only `ruff` and `ty`, which are intentionally invoked as tools rather than imported.
- `uv run pytest --collect-only -q --no-cov`: collected 24 tests in 9 files before and after removing the empty test package marker.
- `fd -a -t e -t d . src tests docs scripts`: found an empty `docs/research/code-quality/` directory, now used by this report.

## Removed

- Removed unused README-generation scaffolding in `scripts/generate_readme.py`:
  - `SRC_DIR`
  - `ModuleApi`
  - `ProjectContext.pyproject_text`
  - `ProjectContext.modules`
  - `_list_source_modules`
  - the unused `_format_api_section` parameter
  Evidence: Vulture reported the stale helper/fields, `rg` found no external references, and focused Ruff/format checks passed.

- Removed unused analysis constants from `src/robofactor/config.py`:
  - `RUFF_COMPLEXITY_CODE`
  - `RUFF_MAX_COMPLEXITY`
  - `LINTING_PENALTY_PER_ISSUE`
  Evidence: Vulture and `rg` agreed they were definition-only; active Ruff complexity behavior lives in `src/robofactor/checks/quality.py` and `pyproject.toml`.

- Removed redundant direct dependency `gepa` from `pyproject.toml` with `uv remove gepa` and removed the matching stale `tach.toml` external exclusion.
  Evidence: deptry reported the direct dependency unused; `uv tree --locked --all-groups --invert --package gepa` shows `gepa` is still supplied transitively by `dspy`, which is the actual Robofactor API path for `dspy.GEPA`.

- Removed empty `tests/modules/__init__.py`.
  Evidence: the file was empty, `rg` found no `tests.modules` imports, and pytest collection remained 24 tests in 9 files.

## Kept

- Kept `ruff` and `ty` dependencies despite deptry reports. They are invoked by `src/robofactor/checks/quality.py` via `sys.executable -m ruff` and `sys.executable -m ty`, and are also first-class project gates.
- Kept Vulture-reported `QualityReport.ty_passed`, enum members, Pydantic model fields, and `_missing_` enum methods. These are schema/model surface or normalization hooks, not normal call references.
- Kept `CodeRefactor.forward`; DSPy calls `forward` by convention.
- Kept Typer command functions `scripts/generate_readme.py::main` and `src/robofactor/main.py::main`; command registration is decorator/entrypoint based.
- Kept package facade exports in `robofactor`, `robofactor.data`, `robofactor.utils`, and `robofactor.checks` because they are public import surfaces even when internal import graph dependents are sparse.

## Validation

- Passed: `uv lock --check`
- Passed: `uv run ruff check . --output-format concise`
- Passed: `uv run tach check-external`
- Passed: `uv audit --locked --ignore-until-fixed GHSA-w8v5-vhqr-4h9v`
- Passed: `uv build --clear`
- Ran with reviewed residual Vulture findings: `uvx --from vulture vulture src tests scripts --min-confidence 60 --sort-by-size`
- Failed: `uv run ruff format --check .`
  - Existing/concurrent blocker: `src/robofactor/data/examples.py` would be reformatted.
- Failed: `uv run ty check --output-format concise`
  - Historical concurrent blocker: `tests/data_internal/test_parsers.py` passed a plain boolean lambda where `BasicParser` now requires a `TypeIs[str]`.
- Failed: `uv run tach check --exact`
  - Historical concurrent blocker: `src/robofactor/checks/behavior.py` imported the old data behavior-input model, crossing the Tach boundary.
- Failed: `uv run pytest`
  - Historical concurrent blocker: `tests/data_internal/test_parsers.py` imported `OptionalParser` after the parser no longer exported it.
- Failed as expected from the first full-gate blocker: `make check`
  - `make check` stopped at `uv run ruff format --check .` because `src/robofactor/data/examples.py` needs formatting.

## Recommendations

- Add a documented dependency-check target that runs deptry with a project-specific ignore for tool-invoked packages: `ruff` and `ty`.
- Consider moving `ruff` and `ty` to the dev group only if Robofactor stops running quality checks inside the installed runtime. Today they are runtime behavior.
- Consider adding a small Vulture whitelist for framework hooks and public models if this audit becomes a recurring gate.
- README generation is deterministic but still deliberately avoids API discovery. If API documentation becomes important, add static AST discovery rather than dynamic imports.
