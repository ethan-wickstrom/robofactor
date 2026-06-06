---
title: Legacy and Fallback Path Audit - Worker 7
tags:
  - code-quality
  - compatibility
  - legacy-cleanup
  - documentation
category: code-quality
status: complete
updated: 2026-06-06
worker: 7
---

# Legacy and Fallback Path Audit - Worker 7

## Scope

Reviewed `AGENTS.md`, `README.md`, `docs/agents/project-structure.md`, `pyproject.toml`,
`CONTEXT-MAP.md`, `src/robofactor/CONTEXT.md`, source modules, tests, GitHub workflow
configuration, and existing `docs/research/code-quality/` notes.

Search terms included `deprecated`, `legacy`, `fallback`, `compat`, `backcompat`, `old-path`,
`old path`, `old_`, `temp`, `TODO`, `isort`, `dspy_modules`, `function_extraction`,
`gepa_utils`, `DSPyTrace`, and `OptionalParser`.

## Tool Catalog

- `rg` and `fd`: primary inventory tools for markers, stale module references, and affected files.
- `uv tree --depth 1 --locked`: direct dependency inventory.
- `uv audit --locked --ignore-until-fixed GHSA-w8v5-vhqr-4h9v`: dependency health check.
- Context7: attempted for DSPy docs but the MCP startup timed out.
- Web primary sources: official DSPy docs and the DSPy GEPA docs were used to assess the
  `dspy.teleprompt.gepa.gepa_utils` import surface.

## Current Check Path

```text
check_candidate_code / check_refactored_code
|-- check_candidate/check_refactor
|   `-- CheckReport
`-- CheckedRefactor
```

Training JSON now parses directly into the `BehaviorTest` model used by deterministic checks.

## Inventory

| Location | Finding | Assessment |
| --- | --- | --- |
| `src/robofactor/checks/behavior.py` | Old behavior-input adapters named current data as old-path code. | Removed the adapter; training examples now parse directly into `BehaviorTest`. |
| `src/robofactor/checks/__init__.py` and `tach.toml` | The checks facade exposed behavior-input conversion helpers. | Facade now exposes only check facts and check functions. |
| Old analysis/evaluation modules | Compatibility-shaped modules kept old mode switches and vague names. | Removed and replaced with `refactor_check.py`. |
| `src/robofactor/data/_internal/parsers.py` | `OptionalParser` encoded an unused null-to-default branch. | Removed the parser and its export. |
| `tests/data_internal/test_parsers.py` | Test coverage preserved the optional/default parser path. | Reworked the test to assert explicit null is a present but invalid value. |
| `src/robofactor/main.py` | Imported `DSPyTrace` from `dspy.teleprompt.gepa.gepa_utils` for annotations only. | Removed the internal import and annotated GEPA trace inputs as `object`. |
| `README.md` and `pyproject.toml` | README still described formatting as Ruff plus isort; `pyproject.toml` labeled Ruff rule `I` as isort. | Updated wording to Ruff import sorting. |
| Existing research notes | Older notes referenced stale behavior-input adapters and now-resolved parser blockers. | Refreshed those references to the current names or historical wording. |
| `docs/agents/project-structure.md` | Documented old scoring modules. | Updated to the current `refactor_check.py` architecture. |
| `src/robofactor/types.py` | Re-exported `Json` from `robofactor.json_value`. | Removed; callers import `Json` from `robofactor.json_value`. |
| `src/robofactor/types.py` | Enum `_missing_` methods normalize alternate strings and some typo-like values. | Preserved for now because these are DSPy/Pydantic schema boundaries; removal needs a focused model-output validation pass. |
| `src/robofactor/data/training.json` | Contains intentionally messy snippets with `temp_list`, fallback-like parsing, and broad exception handling. | Preserved. These are refactoring training examples, not Robofactor runtime paths. |
| `src/robofactor/checks/quality.py` | Uses `tempfile.NamedTemporaryFile` for Ruff and ty checks. | Preserved. This is an IO implementation detail with cleanup in `finally`, not a temporary code path. |

## Dependency Assessment

Direct dependencies are current enough for this pass: `uv tree --depth 1 --locked` resolved
`dspy-ai 3.2.1`, `gepa 0.0.27`, `ruff 0.15.16`, `ty 0.0.44`, `typer 0.26.7`, and the expected
runtime/test tools.

`uv audit --locked --ignore-until-fixed GHSA-w8v5-vhqr-4h9v` found no known vulnerabilities and no
adverse project statuses. No deprecated package dependency was found.

The only high-confidence deprecated usage was not a package but an import path:
`dspy.teleprompt.gepa.gepa_utils.DSPyTrace`. Official DSPy docs present stable public usage through
`dspy.GEPA`, `dspy.Predict`, `dspy.Module`, and `dspy.PythonInterpreter`, while the GEPA docs
describe feedback metrics by call shape rather than requiring an internal trace type import:

- [DSPy docs](https://dspy.ai/)
- [DSPy GEPA overview](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/GEPA/overview.md)

## Changes Made

- Removed legacy-labeled behavior-test conversion by parsing training examples directly into `BehaviorTest`.
- Replaced old analysis/evaluation modules with `refactor_check.py`.
- Removed the old `Json` re-export from `types.py`.
- Removed the singular adapter from the public `robofactor.checks` facade.
- Removed the unused `OptionalParser` fallback branch and adjusted parser tests.
- Removed the internal DSPy GEPA utility import used only for annotations.
- Updated stale dependency/tool wording in README and pyproject comments.
- Refreshed research notes that pointed at old helper names or already-resolved blockers.

## Validation

- Passed: `uv run ruff format src/robofactor/checks/behavior.py src/robofactor/data/_internal/parsers.py src/robofactor/main.py tests/data_internal/test_parsers.py`
- Passed: `uv run ruff check src/robofactor/checks/behavior.py src/robofactor/data/_internal/parsers.py src/robofactor/main.py tests/data_internal/test_parsers.py`
- Passed: `uv run pytest tests/data_internal/test_parsers.py tests/test_checks.py tests/test_refactor_check.py`
- Passed: `make domain-language`
- Passed: `make architecture`
- Passed: `make check`

`make check` emitted third-party deprecation warnings from DSPy internals about the deprecated
`prefix` argument and from Rope internals about deprecated project/path helpers. These warnings do
not point to Robofactor source usage in this pass.

## Critical Assessment

The codebase had a small number of accidental old-path names. The dangerous part was not behavior
duplication; it was language drift. The previous behavior-input adapter made training data look
like a separate domain from behavior-preservation evidence. Removing it makes the current domain
process easier to infer: training JSON becomes `BehaviorTest` facts, then checks produce a
`CheckReport`.

`OptionalParser` was the only clear fallback implementation with no runtime use. Keeping it would
have expanded parser states from "required field exists and parses" to "missing fails, null may
default, other values parse." The current project data does not need that state.

The DSPy trace annotation was also low-value risk. It tied `main.py` to an internal-looking module
while the code only uses truthiness, `str(...)`, and `getattr(...)` at a dynamic callback boundary.
Typing that boundary as `object` is more honest and avoids depending on a private import path.

No high-confidence legacy/fallback recommendations remain in this pass.

## Recommendations Not Implemented

1. Audit `OpportunityCategory._missing_`, `PlanStepFocus._missing_`, and
   `RecommendationPriority._missing_` with DSPy structured-output fixtures. Prefer strict enum
   failures unless real model outputs require normalization.
2. Add a docs check for stale `src/robofactor/...` file references so README and research notes
   cannot keep pointing to removed modules.
