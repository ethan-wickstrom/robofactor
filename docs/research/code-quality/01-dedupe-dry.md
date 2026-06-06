---
title: Dedupe and DRY assessment
date: 2026-06-06
status: implemented
worker: "1/8"
scope:
  - src/robofactor
  - tests
domain_context: Robofactor
---

# Dedupe and DRY assessment

## Objective

Find duplicate logic and shallow pass-throughs across `src/robofactor` and `tests`, then apply DRY only where it improves locality and reduces the interface a maintainer has to understand.

## Method

- Read the repo instructions, context map, Robofactor glossary, code-quality guidance, testing guidance, verification guidance, Makefile, and project metadata.
- Ran an AST census over `src/robofactor` and `tests` for identical function bodies and one-line wrappers.
- Used `rg` to inspect repeated Check, QualityCheck, BehaviorTest, Evaluation, and score calculations.
- Read the source around each candidate before deciding whether dedupe would pass the deletion test.

The AST census found no identical function bodies. The useful findings were semantic duplicates: repeated Check execution, repeated pass-rate math, a parser collection pass that parsed successful values twice, and duplicated Python tool subprocess handling.

## Implemented

### Refactor Check now adapts CheckReport instead of rerunning checks

Before:

```text
check_candidate_code / check_refactored_code
|-- check_candidate / check_refactor
`-- quality_metrics_from_report
```

After:

```text
check_candidate/check_refactor
`-- CheckReport
    |-- CheckedRefactor.quality
    `-- CheckedRefactor.behavior
```

This makes `CheckReport` the single source of Check facts. `check_candidate_code` and
`check_refactored_code` now express their modes as separate functions instead of an optional
`source_code` switch.

### Training data now produces BehaviorTest facts directly

Training JSON now parses directly into `BehaviorTest` values with stable generated IDs. This removes
the old intermediate behavior-input shape and keeps behavior-preservation evidence in one model.

### QualityReport to QualityMetrics conversion is shared

`quality_metrics_from_report` owns the adaptation from Check quality facts to the DSPy-facing `QualityMetrics` model.

### Functional score consumers use CheckSummary.pass_rate

Main reward scoring, UI rendering, and the DSPy module score path consume the named `pass_rate` rather than repeating `passed / total` branches. `CodeRefactor.forward` now checks a generated artifact once in the normal path and reuses both quality metrics and the functional score.

### Ruff and ty diagnostic handling share one helper

`_ruff_issues` and `_ty_issues` keep their tool-specific commands, but share `_python_tool_issues` for subprocess execution and stdout/stderr normalization.

## Recommendations Not Implemented

### Do not dedupe FunctionSignature.render yet

`FunctionSignature.render` repeats parameter-kind filtering, but the repetition mirrors Python callable ordering. A generic grouping helper would hide a domain rule behind a lower-level collection abstraction. Leave it explicit unless another caller needs the same rendered parameter partitions.

### Keep the remaining collector until training parsing is redesigned

`collect(raw_items, parser)` is now the only collector entry point. It remains because it names the
training-data collection operation while hiding `ListParser` construction from `examples.py`.

### Leave the DSPy parser tree intact for now

`_create_dspy_example_parser` is visually large, but it represents one training-example schema. Splitting it would add names without adding leverage unless more schemas appear.

### Treat GEPA feedback construction as a future design pass

`_GEPARefactorMetric.__call__` still repeats failure-prediction construction and string assembly. A helper might reduce lines, but the deeper issue is a missing feedback taxonomy. Extracting a generic string builder now would probably move complexity instead of reducing it.

### Avoid test fixture dedupe until more Check tests accumulate

`tests/test_checks.py` repeats small code snippets and `BehaviorTest` construction. The tests are still clearer as local examples. Add fixtures only when the same named behavior appears across multiple files.

## Residual Risk

The worktree is intentionally busy and shared with other workers. During this pass, related cleanup also appeared around `CheckSummary.pass_rate`, `Json`, and typed training data. The implemented changes were kept consistent with the current worktree instead of reverting unrelated changes.

## Validation

Focused checks run during the pass:

- `uv run ruff check src/robofactor/checks/behavior.py src/robofactor/checks/__init__.py src/robofactor/refactor_check.py src/robofactor/modules/code_refactor.py src/robofactor/main.py src/robofactor/ui.py src/robofactor/data/_internal/collectors.py src/robofactor/checks/quality.py`
- `uv run ruff format --check src/robofactor/checks/behavior.py src/robofactor/checks/__init__.py src/robofactor/refactor_check.py src/robofactor/modules/code_refactor.py src/robofactor/main.py src/robofactor/ui.py src/robofactor/data/_internal/collectors.py src/robofactor/checks/quality.py`
- `uv run pytest tests/test_checks.py tests/test_refactor_check.py tests/modules/test_code_refactor_functional_score.py tests/data_internal/test_collectors.py`
- `make domain-language`
- `uv run ty check`
- `make architecture`

All focused checks passed. Pytest emitted existing DSPy/Rope deprecation warnings.

The full `make check` was attempted and stopped at `format-check` because `src/robofactor/data/examples.py` would be reformatted. That file had shared-worktree changes outside this DRY patch, so this pass did not format it blindly.
