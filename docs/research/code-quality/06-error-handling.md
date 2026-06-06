---
title: Error Handling Inventory
tags:
  - code-quality
  - error-handling
  - behavior-preservation
category: compound-engineering
status: active
updated: 2026-06-06
worker: 6
---

# Error Handling Inventory

Worker 6 focused on defensive control flow: broad `try`/`except`, `returns.safe`,
`contextlib.suppress`, default fallbacks, dynamic `getattr`, and catch-all scoring paths.
The cleanup rule was: keep explicit edge handling for external input, file IO, DSPy's
Python sandbox, and user-facing CLI errors; remove internal fallbacks that hide broken
assumptions.

## Program Shape

```text
external input, files, model output, tool output
  -> named boundary adapter
     -> Result, CheckReport, Raised, or CLI message
        -> pure scoring, rendering, AppliedChange, or ApplyFailed
```

The bad shape is the inverse: core scoring or rendering code invents an empty list,
`None`, zero score, or generic failure after a value should already exist.

## Inventory

| Location | Handling | Classification | Assessment |
| --- | --- | --- | --- |
| `src/robofactor/checks/signature.py` | `SyntaxError` becomes `Failure` | Boundary input handling | Keep. Candidate/refactored code is unknown Python input. |
| `src/robofactor/data/_internal/parsers.py` | JSON type, missing field, parse, and constructor errors become `Failure` | Boundary input handling | Keep, but tightened. Missing key and explicit `null` are now distinct, and constructor catches are narrowed to `TypeError` and `ValueError`. |
| `src/robofactor/types.py` | enum `_missing_` methods normalize model text | Boundary input handling | Keep for now. This is LLM/Pydantic edge normalization, but it should stay visible because aliases can become too permissive. |
| `src/robofactor/utils/load_json.py` | file read and JSON decode errors become `Failure` | IO boundary | Keep, but replaced `@safe` with explicit `OSError` and `json.JSONDecodeError`. |
| `src/robofactor/checks/quality.py` | temp-file cleanup in `finally` | IO boundary | Keep. The effect is local and required to avoid leaked temp files. |
| `src/robofactor/checks/quality.py` | `tokenize.TokenError` during suppression scan | Boundary input handling | Changed. It now reports a suppression-scan failure instead of returning no warnings. |
| `src/robofactor/checks/behavior.py` | `NoSuchExample` from Hypothesis `find` | Check search control flow | Keep. This is the library's signal that generated comparisons found no counterexample. |
| `src/robofactor/checks/behavior.py` | sandbox execution exceptions become `Raised` | Interpreter sandbox handling | Keep. User/model code failures are behavior outcomes for a Check. |
| `src/robofactor/refactoring.py` | Rope project/resource failures become `Failure` | IO/tool boundary | Keep with clarification. Rope exposes several library failures; the domain result is "RopeProject inspection failed." |
| `src/robofactor/main.py` | Typer command surfaces errors to console and exits | User-facing CLI errors | Keep. This is the CLI edge, not core logic. |
| `src/robofactor/main.py` | dynamic DSPy prediction/trace `getattr` | Boundary input handling | Keep for now. DSPy predictions are dynamic model/tool output. A typed adapter would be better, but that is a broader consolidation. |
| `src/robofactor/modules/code_refactor.py` | invalid model output scores zero | Boundary input handling | Keep. The score is the GEPA contract for invalid generated code. |
| `scripts/generate_readme.py` | outer command catches exceptions and exits | User-facing CLI errors | Keep. Internal discovery fallbacks were removed. |

## Removed Or Clarified

- Removed `contextlib.suppress(Exception)` from package import-time beartype setup. Enabling beartype is explicit, so import or hook failures should fail loudly.
- Replaced `returns.safe` in `load_json` with explicit IO and JSON parse failures.
- Removed `returns.safe` around checked-refactor construction; after a successful `CheckReport`, construction should not fail.
- Replaced `DictParser.raw_data.get(...)` with explicit key presence checks so missing fields do not erase explicit JSON `null`.
- Narrowed `DictParser` constructor failure handling to `TypeError` and `ValueError`.
- Changed `suppression_warnings` so tokenization failure becomes a warning message instead of an empty tuple.
- Replaced candidate-check `.value_or(0)` scoring with an assertion on impossible check failure.
- Made reward scoring log training-example load failures instead of silently falling through.
- Made `--optimize` fail when training examples cannot load, instead of proceeding without optimization.
- Removed impossible catch-all zero scores in GEPA/check paths.
- Replaced empty behavior-test fallbacks with explicit empty tuples at call sites.
- Removed typed-list fallbacks in UI rendering.
- Removed README generation fallbacks for missing project metadata, failed CLI help, Makefile sniffing, and pip/python instructions.

## Remaining Recommendations

- Move DSPy prediction extraction into a small named adapter, for example `extract_refactored_artifact`, and keep all dynamic `getattr` calls there. I did not do this because it crosses into type-consolidation work shared with other workers.
- Consider naming reward-score failures as data, such as `ZeroRewardReason`, so no-artifact, empty-code, check-failed, and no-example cases remain distinguishable during GEPA debugging.
- Revisit enum `_missing_` alias maps after more examples are collected. They are useful at the model-input edge, but misspelling aliases such as `ERRORHANDLNG` may train the system to accept vague output.
- Map known Rope exceptions if tests reveal stable exception classes. Until then, the broad catch remains isolated at the RopeProject inspection edge.
- Decide whether non-`--optimize` startup should continue with an unoptimized model when training examples fail. The current behavior is explicit and user-facing, but a stricter CLI mode may be preferable once optimized models are expected in normal use.
