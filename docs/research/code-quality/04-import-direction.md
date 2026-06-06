---
title: Import Direction and Cycle Assessment
tags:
  - code-quality
  - architecture
  - imports
  - tach
  - research-notes
category: compound-engineering
status: implemented
updated: 2026-06-06
worker: "4/8"
---

# Import Direction and Cycle Assessment

## Objective

Worker 4/8 assessed Robofactor's import graph for circular dependencies, over-permissive module
rules, missing interfaces, and utility modules that could hide dependency direction drift.

## Tool Catalog

- Tach is the primary architecture tool because Robofactor is a Python package and already keeps
  `tach.toml` in the project gate.
- `rg`, `fd`, and local source inspection were used for import searches and public facade checks.
- A small `uv run python` AST pass checked strongly connected components at source-module level.
- Context7 was used for current Tach command/configuration behavior. The useful confirmed details
  were: `tach check --exact` fails stale dependency declarations, interfaces limit imports to
  public names, `check-external` compares third-party imports to declared package dependencies,
  and `utility = true` allows imports from every module without explicit `depends_on`.
- Madge was not used. It is aimed at JavaScript/TypeScript dependency graphs and would not inspect
  Python imports or `tach.toml` boundaries here.
- `pydeps` and import-linter were checked through `uv run ... --help`; neither command is installed
  in the current project environment.

## Current Import Tree

Tach's module graph after the implemented changes:

```text
robofactor.main
|- robofactor.config
|- robofactor.data
|  |- robofactor.data._internal
|  `- robofactor.utils
|- robofactor.refactor_check
|  |- robofactor.checks
|  |- robofactor.data
|  `- robofactor.types
|- robofactor.modules
|  |- robofactor.data
|  |- robofactor.refactor_check
|  |- robofactor.signatures
|  `- robofactor.types
|- robofactor.types
`- robofactor.ui
   |- robofactor.config
   |- robofactor.refactor_check
   `- robofactor.types

robofactor.checks
|- robofactor.checks.engine
|  |- robofactor.checks.behavior
|  |- robofactor.checks.model
|  |- robofactor.checks.quality
|  `- robofactor.checks.signature
|- robofactor.checks.behavior -> robofactor.checks.model
|- robofactor.checks.quality -> robofactor.checks.model
`- robofactor.checks.signature -> robofactor.checks.model
```

`robofactor.json_value` is the only remaining universal utility module. It has no internal or
external dependencies, so imports to it cannot create a cycle.

## Findings

### No active circular dependencies

`uv run tach check --exact --output json` returned `[]`, and the AST strongly connected component
pass found no multi-module cycles. `forbid_circular_dependencies = true` remains enabled in
`tach.toml`, so future package-level cycles should fail the architecture gate.

### The former utility set was too permissive

`robofactor.types`, `robofactor.config`, and `robofactor.utils` were all `utility = true`.
That made them importable everywhere without exact dependency declarations. This was too broad:

- `types.py` is a DSPy/Pydantic schema surface, not a dependency-free primitive module.
- `config.py` is only used by CLI and terminal presentation code.
- `utils.load_json` is only used by data example loading.

These modules now have explicit Tach dependencies and, where appropriate, visibility rules.

### `Json` needed its own dependency-free home

Before this pass, pure check/data code imported `Json` through `robofactor.types`, which also imports
DSPy and Pydantic. That coupled deterministic checks and JSON parsing to the DSPy schema module for
one recursive alias.

`src/robofactor/json_value.py` now owns `Json`. Checks, data loading, parsers, and collectors import
from that module directly.

### `data._internal` was private by convention only

`robofactor.data._internal` had no Tach boundary, so source modules could import data parsers or
collectors directly while Tach would only see a broad dependency on `robofactor.data`. It is now a
separate module visible only to `robofactor.data`.

### The checks facade is the right public interface

External source modules import deterministic check behavior through `robofactor.checks`, not through
`checks.behavior`, `checks.engine`, or `checks.model`.

### `modules -> refactor_check` is now explicit

`CodeRefactor` computes functional score through `check_candidate_code`. The dependency is explicit
in Tach and names the Check domain directly.

## Implemented

- Added `src/robofactor/json_value.py` as the dependency-free home of the recursive `Json` alias.
- Updated pure check/data modules to import `Json` from `robofactor.json_value`.
- Removed the old broad JSON re-export.
- Removed the behavior-input adapter; training examples now parse directly into `BehaviorTest`.
- Tightened `tach.toml`:
  - `robofactor.types` is no longer a utility module.
  - `robofactor.config` is visible only to `robofactor.main` and `robofactor.ui`.
  - `robofactor.utils` is visible only to `robofactor.data`.
  - `robofactor.data` explicitly depends on `robofactor.checks`, `robofactor.data._internal`, and
    `robofactor.utils`.
  - `robofactor.data._internal` is visible only to `robofactor.data`.
- `robofactor.refactor_check`, `modules`, `signatures`, `ui`, and `main` declare their current
  `types`, `config`, and Check edges explicitly.
  - `robofactor.types` has a Tach interface matching its public schema exports.
- Added explicit impossible-result branches in `refactor_check` and `modules._compute_functional_score`
  so focused type checking sees total returns.

## Recommendations Not Implemented

### Split `types.py` by schema ownership

`types.py` is still broad. It now has an explicit Tach interface, but the name remains generic and
the module mixes code analysis reports, refactoring plans, artifacts, quality metrics,
recommendations, and assessment models. A future pass should consider a more precise DSPy schema
module if the surface keeps growing.

### Decide whether `modules -> refactor_check` should stay

The current dependency is honest and acyclic, but the domain direction deserves a design decision.
`modules` depends on `refactor_check` for candidate quality and behavior facts. That is acceptable
while `CodeRefactor` owns assessment inputs. If more callers need the same facts, keep the boundary
in `refactor_check` instead of letting modules import lower-level check internals.

### Consider a public data facade

Source modules now import `robofactor.data.examples` through the data facade. Keep any future data
types private unless they name a distinct domain concept.

### Do not add pydeps/import-linter yet

Tach already enforces the important constraints in this repo. Adding another import tool would add
configuration surface before there is a gap Tach cannot cover. Revisit only if a future check needs
layer contracts or import-style assertions Tach cannot express.

## Validation

- Passed: `make check`
- Passed: `make domain-language`
- Passed: `make architecture`
- Passed: `uv run tach check --exact --output json`
- Passed: `uv run tach check-external`
- Passed: `uv run tach map -o -`
- Passed: `uv run tach map --direction dependents -o -`
- Passed: `uv run tach show --mermaid -o -`
- Passed: `uv run ruff check src/robofactor/refactor_check.py src/robofactor/checks/behavior.py src/robofactor/modules/code_refactor.py`
- Passed: `uv run ruff check src/robofactor/checks/behavior.py src/robofactor/modules/code_refactor.py src/robofactor/json_value.py src/robofactor/types.py src/robofactor/checks/model.py src/robofactor/data/examples.py src/robofactor/utils/load_json.py`
- Passed: `uv run ty check src/robofactor/checks/behavior.py src/robofactor/refactor_check.py src/robofactor/modules/code_refactor.py`
- Passed: `uv run pytest tests/test_checks.py tests/test_refactor_check.py tests/modules/test_code_refactor_functional_score.py tests/data/test_examples.py tests/utils/test_load_json.py tests/data_internal/test_collectors.py tests/data_internal/test_parsers.py`

## Residual Risk

The worktree is shared with other workers and changed during this pass. The assessment and edits
track the final local state observed by Worker 4, not a clean baseline. Full `make check` may still
surface unrelated concurrent issues in files outside this import-direction slice after other workers
continue editing.
