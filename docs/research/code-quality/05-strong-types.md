---
title: Strong Type Boundary Assessment
tags:
  - code-quality
  - typing
  - domain-language
  - research-notes
category: compound-engineering
status: active
updated: 2026-06-06
worker: "5/8"
---

# Strong Type Boundary Assessment

Worker 5 focused on weak types: `Any`, broad `object`, casts, and loose dictionaries/lists where
Robofactor already proves a stronger domain type. The rule was to strengthen proven data flow
without changing exception/error-control behavior or crossing other workers' ownership.

## Tool Catalog

- Local inventory: `rg`, `fd`, and a small `uv run python` AST scan over `src` and `tests`.
- Type/lint gates: `uv run ty check`, focused `uv run ruff check ...`, and `uv run tach check --exact`.
- Context7 research:
  - `ty` docs confirmed Python-version configuration and gradual typing/Unknown behavior.
  - Ruff docs confirmed `ruff check` and formatter/linter configuration behavior.
  - Python 3.14 typing docs confirmed `TypeIs` as the right predicate type when a parser proves a JSON subtype.

## Type Flow

```text
training.json
  -> Json
     -> checks.model.BehaviorTest
        -> check_candidate/check_refactor execution

raw Json
  -> Parser[T].parse(Json)
     -> BasicParser TypeIs predicate
     -> ListParser / DictParser
        -> typed training examples
```

## Inventory

| Location | Weak shape found | Assessment |
| --- | --- | --- |
| `src/robofactor/data/examples.py` | Training behavior examples formerly used a duplicate behavior-input shape. | Strong type proven. Training data now parses directly into `BehaviorTest` with JSON args, kwargs, and expected output. |
| `src/robofactor/data/_internal/parsers.py` | Parser input `Any`, `Callable[[Any], bool]`, `field_parsers: dict[str, Parser[Any]]` | Parser input is JSON, not arbitrary Python. Scalar predicates should use `TypeIs[T]`; heterogeneous field parser storage remains a real weak point. |
| `src/robofactor/data/_internal/collectors.py` | `list[Any]` raw items | Strong type proven. Collectors collect JSON arrays. |
| `src/robofactor/main.py` | `_reward_fn(inputs: dict[str, Any])` | Stronger type proven for the used field. DSPy example inputs provide a `code_snippet` source string. |
| `src/robofactor/checks/behavior.py` | `cast("Json", json.loads(output))` | Keep for now. This is an external JSON decoder boundary; replacing it safely needs a runtime JSON validator. |
| `src/robofactor/data/_internal/parsers.py` | `Mapping[str, Parser[object]]`, `dict[str, object]` | Keep for now. This is the heterogeneous field-parser/constructor boundary. A better type requires a schema abstraction, not a blind alias. |
| `src/robofactor/main.py` | DSPy trace arguments typed as `object` | Keep for now. GEPA trace payloads are dynamic external callback data. A typed adapter should be designed separately. |
| `src/robofactor/types.py` | enum `_missing_(value: object)` hooks | Keep. The Python enum hook accepts arbitrary values before normalization. |

## Implemented

- Removed the duplicate behavior-input dataclass and parse training examples directly into `BehaviorTest`.
- Typed parser and collector inputs as `Json`.
- Changed `BasicParser` predicates to `Callable[[Json], TypeIs[T]]` and added typed guards in examples/tests.
- Removed `OptionalParser`; explicit JSON `null` is now parsed as a present invalid value unless a field parser accepts it.
- Narrowed `_reward_fn` inputs to `Mapping[str, str]`.
- Made two `returns.result` handlers explicit enough for `ty` to prove total return paths.

## Critical Assessment

The highest-confidence type strengthening is the `Json -> BehaviorTest` path. It is
domain-backed, already enforced by training data loading, and used by the check engine's JSON-only
execution model. Keeping that path as `Any` made future agents infer that behavior checks can accept
arbitrary Python objects, which is false today.

The parser layer is now better but not finished. `Parser[T].parse(Json)` is the right boundary, and
`TypeIs` makes scalar parsing honest. `DictParser` still stores heterogenous field parser results as
`object` before splatting them into a constructor. That is not a cosmetic issue; it means the parser
abstraction cannot prove that required constructor fields match the parser map. Fixing it properly
requires either a small schema object per parsed domain model or dedicated parsers for training
examples and behavior checks.

The remaining `cast("Json", json.loads(output))` is acceptable only because it is localized at the
interpreter output boundary. If behavior checks start supporting non-JSON return values, this cast
will become misleading and should be replaced by a named outcome decoder.

## Recommendations Not Implemented

1. Replace `DictParser` with typed domain parsers for training examples.
   The generic constructor-splat abstraction is the last meaningful `object` pocket in data loading.

2. Add a runtime `parse_json_value` validator before removing the `json.loads` cast.
   This should preserve current `JSONDecodeError` behavior and add a named failure only if decoded
   values fall outside Robofactor's `Json` alias.

3. Introduce a typed DSPy/GEPA trace adapter.
   Keep dynamic `getattr` and `object` trace payloads at one boundary instead of spreading trace
   assumptions through `_GEPARefactorMetric`.

4. Consider a more precise input model for `_reward_fn`.
   `Mapping[str, str]` is better than `dict[str, Any]`, but a `TypedDict` such as
   `RefactorInputs` would make `code_snippet` required if DSPy callback typing allows it.

## Validation

- Passed: `uv run ty check`
- Passed: `uv run ruff check src/robofactor/checks/behavior.py src/robofactor/data src/robofactor/refactor_check.py src/robofactor/modules/code_refactor.py src/robofactor/main.py tests/data_internal`
- Passed: `uv run tach check --exact`
