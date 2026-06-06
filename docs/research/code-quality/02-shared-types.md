---
title: Shared Type Definition Inventory
tags:
  - code-quality
  - domain-language
  - typing
  - research-notes
category: compound-engineering
status: complete
updated: 2026-06-06
worker: 2/8
---

# Shared Type Definition Inventory

This note inventories Robofactor's type aliases, dataclasses, named tuples, Pydantic models,
DSPy signature fields, protocols, and domain models. The goal is consolidation only where the
domain concept is the same, not where two objects merely have a similar shape.

## Coverage

Searched `src`, `tests`, and `scripts` with `rg`, `fd`, `ast-grep`, and a Python AST inventory
run through `uv run python`. No `TypedDict`, `TypeAlias`, or `NewType` definitions were found.

Type surface as a tree:

```text
robofactor
|- json_value.Json
|- types
|  |- PythonCode
|  |- CodeAnalysisReport -> RefactoringOpportunity -> OpportunityCategory
|  |- RefactoringPlanModel -> PlanStep -> PlanStepFocus
|  |- RefactoredArtifact
|  |- QualityMetrics -> LintingReport, ComplexityReport, TypingReport, DocumentationReport
|  `- AssessmentRecommendation -> RecommendationPriority, QualityAssessment
|- checks.model
|  |- FunctionSignature -> FunctionParameter -> ParameterKind
|  |- BehaviorTest
|  |- CheckReport -> CheckSummary, CheckFailure -> CheckName, CallOutcome
|  `- QualityReport
|- refactor_check.CheckedRefactor
|- refactoring.AppliedChange, ApplyFailed, RopeFile
|- data._internal.parsers.Parser
`- signatures.* DSPy signature classes
```

## Inventory

| Location | Definitions | Assessment |
| --- | --- | --- |
| `src/robofactor/json_value.py` | `Json` recursive type alias | Correct shared home for JSON-serializable values used by data loading and behavior checks. This avoids coupling check models to DSPy-facing schemas. |
| `src/robofactor/types.py` | `PythonCode`, Pydantic models, `StrEnum` categories/priorities | Mixed but intentional facade for DSPy structured inputs and outputs. `Json` now lives only in `json_value.py`. |
| `src/robofactor/checks/model.py` | `ParameterKind`, `CheckName`, `FunctionParameter`, `FunctionSignature`, `BehaviorTest`, `Returned`, `Raised`, `CallOutcome`, `CheckFailure`, `CheckSummary`, `QualityReport`, `CheckReport` | Cohesive deterministic check model. Training examples now parse directly into `BehaviorTest` values, so behavior-preservation evidence has one shared shape. |
| `src/robofactor/data/_internal/parsers.py` | `Parser[T]` protocol and parser classes | Internal parsing infrastructure, not domain model. Keep local unless another parser family appears outside data loading. |
| `src/robofactor/refactor_check.py` | `CheckedRefactor` dataclass | App-facing Check aggregation. Its `behavior` field uses shared `CheckSummary` rather than a duplicate count model. |
| `src/robofactor/refactoring.py` | `AppliedChange`, `ApplyFailed`, `RopeFile` | Apply and Rope inspection models are separate domain concepts. No shared type candidate found. |
| `src/robofactor/signatures/*.py` | `CodeAnalysis`, `RefactoringPlan`, `RefactoredCode`, `FinalAssessment` DSPy signatures | Signature fields correctly reference shared Pydantic/DSPy schema models from `types.py`. Do not merge with deterministic check dataclasses. |
| `scripts/generate_readme.py` | `ProjectMeta`, `ProjectContext` | Script-local documentation generation types. No runtime sharing needed. |

## Implemented Consolidations

1. `CheckSummary` now owns `pass_rate`.
   `CheckedRefactor.behavior` uses `CheckSummary` directly, removing the duplicate count model and
   aligning app-facing checks with the check model.

2. Training examples now construct `BehaviorTest` directly.
   The intermediate behavior-input model was removed because it only duplicated args, kwargs, and
   expected output without adding a lifecycle concept.

## Critical Assessment

The previous behavior-input model and `BehaviorTest` were the closest duplicate. They are now
collapsed into `BehaviorTest`, with training data assigning stable `training-{index}` IDs at the
JSON boundary.

`QualityReport` and `QualityMetrics` also look duplicate by field names, but they are different
interfaces. `QualityReport` is a deterministic check fact with tool pass/fail booleans and raw
diagnostics. `QualityMetrics` is a DSPy-facing quality summary with Pydantic validation and prompt
serialization. The adapter in `refactor_check.quality_metrics_from_report` is the right seam.

`types.py` remains broad. It is currently a schema facade rather than a pure "types" module:
Python code values, analysis reports, plans, artifacts, metrics, recommendations, and assessments
all live there. That is acceptable for now because callers import these as DSPy structured schema
types. If more non-DSPy domain types accumulate, split the DSPy schema surface into a more precise
module name instead of adding more unrelated aliases to `types.py`.

## Recommendations Not Implemented

1. Keep `Json` sourced from `json_value.py`.
   Future code should import `Json` from `robofactor.json_value`; the old `types.py` re-export has
   been removed.

2. Consider first-class aliases for `AssessmentRecommendation.area` and `QualityAssessment.outcome`
   only when those value sets are reused outside their Pydantic models. They are domain concepts,
   but adding aliases today would be anticipatory rather than consolidating duplication.

3. Consider a `CheckFailureReason` or enum only if `ApplyFailed.reason` grows more callers. Today
   the string values are local to `apply_change`, so introducing a shared type would add surface
   without much leverage.
