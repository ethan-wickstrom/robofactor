# Robofactor

Robofactor helps software engineers review, check, and apply behavior-preserving Python refactors.

## Language

**Review**:
An engineer-facing summary of refactoring opportunities in code.
_Avoid_: Note, observation

**Check**:
A deterministic verification step that decides whether a proposed refactor preserved required behavior.
_Avoid_: Probe, binary judgment, acceptance

**Apply**:
The action of writing a checked refactor back to the source file.
_Avoid_: Write-back decision, acceptance action

**AppliedChange**:
A source file that was updated after checks passed.
_Avoid_: Apply result

**ApplyFailed**:
A source file left unchanged because checks failed or could not run.
_Avoid_: Apply result

**FunctionSignature**:
The public function name and callable parameter shape that must stay compatible during a single-function refactor.
_Avoid_: Public function contract

**BehaviorTest**:
An input and expected output that states behavior the refactor must preserve.
_Avoid_: Harness case, oracle row

**ComparisonCheck**:
A source-versus-refactored execution using the same call. The check passes only when both outcomes match.
_Avoid_: Differential probe

**GeneratedComparison**:
A comparison check synthesized near existing behavior tests to search for counterexamples.
_Avoid_: Adversarial probe

**QualityCheck**:
Static feedback from Python tools such as Ruff, ty, AST metrics, and Rope.
_Avoid_: Binary quality judgment, quality review

**CheckedRefactor**:
A generated refactor after Robofactor has produced behavior and quality check facts for it.
_Avoid_: Evaluation result, check result

**QualityAssessment**:
An engineer-facing assessment of whether a refactor is safe to apply, needs revision, or is unsafe.
_Avoid_: Binary quality judgment, quality review

**RopeProject**:
Rope's view of a Python project used for safe structural refactoring previews and project-aware analysis.
_Avoid_: Rope cache, IDE state
