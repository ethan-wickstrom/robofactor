---
title: Strategy Confidence Notes
tags:
  - architecture
  - domain-language
  - behavior-preservation
  - research-notes
category: compound-engineering
status: active
updated: 2026-06-06
---

# Strategy Confidence Notes

This file records evidence, hypotheses, loopholes, fixes, and confidence for the current compounding engineering strategy.

## Strategy Under Test

1. Codify Robofactor's domain language with `CONTEXT.md`, `docs/agents/domain-language.md`, and `tests/test_domain_language.py`.
2. Keep `robofactor.checks` as the small public facade for behavior-preservation checks.
3. Hide check implementation detail in deeper modules: `model`, `signature`, `behavior`, `quality`, and `engine`.

## Tool Catalog

- Local repo tools: `rg`, `fd`, `wc`, `sed`, `jj`, `make`, `uv`, `ruff`, `ty`, `pytest`, and `tach`.
- Context7: current docs for pytest, Tach, Ruff, and tool-specific behavior.
- Exa and web search: primary or near-primary sources for module design, information hiding, ubiquitous language, and architecture-check tooling.
- Verification gates: `make domain-language`, `make check`, and focused tests for behavior-preservation changes.

## Hypothesis Tree

- H1: The domain-language guard creates compounding value by turning naming taste into an automated check.
  - Initial confidence: 0.62
  - Current confidence: 0.78
  - Counter-hypothesis: A substring-based guard will create false confidence, miss variants, and annoy future contributors.
- H2: The behavior-check package split improves locality while preserving the public `robofactor.checks` interface.
  - Initial confidence: 0.68
  - Current confidence: 0.86
  - Counter-hypothesis: The split increases surface area and may hide coupling unless Tach boundaries and tests improve.
- H3: The current strategy is incomplete until the guard and package split are documented as reusable patterns.
  - Initial confidence: 0.70
  - Current confidence: 0.76
  - Counter-hypothesis: Documentation without enforcement is not compounding; it becomes stale.
- H4: A better next step may be to deepen tests around the generated comparison engine instead of adding more architecture rules.
  - Initial confidence: 0.45
  - Current confidence: 0.53
  - Counter-hypothesis: Architecture guardrails eliminate more future bug classes than additional examples right now.

## Evidence Log

- 2026-06-06: Local `make check` passed after adding the domain-language guard and splitting `checks`.
- 2026-06-06: Local vocabulary search showed legacy terms only in `tests/test_domain_language.py`.
- 2026-06-06: Exa search was attempted and failed because the Exa account exceeded its credits limit.
- 2026-06-06: Context7 and official Tach docs confirmed that `exact = true` and `tach check --exact` fail on unused dependency declarations, utility modules are importable without explicit declarations, and public interfaces prevent imports from implementation details.
- 2026-06-06: `uv run tach check --exact` found stale dependency declarations in `tach.toml`; tightening the config made `make architecture` pass.
- 2026-06-06: Broadening the domain-language guard found stale apply-language prose in `docs/agents/project-structure.md`; renaming it made `make domain-language` pass.
- 2026-06-06: Context7 and official Hypothesis docs support generated counterexample checks with shrinking, while also confirming the remaining limit: counterexample quality depends on the strategy and execution model.
- 2026-06-06: Adding a Tach interface for `robofactor.checks` made the facade explicit while keeping implementation modules private.
- 2026-06-06: `make check` passed after the stricter vocabulary guard, exact Tach config, and facade interface changes.

## Open Loopholes

- The language guard is still syntactic; it cannot detect misleading synonyms or new vague terms without adding them to the forbidden list.
- The guard excludes reference files but lacks a documented rule for adding future reference files.
- `robofactor.checks` intentionally exposes the report model; revisit the exact facade surface before declaring a stable external API.
- Generated comparison checks still rely on JSON-serializable behavior and may miss side effects, exceptions that should be equivalent, or non-JSON return values.

## Fix Log

- Added compacted matching in `tests/test_domain_language.py` so CamelCase, snake_case, kebab-case, and spaced variants are caught.
- Expanded the domain-language guard to scan project docs, scripts, `tach.toml`, and GitHub YAML files.
- Made `make architecture` run `uv run tach check --exact`.
- Removed stale Tach dependency declarations and modeled the internal `robofactor.checks` package direction.
- Added a Tach interface for the public `robofactor.checks` facade.
- Documented exact Tach validation in `docs/agents/code-quality.md` and `docs/agents/verification.md`.

## Current Assessment

I am not claiming 1.0 confidence. The strategy is now evidence-backed and meaningfully stronger, with current confidence around 0.86 for the architecture-and-language direction. The next highest-yield improvement is to make generated behavior comparison less JSON-only and more explicit about supported function shapes.
