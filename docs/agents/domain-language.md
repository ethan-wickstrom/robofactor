---
title: Domain Language Guard
tags:
  - domain-language
  - testing
  - refactoring
  - agent-workflow
category: compound-engineering
---

# Domain Language Guard

Use this file when changing names, public models, prompts, signatures, checks, or documentation.

Robofactor treats domain language as executable design. The glossary in `src/robofactor/CONTEXT.md` defines the preferred terms, and `tests/test_domain_language.py` enforces the most important forbidden legacy terms across source, tests, docs, and project configuration.

## Working Rule

When a term feels awkward, overloaded, or borrowed from another domain, update the glossary and the guard in the same change. Do not rely on review to catch vocabulary drift.

## Local Check

Run the focused guard with:

```bash
make domain-language
```

The full gate also runs the guard through `make test`.
