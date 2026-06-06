# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT-MAP.md`** at the repo root — it points at one `CONTEXT.md` per context. Read each one relevant to the topic.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.
- Also check `src/<context>/docs/adr/` for context-scoped decisions.

If `CONTEXT-MAP.md` or a referenced `CONTEXT.md` doesn't exist, the domain-doc setup is incomplete. Create the missing glossary file with `grill-with-docs` before running skills that depend on domain language.

If an ADR directory doesn't exist, proceed silently. ADRs are created lazily only when a real architectural decision needs one.

## File structure

Multi-context repo:

```
/
├── CONTEXT-MAP.md
├── docs/adr/                          ← system-wide decisions
└── src/
    ├── <context-a>/
    │   ├── CONTEXT.md
    │   └── docs/adr/                  ← context-specific decisions
    └── <context-b>/
        ├── CONTEXT.md
        └── docs/adr/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in the relevant `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `grill-with-docs`).

Run `make domain-language` after changing domain names, signatures, public models, prompts, or glossary entries. The guard turns the most important naming decisions into a test instead of leaving them for human review.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_
