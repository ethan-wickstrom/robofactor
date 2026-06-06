# Robofactor Agent Instructions

Robofactor is a Python refactoring tool that helps software engineers review, check, and apply behavior-preserving refactorings.

## Essentials

- Use `uv` for Python tooling. Run project commands with `uv run`, add dependencies with `uv add`, and never use bare `python`, `pip`, or shell-activated environments.
- Python is `3.14`; keep `project.requires-python`, `.python-version`, and `[tool.ty.environment]` aligned.
- Deno `2.x` is required for DSPy's Pyodide-backed Python sandbox.
- Run `make check` before completion when a change affects code, packaging, or CI. Use focused Makefile targets while iterating.
- Before domain-sensitive work, read `CONTEXT-MAP.md` and the relevant `CONTEXT.md`.

## Ownership Model

- Treat Robofactor as fully owned internal code with zero external API consumers.
- Do not preserve backwards compatibility, old import paths, legacy names, or fallback behavior unless a current feature explicitly requires it.
- Prefer deletion and direct replacement over adapters, aliases, compatibility layers, optional mode switches, and hidden defaults.
- Make states explicit with narrow functions, required inputs, discriminated data when useful, and early returns.
- Use new tools or libraries when they clearly simplify the system and earn their dependency cost.

## Task-Specific Instructions

- Python tooling, dependency management, and local environments: [docs/agents/python-tooling.md](docs/agents/python-tooling.md)
- Project layout and metadata conventions: [docs/agents/project-structure.md](docs/agents/project-structure.md)
- Code quality, Ruff, ty, and architecture rules: [docs/agents/code-quality.md](docs/agents/code-quality.md)
- Domain language guard and naming workflow: [docs/agents/domain-language.md](docs/agents/domain-language.md)
- Pytest conventions: [docs/agents/testing.md](docs/agents/testing.md)
- Verification gates and Makefile targets: [docs/agents/verification.md](docs/agents/verification.md)
- Packaging, build, and release workflow: [docs/agents/packaging-release.md](docs/agents/packaging-release.md)
- Domain documentation workflow: [docs/agents/domain.md](docs/agents/domain.md)
- GitHub issue workflow: [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md)
- Triage labels: [docs/agents/triage-labels.md](docs/agents/triage-labels.md)

## Agent skills

### Issue tracker

Issues and PRDs live in this repo's GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Multi-context layout: `CONTEXT-MAP.md` at the repo root points to per-context `CONTEXT.md` files. See `docs/agents/domain.md`.
Run `make domain-language` after changing domain names, signatures, public models, prompts, or glossary entries.
