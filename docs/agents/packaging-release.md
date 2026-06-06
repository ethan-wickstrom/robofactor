# Packaging and Release

Use this file when a task changes build metadata, package data, entry points, publishing, or release automation.

## Packaging

- Use `uv build` for package builds.
- Build both sdist and wheel for distribution verification.
- Use `uv build --sdist` only when intentionally checking source distributions.
- Use `uv build --wheel` only when intentionally checking wheels.
- Define CLIs in `[project.scripts]`.
- Smoke-test package entry points with `uv run <command>`.
- Verify package data, `py.typed`, entry points, imports, and version metadata after packaging changes.
- Publish only with `uv publish`.
- Do not add credentials, tokens, passwords, or private index secrets to source files.
- Do not add `python -m build`, twine commands, or setup.py release commands.

## Release Workflow

The GitHub release workflow builds and publishes to PyPI on published releases. It installs uv, installs Python 3.14, installs Deno 2.x, syncs locked dependencies, runs `make check`, validates a `vX.Y.Z` tag, builds the package, and publishes with trusted publishing.

Keep local packaging instructions aligned with `.github/workflows/publish.yml`.
