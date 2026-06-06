# Python Tooling

Use this file when a task touches Python versions, environments, dependencies, lockfiles, or command invocation.

## Versions and Environments

- Declare supported Python versions in `project.requires-python`.
- Pin the local development interpreter with `.python-version`.
- Install interpreters with `uv python install` or `uv python install <version>`.
- Set or update the repo-local pin with `uv python pin <version>`.
- Run all project commands through `uv run`.
- Sync environments with `uv sync`.
- Do not use `python -m venv`, `virtualenv`, pyenv, shell activation, global Python installs, or manual interpreter downloads.
- Do not modify `.venv` manually.
- Do not commit `.venv`.

## Dependencies and Lockfiles

- Add runtime dependencies with `uv add <package>`.
- Add development dependencies with `uv add --dev <package>`.
- Add named development group dependencies with `uv add --group <group> <package>`.
- Add optional feature dependencies with `uv add --optional <extra> <package>`.
- Add local editable dependencies with `uv add --editable <path>`.
- Add workspace dependencies through `tool.uv.sources` with `{ workspace = true }`.
- Remove dependencies with `uv remove <package>`.
- Remove development dependencies with `uv remove --dev <package>`.
- Remove grouped dependencies with `uv remove --group <group> <package>`.
- Remove optional dependencies with `uv remove --optional <extra> <package>`.
- Update resolution with `uv lock`.
- Check lockfile freshness with `uv lock --check`.
- Sync all feature and development dependencies for verification with `uv sync --all-extras --all-groups`.
- Use `uv sync --locked --all-extras --all-groups` in CI.
- Never edit `uv.lock` by hand.
- Never use `uv pip install <package>` to add a dependency to a managed project.

## Running Code and Tools

- Run scripts with `uv run <script.py>`.
- Run modules with `uv run python -m <module>`.
- Run project CLIs with `uv run <command> ...`.
- Run ambiguous commands with `uv run -- <command> ...`.
- Run one-off invocation dependencies with `uv run --with <package> <command> ...`.
- Run one-off tools with `uvx <tool> ...` or `uv tool run <tool> ...`.
- Add repeatedly used tools to a development dependency group and run them with `uv run`.
- For standalone scripts with their own dependencies, add inline metadata with `uv add --script <script.py> <package>` and run them with `uv run <script.py>`.
- Do not use bare `python`, `python3`, `pip`, `pip3`, global tool installs, or shell-activated environments in final commands.

## Dev Shells, Containers, and Editors

- Devcontainers, Dockerfiles, Nix shells, and bootstrap scripts must install or provide `uv`.
- They must not install project dependencies through pip, Poetry, Pipenv, or global tool installers.
- They must use `uv sync --locked --all-extras --all-groups` for reproducible setup.
- Editor settings should point to the project `.venv` created by `uv`.
- Editor integrations should use Ruff for linting and formatting and ty for type checking.
- Do not configure duplicate editor formatters, linters, or type checkers.
