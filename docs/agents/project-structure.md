# Project Structure

Use this file when a task changes package layout, import boundaries, metadata, or configuration tables.

## Layout

- Use a `src/` layout for all packages unless moving files would break public import paths or generated paths.
- Keep runtime code under `src/robofactor/`.
- Keep tests under `tests/` unless package-internal tests are required for distribution.
- Keep domain vocabulary in `CONTEXT-MAP.md` and the relevant `CONTEXT.md` files.

## Package Metadata

- Put runtime package metadata in `[project]`.
- Put build backend configuration in `[build-system]`.
- Put command-line entry points in `[project.scripts]`.
- Put optional feature dependencies in `[project.optional-dependencies]`.
- Put local development dependency groups in `[dependency-groups]`.
- Put uv-specific dependency sources in `[tool.uv.sources]`.
- Put Ruff config in `[tool.ruff]`, `[tool.ruff.lint]`, and `[tool.ruff.format]`.
- Put pytest config in `[tool.pytest.ini_options]`.
- Put ty config in `[tool.ty]`.

## Architecture Layers

The current package keeps deep modules under `src/robofactor/` and checks their dependency direction with Tach:

- `checks/`: behavior preservation package with a small public facade.
- `checks/model.py`: check data, function signatures, outcomes, and reports.
- `checks/signature.py`: Python function signature parsing and top-level shape rules.
- `checks/behavior.py`: explicit behavior, source/refactor, and generated comparison checks.
- `checks/quality.py`: Ruff, ty, AST, and suppression-comment quality checks.
- `checks/engine.py`: orchestration for `check_candidate` and `check_refactor`.
- `refactor_check.py`: DSPy-facing CheckedRefactor adapter with explicit candidate and source/refactor check entry points.
- `refactoring.py`: apply workflow and Rope project inspection.
- `modules/` and `signatures/`: DSPy refactoring program.
- `data/`: training examples and JSON parsing.
- `main.py` and `ui.py`: CLI and terminal presentation.

Keep behavior checks and the Apply workflow independent of terminal presentation.
External callers should import check behavior through `robofactor.checks`; Tach keeps the implementation modules private and codifies the facade as the public interface.
Do not add compatibility modules or old import-path adapters; this repo has no external API consumers.
