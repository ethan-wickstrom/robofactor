from __future__ import annotations

import importlib
import importlib.util
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src" / "robofactor"
README_PATH = PROJECT_ROOT / "README.md"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
MAKEFILE_PATH = PROJECT_ROOT / "Makefile"

@dataclass(frozen=True)
class ProjectMeta:
    name: str
    description: str

@dataclass(frozen=True)
class ModuleApi:
    module: str
    signatures: tuple[str, ...]

@dataclass(frozen=True)
class ProjectContext:
    meta: ProjectMeta
    cli_help: str | None
    pyproject_text: str
    makefile_text: str | None
    modules: tuple[ModuleApi, ...]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")

def _list_source_modules(directory: Path) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    return tuple(p for p in directory.glob("*.py") if p.name != "__init__.py")

def _read_makefile_optional() -> str | None:
    try:
        return MAKEFILE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None

def _capture_cli_help_optional() -> str | None:
    # Best-effort: returns None on any problem
    try:
        from typer.testing import CliRunner

        module: ModuleType = importlib.import_module("robofactor.main")
        app_obj = getattr(module, "app", None)
        if not isinstance(app_obj, typer.Typer):
            return None

        runner = CliRunner()
        result = runner.invoke(app_obj, ["--help"], catch_exceptions=False)
        return result.stdout if result.exit_code == 0 else None
    except Exception:
        return None


def _parse_pyproject_meta(text: str) -> ProjectMeta:
    data = tomllib.loads(text)
    meta = data.get("project", {})
    name = str(meta.get("name", "robofactor")).strip() or "robofactor"
    desc = str(meta.get("description", "")).strip()
    return ProjectMeta(name=name, description=desc)

def _format_installation(makefile_text: str | None) -> str:
    if makefile_text and "uv " in makefile_text:
        return (
            "```bash\n"
            "# Install (prod)\n"
            "uv sync --no-dev\n\n"
            "# Install (dev)\n"
            "uv sync --all-groups\n\n"
            "# Run CLI\n"
            "uv run robofactor --help\n"
            "```"
        )
    return (
        "```bash\n"
        "pip install .\n\n"
        "# Run CLI\n"
        "python -m robofactor.main --help\n"
        "```"
    )

def _format_cli_usage(cli_help: str | None) -> str:
    if not cli_help:
        return "CLI is available via `robofactor --help`."
    return f"```text\n{cli_help.strip()}\n```"

def _format_api_section(mods: Iterable[ModuleApi]) -> str:
    lines: list[str] = []
    for mod in mods:
        if not mod.signatures:
            continue
        lines.append(f"- {mod.module}")
        lines.extend(f"  - `{sig}`" for sig in mod.signatures)
    return "\n".join(lines) if lines else "(API signatures discovered automatically)."


def _analyze_modules(paths: Iterable[Path]) -> tuple[ModuleApi, ...]:
    """Loads function_extraction dynamically and returns discovered API signatures."""
    fe_path = SRC_DIR / "function_extraction.py"
    spec = importlib.util.spec_from_file_location("rf_function_extraction", fe_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load function_extraction module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parse_python_source = getattr(module, "parse_python_source", None)
    format_function_signature = getattr(module, "format_function_signature", None)
    if parse_python_source is None or format_function_signature is None:
        raise AttributeError(
            "robofactor.function_extraction must define parse_python_source and format_function_signature"
        )

    from returns.result import Failure

    modules: list[ModuleApi] = []
    for path in paths:
        text = _read_text(path)
        parsed_result = parse_python_source(text, module_name=path.stem)

        # Accept either a Result[...] or a plain value
        if isinstance(parsed_result, Failure):
            raise parsed_result.failure()
        funcs = parsed_result.unwrap() if hasattr(parsed_result, "unwrap") else parsed_result
        signatures = tuple(format_function_signature(func) for func in funcs)
        modules.append(ModuleApi(module=path.stem, signatures=signatures))

    return tuple(modules)


def _build_context() -> ProjectContext:
    py_text = _read_text(PYPROJECT_PATH)
    meta = _parse_pyproject_meta(py_text)
    makefile_text = _read_makefile_optional()
    cli_help = _capture_cli_help_optional()
    paths = _list_source_modules(SRC_DIR)
    modules = _analyze_modules(paths)
    return ProjectContext(
        meta=meta,
        cli_help=cli_help,
        pyproject_text=py_text,
        makefile_text=makefile_text,
        modules=modules,
    )

def _render_readme(ctx: ProjectContext) -> str:
    title = ctx.meta.name.strip() or "robofactor"
    description = ctx.meta.description.strip()
    sections: list[tuple[str, str]] = [
        ("Overview", description or "The robot who refactors."),
        ("Installation", _format_installation(ctx.makefile_text)),
        ("CLI", _format_cli_usage(ctx.cli_help)),
        ("API", _format_api_section(ctx.modules)),
        (
            "Development",
            "- Lint: `uv run ruff check src tests`\n"
            "- Format: `uv run ruff format src tests && uv run isort src tests`\n"
            "- Type-check: `uv run mypy src`\n"
            "- Tests: `uv run pytest`\n",
        ),
    ]
    return _build_markdown(title, description, sections)

def _build_markdown(title: str, description: str, sections: list[tuple[str, str]]) -> str:
    toc_lines = [f"- [{name}](#{name.lower().replace(' ', '-')})" for name, _ in sections]
    body_parts = [
        f"# {title}",
        "",
        description,
        "",
        "## Contents",
        *toc_lines,
    ]
    for name, content in sections:
        body_parts.extend(("", f"## {name}", "", content))
    return "\n".join(part for part in body_parts if part is not None)


app = typer.Typer(add_completion=False, no_args_is_help=False)

OUTPUT_OPTION = typer.Option(README_PATH, "--output", "-o", help="Output README path")
DRY_RUN_OPTION = typer.Option(False, "--dry-run", help="Print to stdout instead of writing")

@app.command()
def main(
    output: Path = OUTPUT_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Generate README.md deterministically."""
    console = Console()
    console.print("[dim]Analyzing project and generating README...[/dim]")
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console,
        ):
            ctx = _build_context()
        content = _render_readme(ctx)

        if dry_run:
            console.print(content)
            return

        _write_text(output, content)
        console.print(f"[green]README written to {output}[/green]")
    except Exception as exc:
        console.print(f"[red]Failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
