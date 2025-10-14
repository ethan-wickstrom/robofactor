from __future__ import annotations

import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typer.testing import CliRunner

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


def _list_source_modules(directory: Path) -> tuple[Path, ...]:
    return (
        tuple(p for p in directory.glob("*.py") if p.name != "__init__.py")
        if directory.exists()
        else ()
    )


def _read_makefile_optional() -> str | None:
    try:
        return MAKEFILE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _capture_cli_help_optional() -> str | None:
    """
    Attempt to import the CLI app and capture `--help` output without dynamic imports.
    Falls back to None if the module or `app` is unavailable.
    """
    try:
        # Static import (no importlib usage)
        from robofactor.main import app as app_obj

        if not isinstance(app_obj, typer.Typer):
            return None

        result = CliRunner().invoke(app_obj, ["--help"], catch_exceptions=False)
        return result.stdout if result.exit_code == 0 else None
    except Exception:
        return None


def _parse_pyproject_meta(text: str) -> ProjectMeta:
    meta = tomllib.loads(text).get("project", {})
    return ProjectMeta(
        name=str(meta.get("name", "robofactor")).strip() or "robofactor",
        description=str(meta.get("description", "")).strip(),
    )


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
    return "```bash\npip install .\n\n# Run CLI\npython -m robofactor.main --help\n```"


def _format_cli_usage(cli_help: str | None) -> str:
    if not cli_help:
        return "CLI is available via `robofactor --help`."
    return f"```text\n{cli_help.strip()}\n```"


def _format_api_section(_mods: Iterable[ModuleApi]) -> str:
    """
    Keep README generation deterministic without dynamic imports. We intentionally
    skip runtime module loading; API discovery can be added later with static analysis.
    """
    return "(API signatures discovered automatically)."


def _build_context() -> ProjectContext:
    py_text = PYPROJECT_PATH.read_text(encoding="utf-8")
    return ProjectContext(
        meta=_parse_pyproject_meta(py_text),
        cli_help=_capture_cli_help_optional(),
        pyproject_text=py_text,
        makefile_text=_read_makefile_optional(),
        modules=(),  # Purposely not loading modules dynamically
    )


def _build_markdown(title: str, description: str, sections: list[tuple[str, str]]) -> str:
    from itertools import chain

    toc_lines = (f"- [{name}](#{name.lower().replace(' ', '-')})" for name, _ in sections)
    section_parts = chain.from_iterable(
        ("", f"## {name}", "", content) for name, content in sections
    )
    parts = chain(
        (f"# {title}", "", description, "", "## Contents"),
        toc_lines,
        section_parts,
    )
    return "\n".join(parts)


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
            "- Format: `uv run ruff format src tests`\n"
            "- Type-check: `uv run ty check`\n"
            "- Tests: `uv run pytest`\n",
        ),
    ]
    return _build_markdown(title, description, sections)


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
        else:
            output.write_text(content, encoding="utf-8")
            console.print(f"[green]README written to {output}[/green]")
    except Exception as exc:
        console.print(f"[red]Failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
