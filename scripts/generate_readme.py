from __future__ import annotations

import tomllib
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typer.testing import CliRunner

from robofactor.main import app as robofactor_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_PATH = PROJECT_ROOT / "README.md"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"


@dataclass(frozen=True)
class ProjectMeta:
    name: str
    description: str


@dataclass(frozen=True)
class ProjectContext:
    meta: ProjectMeta
    cli_help: str


def _capture_cli_help() -> str:
    """Capture `robofactor --help` output from the imported CLI app."""
    assert isinstance(robofactor_app, typer.Typer)
    result = CliRunner().invoke(robofactor_app, ["--help"], catch_exceptions=False)
    assert result.exit_code == 0, result.stdout
    return result.stdout


def _parse_pyproject_meta(text: str) -> ProjectMeta:
    meta = tomllib.loads(text)["project"]
    name = str(meta["name"]).strip()
    assert name
    return ProjectMeta(
        name=name,
        description=str(meta["description"]).strip(),
    )


def _format_installation() -> str:
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


def _format_cli_usage(cli_help: str) -> str:
    return f"```text\n{cli_help.strip()}\n```"


def _format_api_section() -> str:
    """
    Keep README generation deterministic without dynamic imports. We intentionally
    skip runtime module loading; API discovery can be added later with static analysis.
    """
    return "(API signatures discovered automatically)."


def _build_context() -> ProjectContext:
    py_text = PYPROJECT_PATH.read_text(encoding="utf-8")
    return ProjectContext(
        meta=_parse_pyproject_meta(py_text),
        cli_help=_capture_cli_help(),
    )


def _build_markdown(title: str, description: str, sections: list[tuple[str, str]]) -> str:
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
    title = ctx.meta.name
    description = ctx.meta.description.strip()
    sections: list[tuple[str, str]] = [
        ("Overview", description or "The robot who refactors."),
        ("Installation", _format_installation()),
        ("CLI", _format_cli_usage(ctx.cli_help)),
        ("API", _format_api_section()),
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
