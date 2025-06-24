#!/usr/bin/env python3
"""Show staged/unstaged git diffs in smart Markdown format."""

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

import typer
from returns.result import Result, Success, Failure

app = typer.Typer(help="Show staged/unstaged git diffs in Markdown")

DiffOptions = Sequence[str]
DiffText = str


class DiffMode(str, Enum):
    staged = "staged"
    unstaged = "unstaged"
    both = "both"


class DiffType(str, Enum):
    stat = "stat"
    patch = "patch"


@dataclass(frozen=True)
class GitError:
    command: tuple[str, ...]
    return_code: int
    stderr: str


def build_diff_opts(is_stat: bool, context: int, word_diff: bool) -> DiffOptions:
    """Return git-diff options for stat-only or patch with context."""
    if is_stat:
        return ("--stat",)
    base: tuple[str, ...] = ("--minimal", f"-U{context}", "--color=never")
    return base + (("--word-diff",) if word_diff else ())


def run_git_diff(opts: DiffOptions) -> Result[DiffText, GitError]:
    """
    Runs `git` with the given options.
    - return code >1 ⇒ Failure(GitError)
    - return code ==1 ⇒ Success("")  (no changes)
    - return code ==0 ⇒ Success(stdout)
    """
    command = tuple(["git", *opts])
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rc = proc.returncode
    if rc > 1:
        return Failure(GitError(command=command, return_code=rc, stderr=proc.stderr))
    if rc == 1:
        return Success("")  # no changes
    return Success(proc.stdout)


def format_section(title: str, diff_text: str) -> str:
    """Render a Markdown section for a diff under the given title."""
    header = f"### {title}\n\n"
    if not diff_text.strip():
        return header + "_No changes_\n\n"
    fence = "```diff" if diff_text.startswith(("diff --", "@@")) else "```text"
    body = f"{fence}\n{diff_text.rstrip()}\n```\n\n"
    return header + body


@app.command()
def main(
    which: Annotated[
        DiffMode,
        typer.Option(
            "-m",
            "--mode",
            help="Which diffs to show: 'staged', 'unstaged', or 'both' (default: both)",
        ),
    ] = DiffMode.both,
    stat: Annotated[
        bool,
        typer.Option("--stat", help="Show only stat for all sections"),
    ] = False,
    word_diff: Annotated[
        bool,
        typer.Option("--word-diff", help="Enable word-level patch diff"),
    ] = False,
    context: Annotated[
        int,
        typer.Option("-c", "--context", help="Context lines (ignored if --stat)"),
    ] = 3,
    staged_type: Annotated[
        DiffType | None,
        typer.Option("--staged-type", help="Override mode for staged changes"),
    ] = None,
    unstaged_type: Annotated[
        DiffType | None,
        typer.Option("--unstaged-type", help="Override mode for unstaged changes"),
    ] = None,
) -> None:
    """
    Show staged/unstaged git diffs formatted as Markdown.
    Errors are printed to stderr and exit with the git return code.
    """

    def section_is_stat(section_mode: DiffMode) -> bool:
        if stat:
            return True
        override = staged_type if section_mode is DiffMode.staged else unstaged_type
        return override is not None and override is DiffType.stat

    staged_opts: DiffOptions = ("diff", "--cached") + tuple(
        build_diff_opts(section_is_stat(DiffMode.staged), context, word_diff)
    )
    unstaged_opts: DiffOptions = ("diff",) + tuple(
        build_diff_opts(section_is_stat(DiffMode.unstaged), context, word_diff)
    )

    # Staged
    staged_diff = ""
    if which is not DiffMode.unstaged:
        res = run_git_diff(staged_opts)
        if isinstance(res, Failure):
            err = res.failure()
            typer.secho(
                f"Error running git {' '.join(err.command)}:\n{err.stderr}",
                err=True,
                fg="red",
            )
            raise typer.Exit(err.return_code)
        staged_diff = res.unwrap()

    # Unstaged
    unstaged_diff = ""
    if which is not DiffMode.staged:
        res = run_git_diff(unstaged_opts)
        if isinstance(res, Failure):
            err = res.failure()
            typer.secho(
                f"Error running git {' '.join(err.command)}:\n{err.stderr}",
                err=True,
                fg="red",
            )
            raise typer.Exit(err.return_code)
        unstaged_diff = res.unwrap()

    # Print results
    if which in (DiffMode.both, DiffMode.staged) and staged_diff:
        print(format_section("Staged", staged_diff))
    if which in (DiffMode.both, DiffMode.unstaged) and unstaged_diff:
        print(format_section("Unstaged", unstaged_diff))


if __name__ == "__main__":
    app()
