"""Rich-based UI for displaying refactoring process and results."""

from difflib import unified_diff

import dspy
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import config
from .evaluation import EvaluationResult
from .types import CodeAnalysisReport, RefactoredArtifact, RefactoringPlanModel


def _section(title: str, renderable, color: str | None = None, expand: bool = True) -> Panel:
    """Create a styled panel for a UI section."""
    color = color or config.UI_COLORS["section"]
    return Panel(
        renderable,
        title=f"[bold {color}]{title}[/bold {color}]",
        border_style=color,
        expand=expand,
    )


def _score_style(score: float) -> str:
    """Return color style based on score value."""
    if score >= 0.9:
        return config.UI_COLORS["success"]
    elif score >= 0.7:
        return config.UI_COLORS["warning"]
    return config.UI_COLORS["error"]


def _truncate_list(items: list[str] | None, limit: int, label: str = "items") -> Text:
    """Truncate a list with a summary line if needed."""
    items = items or []
    text = Text()

    if len(items) <= limit:
        for item in items:
            text.append(f"• {item}\n")
    else:
        for item in items[:limit]:
            text.append(f"• {item}\n")
        text.append(f"\n[dim]+{len(items) - limit} more {label}[/dim]")

    return text if text.plain else Text("[dim]None[/dim]")


def _build_summary_table(
    prediction: dspy.Prediction, result: EvaluationResult | None = None
) -> Table:
    """Build a compact summary table of key metrics."""
    table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    table.add_column(style="bold")
    table.add_column()

    report = prediction.analysis_report
    plan = prediction.plan

    table.add_row("Function:", result.func_name if result else "Pending")
    table.add_row("Opportunities:", str(len(report.opportunities or [])))
    table.add_row("Refactoring Steps:", str(len(plan.steps)))

    if result:
        fc = result.functional_check
        if fc.total_tests > 0:
            style = _score_style(fc.passed_tests / fc.total_tests)
            table.add_row(
                "Tests Passing:", f"[{style}]{fc.passed_tests}/{fc.total_tests}[/{style}]"
            )

        avg_quality = (
            sum(
                [
                    result.quality_metrics.linting.score,
                    result.quality_metrics.complexity.score,
                    result.quality_metrics.typing.score,
                    result.quality_metrics.documentation.score,
                ]
            )
            / 4
        )
        style = _score_style(avg_quality)
        table.add_row("Avg Quality:", f"[{style}]{avg_quality:.2%}[/{style}]")

    return table


def _build_analysis_text(report: CodeAnalysisReport) -> Text:
    """Build analysis text from report."""
    text = Text()
    text.append("Purpose: ", style="bold")
    text.append(f"{report.purpose}\n\n")
    text.append("Complexity: ", style="bold")
    text.append(f"{report.complexity}\n\n")

    if report.dependencies:
        text.append("Dependencies: ", style="bold")
        text.append(", ".join(report.dependencies))
        text.append("\n\n")

    text.append(report.summary)

    return text


def _build_opportunities_table(report: CodeAnalysisReport) -> Table:
    """Build a table of refactoring opportunities."""
    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("Category", style=config.UI_COLORS["accent"], no_wrap=True)
    table.add_column("Description")
    table.add_column("Expected Benefit", style=config.UI_COLORS["info"])

    for opp in report.opportunities or []:
        table.add_row(opp.category.value, opp.description, opp.expected_benefit or "—")

    return table


def _build_plan_table(plan: RefactoringPlanModel) -> Table:
    """Build a table of refactoring steps."""
    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("#", justify="right", width=3, style=config.UI_COLORS["accent"])
    table.add_column("Description")
    table.add_column("Focus", no_wrap=True, style=config.UI_COLORS["info"])
    table.add_column("Success Criteria", style="dim")

    for i, step in enumerate(plan.steps, 1):
        table.add_row(str(i), step.description, step.focus.value, step.success_criteria or "—")

    return table


def _build_explanation_text(artifact: RefactoredArtifact) -> Text:
    """Build explanation text from artifact."""
    text = Text(artifact.explanation)

    if artifact.key_changes:
        text.append("\n\nKey Changes:\n", style="bold")
        for change in artifact.key_changes:
            text.append(f"• {change}\n", style=config.UI_COLORS["success"])

    return text


def _generate_diff(original: str, refactored: str) -> str | None:
    """Generate unified diff between original and refactored code."""
    diff_lines = list(
        unified_diff(
            original.splitlines(keepends=True),
            refactored.splitlines(keepends=True),
            fromfile="original.py",
            tofile="refactored.py",
            lineterm="",
        )
    )
    return "".join(diff_lines) if diff_lines else None


def display_refactoring_process(
    console: Console,
    prediction: dspy.Prediction,
    original_code: str | None = None,
    show_diff: bool = False,
) -> None:
    """Display the LLM's refactoring process with rich formatting.

    Args:
        console: Rich console for output
        prediction: DSPy prediction containing refactoring results
        original_code: Original source code for diff generation
        show_diff: Whether to display unified diff
    """
    console.print(Rule("[bold magenta]Refactoring Process[/bold magenta]"))

    # Summary
    console.print(_section("Summary", _build_summary_table(prediction), expand=False))

    # Analysis
    console.print(_section("Analysis", _build_analysis_text(prediction.analysis_report)))

    # Opportunities (as table if available)
    if prediction.analysis_report.opportunities:
        console.print(
            _section(
                "Refactoring Opportunities",
                _build_opportunities_table(prediction.analysis_report),
            )
        )

    # Plan
    console.print(_section("Refactoring Plan", _build_plan_table(prediction.plan)))

    # Diff (optional)
    if show_diff and original_code:
        diff = _generate_diff(original_code, prediction.artifact.code.code)
        if diff:
            console.print(
                _section(
                    "Changes (Diff)",
                    Syntax(diff, "diff", theme=config.RICH_SYNTAX_THEME),
                    color=config.UI_COLORS["info"],
                )
            )

    # Final code
    console.print(
        _section(
            "Refactored Code",
            Syntax(
                prediction.artifact.code.code,
                "python",
                theme=config.RICH_SYNTAX_THEME,
                line_numbers=True,
            ),
        )
    )

    # Explanation
    console.print(_section("Implementation Details", _build_explanation_text(prediction.artifact)))


def display_evaluation_results(
    console: Console, result: EvaluationResult, verbose: bool = False
) -> None:
    """Display evaluation results with color-coded metrics.

    Args:
        console: Rich console for output
        result: Evaluation result containing quality metrics
        verbose: Show full details including all issues
    """
    console.print(Rule("[bold yellow]Evaluation Results[/bold yellow]"))

    quality = result.quality_metrics
    func_check = result.functional_check

    # Metrics table
    table = Table(show_header=True, header_style="bold", expand=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")

    # Functional tests
    if func_check.total_tests > 0:
        score = func_check.passed_tests / func_check.total_tests
        style = _score_style(score)
        icon = "✓" if score == 1.0 else "✗" if score == 0 else "~"
        table.add_row(
            "Functional Tests",
            f"[{style}]{icon} {func_check.passed_tests}/{func_check.total_tests}[/{style}]",
        )
    else:
        table.add_row("Functional Tests", "[dim]N/A (no tests)[/dim]")

    # Quality metrics
    for label, score in [
        ("Linting", quality.linting.score),
        ("Complexity", quality.complexity.score),
        ("Type Coverage", quality.typing.score),
        ("Documentation", quality.documentation.score),
    ]:
        style = _score_style(score)
        table.add_row(label, f"[{style}]{score:.2%}[/{style}]")

    console.print(table)

    # Issues and warnings
    limit = None if verbose else config.UI_TRUNCATE_LIMIT

    if quality.linting.issues:
        issues_text = _truncate_list(
            quality.linting.issues, limit or len(quality.linting.issues), "issues"
        )
        console.print(
            _section(
                "Linting Issues",
                issues_text,
                color=config.UI_COLORS["warning"],
            )
        )

    if quality.complexity.warnings:
        warnings_text = _truncate_list(
            quality.complexity.warnings,
            limit or len(quality.complexity.warnings),
            "warnings",
        )
        console.print(
            _section(
                "Complexity Warnings",
                warnings_text,
                color=config.UI_COLORS["warning"],
            )
        )


__all__ = [
    "display_evaluation_results",
    "display_refactoring_process",
]
