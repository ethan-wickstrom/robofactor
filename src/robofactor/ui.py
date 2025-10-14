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


def _build_analysis_text(report: CodeAnalysisReport) -> Text:
    """Build analysis text from report."""
    text = Text()
    text.append("Purpose: ", style="bold")
    text.append(report.purpose)
    text.append("\nComplexity: ", style="bold")
    text.append(report.complexity)

    if report.dependencies:
        text.append("\nDependencies: ", style="bold")
        text.append(", ".join(report.dependencies))

    text.append("\n\n")
    text.append(report.summary)

    if report.opportunities:
        text.append("\n\nRefactoring Opportunities:\n", style="bold")
        for opp in report.opportunities:
            text.append(f"- ({opp.category.value}) {opp.description}")
            if opp.expected_benefit:
                text.append(f" — {opp.expected_benefit}")
            text.append("\n")

    return text


def _build_plan_text(plan: RefactoringPlanModel) -> Text:
    """Build plan text from plan object."""
    text = Text()
    text.append("Objective: ", style="bold")
    text.append(plan.objective)
    text.append("\n\n")

    for i, step in enumerate(plan.steps, 1):
        text.append(f"{i}. {step.description} [{step.focus.value}]")
        if step.success_criteria:
            text.append(f"\n   Success Criteria: {step.success_criteria}")
        text.append("\n")

    return text


def _build_explanation_text(artifact: RefactoredArtifact) -> Text:
    """Build explanation text from artifact."""
    text = Text(artifact.explanation)

    if artifact.key_changes:
        text.append("\n\nKey Changes:\n", style="bold")
        for change in artifact.key_changes:
            text.append(f"- {change}\n")

    return text


def display_refactoring_process(console: Console, prediction: dspy.Prediction) -> None:
    """Displays the LLM's refactoring process using rich components."""
    console.print(
        Panel(
            _build_analysis_text(prediction.analysis_report),
            title="[bold cyan]Analysis[/bold cyan]",
            expand=False,
        )
    )

    console.print(
        Panel(
            _build_plan_text(prediction.plan),
            title="[bold cyan]Refactoring Plan[/bold cyan]",
        )
    )

    console.print(
        Panel(
            Syntax(
                prediction.artifact.code.code,
                "python",
                theme=config.RICH_SYNTAX_THEME,
                line_numbers=True,
            ),
            title="[bold cyan]Final Refactored Code[/bold cyan]",
        )
    )

    console.print(
        Panel(
            _build_explanation_text(prediction.artifact),
            title="[bold cyan]Implementation Explanation[/bold cyan]",
        )
    )


def display_evaluation_results(console: Console, result: EvaluationResult) -> None:
    """Displays the evaluation results using rich components."""
    console.print(Rule("[bold yellow]Final Output Evaluation[/bold yellow]"))

    quality = result.quality_metrics
    func_check = result.functional_check

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column()
    table.add_column(style="bold magenta")

    if func_check.total_tests > 0:
        table.add_row(
            "Functional Equivalence:", f"{func_check.passed_tests} / {func_check.total_tests}"
        )
    else:
        table.add_row("Functional Equivalence:", "N/A (no tests)")

    table.add_row("Linting Score:", f"{quality.linting.score:.2f}")
    table.add_row("Typing Score:", f"{quality.typing.score:.2f}")
    table.add_row("Docstring Score:", f"{quality.documentation.score:.2f}")
    table.add_row("Complexity Score:", f"{quality.complexity.score:.2f}")
    console.print(table)

    if quality.linting.issues:
        lint_issues_text = Text("\n".join(f"- {issue}" for issue in quality.linting.issues))
        console.print(
            Panel(lint_issues_text, title="[yellow]Linting Issues[/yellow]", border_style="yellow")
        )
    if quality.complexity.warnings:
        complexity_text = Text("\n".join(f"- {warning}" for warning in quality.complexity.warnings))
        console.print(
            Panel(
                complexity_text,
                title="[yellow]Complexity Warnings[/yellow]",
                border_style="yellow",
            )
        )
