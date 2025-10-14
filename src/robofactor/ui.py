import dspy
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import analysis, config
from .evaluation import EvaluationResult


def display_refactoring_process(console: Console, prediction: dspy.Prediction) -> None:
    """Displays the LLM's refactoring process using rich components."""
    report = prediction.analysis_report
    plan = prediction.plan
    artifact = prediction.artifact

    analysis_text = Text()
    analysis_text.append("Purpose: ", style="bold")
    analysis_text.append(report.purpose)
    analysis_text.append("\nComplexity: ", style="bold")
    analysis_text.append(report.complexity)
    if report.dependencies:
        analysis_text.append("\nDependencies: ", style="bold")
        analysis_text.append(", ".join(report.dependencies))
    analysis_text.append("\n\n")
    analysis_text.append(report.summary)

    if report.opportunities:
        analysis_text.append("\n\nRefactoring Opportunities:\n", style="bold")
        for opportunity in report.opportunities:
            prefix = f"- ({opportunity.category.value}) "
            analysis_text.append(f"{prefix}{opportunity.description}")
            if opportunity.expected_benefit:
                analysis_text.append(f" — {opportunity.expected_benefit}")
            analysis_text.append("\n")

    console.print(
        Panel(
            analysis_text,
            title="[bold cyan]Analysis[/bold cyan]",
            expand=False,
        )
    )

    plan_text = Text()
    plan_text.append("Objective: ", style="bold")
    plan_text.append(plan.objective)
    plan_text.append("\n\n")
    for i, step in enumerate(plan.steps, 1):
        plan_text.append(f"{i}. {step.description}")
        plan_text.append(f" [{step.focus.value}]")
        if step.success_criteria:
            plan_text.append(f"\n   Success Criteria: {step.success_criteria}")
        plan_text.append("\n")
    console.print(Panel(plan_text, title="[bold cyan]Refactoring Plan[/bold cyan]"))

    extracted_code = analysis.extract_python_code(artifact.code).code
    console.print(
        Panel(
            Syntax(
                extracted_code,
                "python",
                theme=config.RICH_SYNTAX_THEME,
                line_numbers=True,
            ),
            title="[bold cyan]Final Refactored Code[/bold cyan]",
        )
    )

    explanation_text = Text(artifact.explanation)
    if artifact.key_changes:
        explanation_text.append("\n\nKey Changes:\n", style="bold")
        for change in artifact.key_changes:
            explanation_text.append(f"- {change}\n")
    console.print(
        Panel(
            explanation_text,
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
