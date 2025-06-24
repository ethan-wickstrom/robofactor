
"""
Presentation logic for displaying results in the console.

This module uses the 'rich' library to create user-friendly, formatted
output for the analysis, plan, and evaluation phases.
"""

import dspy
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..parsing import analysis
from . import config
from ..evaluation import EvaluationResult


def display_refactoring_process(console: Console, prediction: dspy.Prediction) -> None:
    """Displays the LLM's refactoring process using rich components."""
    console.print(Panel(prediction.analysis, title="[bold cyan]Analysis[/bold cyan]", expand=False))

    plan_text = Text()
    _ = plan_text.append("Summary: ", style="bold")
    _ = plan_text.append(prediction.refactoring_summary)
    _ = plan_text.append("\n\n")
    for i, step in enumerate(prediction.plan_steps, 1):
        _ = plan_text.append(f"{i}. {step}\n")
    console.print(Panel(plan_text, title="[bold cyan]Refactoring Plan[/bold cyan]"))

    console.print(
        Panel(
            Syntax(
                analysis.extract_python_code(prediction.refactored_code),
                "python",
                theme=config.RICH_SYNTAX_THEME,
                line_numbers=True,
            ),
            title="[bold cyan]Final Refactored Code[/bold cyan]",
        )
    )
    console.print(
        Panel(
            prediction.implementation_explanation,
            title="[bold cyan]Implementation Explanation[/bold cyan]",
        )
    )


def display_evaluation_results(console: Console, result: EvaluationResult) -> None:
    """Displays the evaluation results using rich components."""
    console.print(Rule("[bold yellow]Final Output Evaluation[/bold yellow]"))

    quality = result.quality_scores
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

    table.add_row("Linting Score:", f"{quality.linting_score:.2f}")
    table.add_row("Typing Score:", f"{quality.typing_score:.2f}")
    table.add_row("Docstring Score:", f"{quality.docstring_score:.2f}")
    console.print(table)

    if quality.linting_issues:
        lint_issues_text = Text("\n- ".join(quality.linting_issues))
        console.print(
            Panel(lint_issues_text, title="[yellow]Linting Issues[/yellow]", border_style="yellow")
        )
