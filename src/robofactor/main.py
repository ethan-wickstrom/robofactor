"""
Main entry point for the command-line interface (CLI) of the refactoring tool.
"""

from pathlib import Path
from typing import Annotated

import dspy
import mlflow
import typer
from returns.result import Failure, Success
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

from . import config, ui, utils
from .analysis import extract_python_code
from .dspy_modules import CodeRefactor, RefactoringEvaluator
from .models import TestCase
from .evaluation import evaluate_refactored_code
from .training.training_loader import load_training_data

app = typer.Typer()


def _setup_environment(tracing: bool, mlflow_uri: str, mlflow_experiment: str) -> Console:
    """Configures warnings, MLflow, and returns a rich Console."""
    utils.suppress_pydantic_warnings()
    console = Console()
    if tracing:
        console.print(f"[bold yellow]MLflow tracing enabled. URI: {mlflow_uri}[/bold yellow]")
        mlflow.set_tracking_uri(mlflow_uri)
        _ = mlflow.set_experiment(mlflow_experiment)
        _ = mlflow.autolog()
    return console


def _load_or_compile_model(
    optimizer_path: Path,
    optimize: bool,
    console: Console,
    prompt_llm: dspy.LM,
    task_llm: dspy.LM,
) -> dspy.Module:
    """Loads an optimized DSPy model or compiles a new one."""
    refactorer = CodeRefactor()
    self_correcting_refactorer = dspy.Refine(
        module=refactorer,
        reward_fn=RefactoringEvaluator(),
        threshold=config.REFINEMENT_THRESHOLD,
        N=config.REFINEMENT_COUNT,
    )

    if optimize or not optimizer_path.exists():
        console.print(
            "[yellow]No optimized model found or --optimize set. Running optimization...[/yellow]"
        )
        teleprompter = dspy.MIPROv2(
            metric=RefactoringEvaluator(),
            prompt_model=prompt_llm,
            task_model=task_llm,
            auto="light",
            num_threads=8,
        )
        teleprompter.compile(
            refactorer, trainset=list(load_training_data()), requires_permission_to_run=False
        )
        console.print(f"Optimization complete. Saving to {optimizer_path}...")
        self_correcting_refactorer.save(str(optimizer_path), save_program=True)
    else:
        console.print(f"Loading optimized model from {optimizer_path}...")
        self_correcting_refactorer = dspy.load(str(optimizer_path))
        console.print("[green]Optimized model loaded successfully![/green]")

    return self_correcting_refactorer


def _run_refactoring_on_file(
    console: Console, refactorer: dspy.Module, script_path: Path, write: bool
):
    """Reads a file, runs the refactoring process, and displays results."""
    console.print(Rule(f"[bold magenta]Refactoring {script_path.name}[/bold magenta]"))
    source_code = script_path.read_text(encoding="utf-8")

    console.print(
        Panel(
            Syntax(source_code, "python", theme=config.RICH_SYNTAX_THEME, line_numbers=True),
            title=f"[bold]Original Code: {script_path.name}[/bold]",
            border_style="blue",
        )
    )

    refactor_example = dspy.Example(code_snippet=source_code, test_cases=[]).with_inputs(
        "code_snippet"
    )
    prediction = refactorer(**refactor_example.inputs())
    ui.display_refactoring_process(console, prediction)

    refactored_code = extract_python_code(prediction.refactored_code)
    raw_tests = refactor_example.get("test_cases", [])
    tests = [TestCase(**tc) for tc in raw_tests] if raw_tests else []

    evaluation = evaluate_refactored_code(refactored_code, tests)

    match evaluation:  # type: ignore[reportMatchNotExhaustive]
        case Success(eval_data):
            ui.display_evaluation_results(console, eval_data)
            if write:
                console.print(
                    f"[yellow]Writing refactored code back to {script_path.name}...[/yellow]"
                )
                _ = script_path.write_text(refactored_code, encoding="utf-8")
                console.print(f"[green]Refactoring of {script_path.name} complete.[/green]")
        case Failure(error_message):
            console.print(
                Panel(
                    f"[bold red]Evaluation Failed:[/bold red]\n{error_message}",
                    border_style="red",
                )
            )
            if write:
                console.print(
                    "[bold yellow]Skipping write-back due to evaluation failure.[/bold yellow]"
                )


@app.command()
def main(
    path: Annotated[
        Path | None,
        typer.Argument(
            help="Path to the Python file to refactor.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    self_refactor: bool = typer.Option(
        False, "--dog-food", help="Self-refactor the script you are running."
    ),
    write: bool = typer.Option(
        False, "--write", help="Write the refactored code back to the file."
    ),
    optimize: bool = typer.Option(
        False, "--optimize", help="Force re-optimization of the DSPy model."
    ),
    task_llm_model: str = typer.Option(
        config.DEFAULT_TASK_LLM,
        "--task-llm",
        help="Model for the main refactoring task.",
    ),
    prompt_llm_model: str = typer.Option(
        config.DEFAULT_PROMPT_LLM,
        "--prompt-llm",
        help="Model for generating prompts during optimization.",
    ),
    tracing: bool = typer.Option(True, "--tracing/--no-tracing", help="Enable MLflow tracing."),
    mlflow_uri: str = typer.Option(
        config.DEFAULT_MLFLOW_TRACKING_URI,
        "--mlflow-uri",
        help="MLflow tracking server URI.",
    ),
    mlflow_experiment: str = typer.Option(
        config.DEFAULT_MLFLOW_EXPERIMENT_NAME,
        "--mlflow-experiment",
        help="MLflow experiment name.",
    ),
):
    """A DSPy-powered tool to analyze, plan, and refactor Python code."""
    console = _setup_environment(tracing, mlflow_uri, mlflow_experiment)

    task_llm = dspy.LM(task_llm_model, max_tokens=config.TASK_LLM_MAX_TOKENS)
    prompt_llm = dspy.LM(prompt_llm_model, max_tokens=config.PROMPT_LLM_MAX_TOKENS)
    dspy.configure(lm=task_llm)

    refactorer = _load_or_compile_model(
        config.OPTIMIZER_FILENAME, optimize, console, prompt_llm, task_llm
    )

    target_path: Path | None = None
    if self_refactor:
        target_path = Path(__file__)
        console.print(Rule("[bold magenta]Self-Refactoring Mode[/bold magenta]"))
    elif path:
        target_path = path

    if target_path:
        _run_refactoring_on_file(console, refactorer, target_path, write)
    else:
        console.print(
            "[bold red]Error:[/bold red] Please provide a path to a file or use --dog-food."
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
