from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import dspy
import mlflow
import typer

if TYPE_CHECKING:
    from dspy.teleprompt.gepa.gepa import DSPyTrace, ScoreWithFeedback
from returns.result import Failure, Success
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

from . import config, ui
from .analysis import extract_python_code
from .data import examples, models
from .evaluation import EvaluationResult, evaluate_refactored_code
from .modules.code_refactor import CodeRefactor

app = typer.Typer()

def _get_functional_score(eval_data: EvaluationResult) -> float:
    total_tests = eval_data.functional_check.total_tests
    passed_tests = eval_data.functional_check.passed_tests
    return (passed_tests / total_tests) if total_tests > 0 else 1.0


def _calculate_reward_score(example: dspy.Example, prediction: dspy.Prediction) -> float:
    """Calculates a reward score for a refactoring prediction."""
    refactored_code = getattr(prediction, "refactored_code", "")
    if not refactored_code:
        return 0.0

    code_to_evaluate = extract_python_code(refactored_code)
    if not code_to_evaluate:
        return 0.0

    test_cases = getattr(example, "test_cases", [])
    eval_result = evaluate_refactored_code(code_to_evaluate, test_cases)

    if isinstance(eval_result, Failure):
        return 0.0

    # For now, we'll just use the functional score as the reward.
    # In the future, we could incorporate quality scores.
    return _get_functional_score(eval_result.unwrap())


def _reward_fn(inputs: dict[str, Any], prediction: dspy.Prediction) -> float:
    """Wrapper to adapt reward function signature."""
    trainset_result = examples.get_examples()
    if isinstance(trainset_result, Failure):
        return 0.0
    train_set = trainset_result.unwrap()
    code_snippet = inputs["code_snippet"]

    # Find the example in the training set that matches the code snippet
    # This is inefficient but necessary given the reward_fn signature
    example = next((ex for ex in train_set if ex.code_snippet == code_snippet), None)

    if example is None:
        return 0.0

    return _calculate_reward_score(example, prediction)

def _metric_fn(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: DSPyTrace | None,
    pred_name: str | None,
    pred_trace: DSPyTrace | None,
) -> float | ScoreWithFeedback:
    """Wrapper to adapt metric function signature."""
    if not gold.test_cases:
        return 0.0
    refactored_code = getattr(pred, "refactored_code", "")
    if not refactored_code:
        return 0.0
    eval_result = evaluate_refactored_code(refactored_code, gold.test_cases)
    if isinstance(eval_result, Failure):
        return 0.0
    return _get_functional_score(eval_result.unwrap())

def _setup_environment(tracing: bool, mlflow_uri: str, mlflow_experiment: str) -> Console:
    """Configures warnings, MLflow, and returns a rich Console."""
    console = Console()
    if tracing:
        console.print(f"[bold yellow]MLflow tracing enabled. URI: {mlflow_uri}[/bold yellow]")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(mlflow_experiment)
        mlflow.dspy.autolog(log_compiles=True, log_traces=True)  # pyright: ignore[reportPrivateImportUsage] mlflow.dspy is lazy-loaded
    return console


def _load_or_compile_model(
    optimizer_path: Path, optimize: bool, console: Console, reflection_lm: dspy.LM
) -> dspy.Module:
    """Loads an optimized DSPy model or compiles a new one."""
    refactorer = CodeRefactor()
    self_correcting_refactorer = dspy.Refine(
        module=refactorer,
        reward_fn=_reward_fn,
        threshold=config.REFINEMENT_THRESHOLD,
        N=config.REFINEMENT_COUNT,
    )

    if optimize or not optimizer_path.exists():
        console.print(
            "[yellow]No optimized model found or --optimize set. Running optimization...[/yellow]"
        )
        teleprompter = dspy.GEPA(
            metric=_metric_fn,
            auto="light",
            reflection_lm=reflection_lm,
            num_threads=8,
        )
        trainset_result = examples.get_examples()
        match trainset_result:
            case Success(trainset):
                teleprompter.compile(refactorer, trainset=trainset)
                console.print(f"Optimization complete. Saving to {optimizer_path}...")
                self_correcting_refactorer.save(str(optimizer_path), save_program=True)
            case Failure(err):
                console.print(
                    Panel(
                        f"[bold red]Failed to load training examples:[/bold red]\n{err}",
                        border_style="red",
                    )
                )
                console.print("[yellow]Proceeding without optimization.[/yellow]")
    else:
        console.print(f"Loading optimized model from {optimizer_path}...")
        self_correcting_refactorer = dspy.load(str(optimizer_path))
        console.print("[green]Optimized model loaded successfully![/green]")

    return self_correcting_refactorer


def _run_refactoring_on_file(
    console: Console, refactorer: dspy.Module, script_path: Path, write: bool
) -> None:
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
    tests = [models.TestCase(**tc) for tc in raw_tests] if raw_tests else []

    evaluation = evaluate_refactored_code(refactored_code, tests)

    match evaluation:
        case Success(eval_data):
            ui.display_evaluation_results(console, eval_data)
            if write:
                console.print(
                    f"[yellow]Writing refactored code back to {script_path.name}...[/yellow]"
                )
                script_path.write_text(refactored_code, encoding="utf-8")
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
        config.DEFAULT_TASK_LLM, "--task-llm", help="Model for the main refactoring task."
    ),
    prompt_llm_model: str = typer.Option(
        config.DEFAULT_PROMPT_LLM,
        "--prompt-llm",
        help="Model for generating prompts during optimization.",
    ),
    tracing: bool = typer.Option(True, "--tracing/--no-tracing", help="Enable MLflow tracing."),
    mlflow_uri: str = typer.Option(
        config.DEFAULT_MLFLOW_TRACKING_URI, "--mlflow-uri", help="MLflow tracking server URI."
    ),
    mlflow_experiment: str = typer.Option(
        config.DEFAULT_MLFLOW_EXPERIMENT_NAME, "--mlflow-experiment", help="MLflow experiment name."
    ),
) -> None:
    """A DSPy-powered tool to analyze, plan, and refactor Python code."""
    console = _setup_environment(tracing, mlflow_uri, mlflow_experiment)

    task_llm = dspy.LM(task_llm_model, max_tokens=config.TASK_LLM_MAX_TOKENS)
    reflection_llm = dspy.LM(prompt_llm_model, max_tokens=config.PROMPT_LLM_MAX_TOKENS)
    dspy.configure(lm=task_llm)

    refactorer = _load_or_compile_model(
        config.OPTIMIZER_FILENAME, optimize, console, reflection_llm
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
