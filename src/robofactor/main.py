from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, Protocol, runtime_checkable

import dspy
import mlflow
import typer
from dspy.teleprompt.gepa.gepa import GEPAFeedbackMetric
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback
from returns.result import Failure, Result, Success
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

from . import config, ui
from .data import examples, models
from .evaluation import EvaluationResult, evaluate_refactored_code
from .modules.code_refactor import CodeRefactor
from .types import PythonCode

app = typer.Typer()


def _get_functional_score(eval_data: EvaluationResult) -> float:
    """Calculate functional test pass rate, defaulting to 1.0 if no tests."""
    fc = eval_data.functional_check
    return fc.passed_tests / fc.total_tests if fc.total_tests > 0 else 1.0


def _calculate_reward_score(example: dspy.Example, prediction: dspy.Prediction) -> float:
    """Calculate reward score from functional test results."""
    artifact = getattr(prediction, "artifact", None)
    if not artifact or not artifact.code.code.strip():
        logging.debug("Reward score 0.0: No artifact or empty code in prediction")
        return 0.0

    test_cases = getattr(example, "test_cases", [])
    eval_result = evaluate_refactored_code(artifact.code, test_cases)

    match eval_result:
        case Success(eval_data):
            score = _get_functional_score(eval_data)
            logging.debug(
                f"Reward score {score}: {eval_data.functional_check.passed_tests}/"
                f"{eval_data.functional_check.total_tests} tests passed"
            )
            return score
        case Failure(error_msg):
            logging.debug(f"Reward score 0.0: Evaluation failed - {error_msg}")
            return 0.0
        case _:
            return 0.0


def _reward_fn(inputs: dict[str, Any], prediction: dspy.Prediction) -> float:
    """Adapter for reward function matching code snippets to training examples."""
    match examples.get_examples():
        case Failure():
            return 0.0
        case Success(train_set):
            code_snippet = inputs["code_snippet"]
            example = next((ex for ex in train_set if ex.code_snippet == code_snippet), None)

            if not example:
                logging.warning(f"No matching example found for code_snippet: {code_snippet!r}")
                if os.environ.get("ROBOFACTOR_DEV_MODE", "0") == "1":
                    raise ValueError(f"Missing example for code_snippet: {code_snippet!r}")
                return 0.0

            return _calculate_reward_score(example, prediction)
        case _:
            return 0.0


@runtime_checkable
class _SupportsTestCase(Protocol):
    args: list[Any]
    kwargs: dict[str, Any]
    expected_output: Any


class _GEPARefactorMetric(GEPAFeedbackMetric):
    """GEPA metric using functional correctness scoring."""

    def __call__(
        self,
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: DSPyTrace | None = None,
        pred_name: str | None = None,
        pred_trace: DSPyTrace | None = None,
    ) -> float | ScoreWithFeedback:
        return _calculate_reward_score(gold, pred)


def _setup_environment(tracing: bool, mlflow_uri: str, mlflow_experiment: str) -> Console:
    """Configure MLflow and return Rich console."""
    console = Console()
    if tracing:
        console.print(f"[bold yellow]MLflow tracing enabled. URI: {mlflow_uri}[/bold yellow]")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(mlflow_experiment)
        mlflow.dspy.autolog(log_compiles=True, log_traces=True)
    return console


def _load_or_compile_model(
    optimizer_path: Path, optimize: bool, console: Console, reflection_lm: dspy.LM
) -> dspy.Module:
    """Load optimized DSPy model or compile a new one if needed."""
    refactorer = CodeRefactor()
    self_correcting_refactorer = dspy.Refine(
        module=refactorer,
        reward_fn=_reward_fn,
        threshold=config.REFINEMENT_THRESHOLD,
        N=config.REFINEMENT_COUNT,
    )

    if not optimize and optimizer_path.exists():
        console.print(f"Loading optimized model from {optimizer_path}...")
        self_correcting_refactorer = dspy.load(str(optimizer_path))
        console.print("[green]Optimized model loaded successfully![/green]")
        return self_correcting_refactorer

    console.print(
        "[yellow]No optimized model found or --optimize set. Running optimization...[/yellow]"
    )
    teleprompter = dspy.GEPA(
        metric=_GEPARefactorMetric(),
        auto="light",
        reflection_lm=reflection_lm,
        num_threads=8,
        track_stats=True,
        track_best_outputs=True,
        add_format_failure_as_feedback=True
    )

    match examples.get_examples():
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

    return self_correcting_refactorer


def _render_original(console: Console, script_path: Path, source_code: str) -> None:
    """Display original source code with syntax highlighting."""
    console.print(
        Panel(
            Syntax(source_code, "python", theme=config.RICH_SYNTAX_THEME, line_numbers=True),
            title=f"[bold]Original Code: {script_path.name}[/bold]",
            border_style="blue",
        )
    )


def _to_test_case(
    test_case: models.TestCase | Mapping[str, Any] | _SupportsTestCase,
) -> models.TestCase:
    """Convert test case from various formats to TestCase model."""
    match test_case:
        case models.TestCase():
            return test_case
        case {"args": args, "kwargs": kwargs, "expected_output": expected}:
            return models.TestCase(args=args, kwargs=kwargs, expected_output=expected)
        case _SupportsTestCase(args=args, kwargs=kwargs, expected_output=expected):
            return models.TestCase(args=args, kwargs=kwargs, expected_output=expected)
        case _:
            raise TypeError(f"Unsupported test case type: {type(test_case)!r}")


def _build_tests(
    raw_tests: Iterable[models.TestCase | Mapping[str, Any] | _SupportsTestCase] | None,
) -> list[models.TestCase]:
    """Convert test cases to list of TestCase models."""
    return [] if raw_tests is None else list(map(_to_test_case, raw_tests))


def _safe_extract_refactored_code(prediction: dspy.Prediction) -> Result[PythonCode, str]:
    """Extract and validate non-empty Python code from prediction."""
    if artifact := getattr(prediction, "artifact", None):
        return (
            Success(artifact.code)
            if artifact.code.code.strip()
            else Failure("Extracted refactored code is empty.")
        )
    else:
        return Failure("No refactored code produced by the model.")


def _evaluate_and_maybe_write(
    console: Console,
    refactored_code: PythonCode,
    tests: list[models.TestCase],
    script_path: Path,
    write: bool,
) -> None:
    """Evaluate refactored code and optionally write to file if successful."""
    match evaluate_refactored_code(refactored_code, tests):
        case Success(eval_data):
            ui.display_evaluation_results(console, eval_data)
            if write:
                console.print(
                    f"[yellow]Writing refactored code back to {script_path.name}...[/yellow]"
                )
                script_path.write_text(refactored_code.code, encoding="utf-8")
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


def _run_refactoring_on_file(
    console: Console, refactorer: dspy.Module, script_path: Path, write: bool
) -> None:
    """Execute refactoring workflow: read, refactor, evaluate, and optionally write."""
    console.print(Rule(f"[bold magenta]Refactoring {script_path.name}[/bold magenta]"))
    source_code = script_path.read_text(encoding="utf-8")
    _render_original(console, script_path, source_code)

    refactor_example = dspy.Example(
        code_snippet=source_code,
        test_cases=[],
    ).with_inputs("code_snippet")

    prediction = refactorer(**refactor_example.inputs())
    ui.display_refactoring_process(console, prediction)

    match _safe_extract_refactored_code(prediction):
        case Success(refactored_code):
            tests = _build_tests(refactor_example.get("test_cases", []))
            _evaluate_and_maybe_write(console, refactored_code, tests, script_path, write)
        case Failure(msg):
            console.print(
                Panel(
                    f"[bold yellow]No usable refactored code:[/bold yellow]\n{msg}",
                    border_style="yellow",
                )
            )
            if write:
                console.print(
                    "[bold yellow]Skipping write-back due to missing refactored code.[/bold yellow]"
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
    tracing: bool = typer.Option(False, "--tracing/--no-tracing", help="Enable MLflow tracing."),
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

    match (self_refactor, path):
        case (True, _):
            console.print(Rule("[bold magenta]Self-Refactoring Mode[/bold magenta]"))
            _run_refactoring_on_file(console, refactorer, Path(__file__), write)
        case (False, Path() as p) if p.is_file() and p.suffix == ".py":
            _run_refactoring_on_file(console, refactorer, p, write)
        case (False, Path() as p):
            console.print(f"[red]Provided path '{p}' is not a Python (.py) file.[/red]")
        case _:
            console.print(
                "[bold red]Error:[/bold red] Please provide a path to a file or use --dog-food."
            )
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
