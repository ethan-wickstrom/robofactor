from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Any

import dspy
import mlflow
import typer
from dspy.teleprompt.gepa.gepa import GEPAFeedbackMetric
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace
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
    return (
        eval_data.functional_check.passed_tests / eval_data.functional_check.total_tests
        if eval_data.functional_check.total_tests > 0
        else 1.0
    )


def _calculate_reward_score(example: dspy.Example, prediction: dspy.Prediction) -> float:
    """Calculate reward score from functional test results."""
    if not (artifact := getattr(prediction, "artifact", None)) or not artifact.code.code.strip():
        logging.debug("Reward score 0.0: No artifact or empty code in prediction")
        return 0.0

    eval_result = evaluate_refactored_code(artifact.code, getattr(example, "test_cases", []))

    match eval_result:
        case Success(eval_data):
            score = _get_functional_score(eval_data)
            fc = eval_data.functional_check
            logging.debug(f"Reward score {score}: {fc.passed_tests}/{fc.total_tests} tests passed")
            return score
        case Failure(error_msg):
            logging.debug(f"Reward score 0.0: Evaluation failed - {error_msg}")
    return 0.0


def _reward_fn(inputs: dict[str, Any], prediction: dspy.Prediction) -> float:
    """Adapter for reward function matching code snippets to training examples."""
    match examples.get_examples():
        case Success(train_set):
            code_snippet = inputs["code_snippet"]
            if example := next((ex for ex in train_set if ex.code_snippet == code_snippet), None):
                return _calculate_reward_score(example, prediction)
            logging.warning(f"No matching example found for code_snippet: {code_snippet!r}")
            if os.environ.get("ROBOFACTOR_DEV_MODE", "0") == "1":
                raise ValueError(f"Missing example for code_snippet: {code_snippet!r}")
        case Failure():
            pass
    return 0.0


class _GEPARefactorMetric(GEPAFeedbackMetric):
    """GEPA metric with rich textual feedback for reflection-driven optimization."""

    def _analyze_trace_for_module_feedback(  # noqa: C901
        self, pred_trace: DSPyTrace, pred_name: str | None
    ) -> str:
        """Extract module-specific insights from execution trace."""
        if not pred_trace or not pred_name:
            return ""

        feedback_parts = []

        # Analyze CodeAnalysis module
        if pred_name == "analyzer" or "analyzer" in str(pred_trace):
            if analysis_report := getattr(pred_trace, "report", None):
                if hasattr(analysis_report, "opportunities") and not analysis_report.opportunities:
                    feedback_parts.append(
                        "MODULE ISSUE: CodeAnalysis found no refactoring opportunities. "
                        "Instruction should emphasize identifying code smells, complexity, and improvement areas."
                    )
                if (
                    hasattr(analysis_report, "complexity")
                    and "complex" not in str(analysis_report.complexity).lower()
                ):
                    feedback_parts.append(
                        "MODULE HINT: CodeAnalysis may be underreporting complexity. "
                        "Look for nested loops, long functions, high cyclomatic complexity."
                    )

        # Analyze RefactoringPlan module
        if pred_name == "planner" or "planner" in str(pred_trace):
            if plan := getattr(pred_trace, "plan", None):
                if hasattr(plan, "steps") and len(plan.steps) == 0:
                    feedback_parts.append(
                        "MODULE ISSUE: RefactoringPlan generated no concrete steps. "
                        "Instruction should require actionable, ordered refactoring actions."
                    )
                # Check for constraint violations in plan description
                if hasattr(plan, "objective"):
                    objective_lower = str(plan.objective).lower()
                    if "class" in objective_lower and "add" in objective_lower:
                        feedback_parts.append(
                            "MODULE WARNING: RefactoringPlan may be suggesting adding classes. "
                            "CONSTRAINT: NO NEW CLASSES unless they exist in original. "
                            "Instruction must emphasize preserving structure."
                        )
                    if "restructure" in objective_lower or "reorganize" in objective_lower:
                        feedback_parts.append(
                            "MODULE WARNING: RefactoringPlan suggests major restructuring. "
                            "CONSTRAINT: KEEP STRUCTURE - functions stay functions. "
                            "Focus on incremental improvements: types, docstrings, readability."
                        )

        # Analyze RefactoredCode module (implementer)
        if pred_name == "implementer" or "implementer" in str(pred_trace):
            feedback_parts.append(
                "MODULE CONTEXT: RefactoredCode is the final generator. "
                "If syntax errors occur here, instruction must stress SYNTACTICALLY VALID constraint. "
                "Common issues: missing colons, bad indentation, incomplete refactoring."
            )

        return "\n".join(feedback_parts) if feedback_parts else ""

    def __call__(  # noqa: C901
        self,
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: DSPyTrace | None = None,
        pred_name: str | None = None,
        pred_trace: DSPyTrace | None = None,
    ) -> dspy.Prediction:
        """Return score with detailed feedback on pipeline stage failures."""
        # Analyze trace for module-specific insights
        trace_feedback = self._analyze_trace_for_module_feedback(pred_trace, pred_name)

        # Extract artifact
        if not (artifact := getattr(pred, "artifact", None)):
            base_feedback = (
                f"STAGE: {'RefactoredCode' if pred_name else 'Unknown'}\n"
                "FAILURE: No artifact produced - module failed to generate refactored code.\n"
                "ACTION: Ensure RefactoredCode signature returns RefactoredArtifact with code field."
            )
            return dspy.Prediction(
                score=0.0,
                feedback=f"{base_feedback}\n\n{trace_feedback}"
                if trace_feedback
                else base_feedback,
            )

        if not artifact.code.code.strip():
            base_feedback = (
                f"STAGE: {'RefactoredCode' if pred_name else 'Unknown'}\n"
                "FAILURE: Empty code artifact generated.\n"
                "ACTION: Refactored code must be non-empty. Check if model is generating placeholder text."
            )
            return dspy.Prediction(
                score=0.0,
                feedback=f"{base_feedback}\n\n{trace_feedback}"
                if trace_feedback
                else base_feedback,
            )

        refactored_code = artifact.code
        test_cases = getattr(gold, "test_cases", [])

        # Stage 1: Syntax validation
        from . import analysis

        syntax_result = analysis.check_syntax(refactored_code)
        match syntax_result:
            case Failure(error_msg):
                base_feedback = (
                    f"STAGE: RefactoredCode (syntax validation)\n"
                    f"FAILURE: {error_msg}\n"
                    "ACTION: Ensure generated code is valid Python. "
                    "Check for: missing colons, indentation errors, unclosed brackets, invalid keywords.\n"
                    f"CONSTRAINTS VIOLATED: SYNTACTICALLY VALID requirement from RefactoredCode signature."
                )
                return dspy.Prediction(
                    score=0.0,
                    feedback=f"{base_feedback}\n\n{trace_feedback}"
                    if trace_feedback
                    else base_feedback,
                )
            case Success(_):
                pass

        # Stage 2: Functional correctness
        eval_result = evaluate_refactored_code(refactored_code, test_cases)
        match eval_result:
            case Success(eval_data):
                fc = eval_data.functional_check
                passed = fc.passed_tests
                total = fc.total_tests
                functional_score = passed / total if total > 0 else 1.0

                # Calculate quality component scores
                qm = eval_data.quality_metrics
                linting_score = qm.linting.score
                complexity_score = qm.complexity.score
                typing_score = qm.typing.score
                documentation_score = qm.documentation.score

                # Aggregate quality score (equal weights)
                quality_score = (
                    linting_score + complexity_score + typing_score + documentation_score
                ) / 4.0

                # Multi-objective weighted score: 70% functional, 30% quality
                # Functional correctness is critical (must preserve behavior)
                # Quality improvements are secondary but important for refactoring
                weighted_score = (0.7 * functional_score) + (0.3 * quality_score)

                # Build detailed feedback with multi-objective decomposition
                feedback_parts = [
                    "STAGE: Full Pipeline (multi-objective evaluation)",
                    f"OBJECTIVES: Functional={functional_score:.2f} (70%), Quality={quality_score:.2f} (30%)",
                    f"WEIGHTED SCORE: {weighted_score:.2f}",
                    "",
                    "BREAKDOWN:",
                    f"  - Functional Correctness: {functional_score:.2f} ({passed}/{total} tests passed)"
                    if total > 0
                    else f"  - Functional Correctness: {functional_score:.2f} (no tests)",
                    f"  - Linting Quality: {linting_score:.2f}",
                    f"  - Complexity: {complexity_score:.2f}",
                    f"  - Type Hints: {typing_score:.2f}",
                    f"  - Documentation: {documentation_score:.2f}",
                ]

                # Success/failure analysis
                if functional_score == 1.0:
                    feedback_parts.append(
                        f"\nFUNCTIONAL: ✓ All {total} test(s) passed"
                        if total > 0
                        else "\nFUNCTIONAL: ✓ No tests to run"
                    )
                else:
                    feedback_parts.append(
                        f"\nFUNCTIONAL: ✗ {total - passed}/{total} test(s) failed - behavior diverges"
                    )
                    feedback_parts.append(
                        "ACTION: Ensure PRESERVE ALL FUNCTIONALITY constraint. "
                        "Check if refactoring changed logic, return values, or edge case handling."
                    )

                # Quality issues
                if linting_score < 1.0 and qm.linting.issues:
                    feedback_parts.append(
                        f"LINTING: ✗ Issues found: {', '.join(qm.linting.issues[:3])}"
                    )
                elif linting_score == 1.0:
                    feedback_parts.append("LINTING: ✓ No issues")

                if complexity_score < 0.8:
                    feedback_parts.append(
                        f"COMPLEXITY: ⚠ Score {complexity_score:.2f} indicates high complexity"
                    )
                elif complexity_score >= 0.8:
                    feedback_parts.append(
                        f"COMPLEXITY: ✓ Acceptable (score {complexity_score:.2f})"
                    )

                # Append trace-based module feedback if available
                if trace_feedback:
                    feedback_parts.append(f"\n{trace_feedback}")

                return dspy.Prediction(score=weighted_score, feedback="\n".join(feedback_parts))

            case Failure(error_msg):
                return dspy.Prediction(
                    score=0.0,
                    feedback=(
                        f"STAGE: Evaluation Pipeline\n"
                        f"FAILURE: {error_msg}\n"
                        "ACTION: Code may have runtime errors or evaluation infrastructure failed."
                    ),
                )
            case _:
                return dspy.Prediction(
                    score=0.0,
                    feedback="STAGE: Unknown\nFAILURE: Unexpected evaluation result type.",
                )


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
        add_format_failure_as_feedback=True,
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


def _safe_extract_refactored_code(prediction: dspy.Prediction) -> Result[PythonCode, str]:
    """Extract and validate non-empty Python code from prediction."""
    if not (artifact := getattr(prediction, "artifact", None)):
        return Failure("No refactored code produced by the model.")
    return (
        Success(artifact.code)
        if artifact.code.code.strip()
        else Failure("Extracted refactored code is empty.")
    )


def _evaluate_and_maybe_write(
    console: Console,
    refactored_code: PythonCode,
    tests: list[models.TestCase],
    script_path: Path,
    write: bool,
    verbose: bool = False,
) -> None:
    """Evaluate refactored code and optionally write to file if successful."""
    with console.status("[bold cyan]Evaluating code quality...[/]"):
        result = evaluate_refactored_code(refactored_code, tests)

    match result:
        case Success(eval_data):
            ui.display_evaluation_results(console, eval_data, verbose=verbose)
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
    console: Console,
    refactorer: dspy.Module,
    script_path: Path,
    write: bool,
    show_diff: bool = False,
    verbose: bool = False,
) -> None:
    """Execute refactoring workflow: read, refactor, evaluate, and optionally write."""
    console.print(Rule(f"[bold magenta]Refactoring {script_path.name}[/bold magenta]"))
    source_code = script_path.read_text(encoding="utf-8")
    _render_original(console, script_path, source_code)

    refactor_example = dspy.Example(
        code_snippet=source_code,
        test_cases=[],
    ).with_inputs("code_snippet")

    with console.status("[bold cyan]Refactoring code...[/]"):
        prediction = refactorer(**refactor_example.inputs())

    ui.display_refactoring_process(
        console, prediction, original_code=source_code, show_diff=show_diff
    )

    match _safe_extract_refactored_code(prediction):
        case Success(refactored_code):
            _evaluate_and_maybe_write(console, refactored_code, [], script_path, write, verbose)
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
    show_diff: bool = typer.Option(False, "--show-diff", help="Display unified diff of changes."),
    verbose: bool = typer.Option(
        False, "--verbose", help="Show full details (all issues, warnings, etc.)."
    ),
) -> None:
    """A DSPy-powered tool to analyze, plan, and refactor Python code."""
    console = _setup_environment(tracing, mlflow_uri, mlflow_experiment)

    task_llm = dspy.LM(task_llm_model, max_tokens=config.TASK_LLM_MAX_TOKENS)
    reflection_llm = dspy.LM(prompt_llm_model, max_tokens=config.PROMPT_LLM_MAX_TOKENS)
    dspy.configure(lm=task_llm)

    refactorer = _load_or_compile_model(config.OPTIMIZER_PATH, optimize, console, reflection_llm)

    match (self_refactor, path):
        case (True, _):
            console.print(Rule("[bold magenta]Self-Refactoring Mode[/bold magenta]"))
            _run_refactoring_on_file(console, refactorer, Path(__file__), write, show_diff, verbose)
        case (False, Path() as p) if p.is_file() and p.suffix == ".py":
            _run_refactoring_on_file(console, refactorer, p, write, show_diff, verbose)
        case (False, Path() as p):
            console.print(f"[red]Provided path '{p}' is not a Python (.py) file.[/red]")
        case _:
            console.print(
                "[bold red]Error:[/bold red] Please provide a path to a file or use --dog-food."
            )
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
