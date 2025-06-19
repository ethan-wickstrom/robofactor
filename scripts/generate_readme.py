#!/usr/bin/env python3
"""
README Generator for robofactor project.

This module uses DSPy to automatically generate comprehensive documentation
by analyzing the codebase structure and synthesizing a well-formatted README.

The implementation follows a pure functional programming paradigm:
- **Stateless & Immutable**: All data is stored in immutable ADTs (dataclasses).
- **Pure Functions**: Core logic consists of pure functions that transform data.
- **Composition**: The generation process is a pipeline of composable functions.
- **Explicit Error Handling**: `Result` type is used to make failures an
  explicit part of the function signature, avoiding exceptions for control flow.
- **Separation of Concerns**: Logic is clearly separated into layers:
  Data (ADTs), Serialization, AI Models (DSPy), Data Acquisition,
  Pipeline Steps, Composition, and I/O.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Any, Callable, TypeVar

import dspy
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# --- Setup Project Path ---
# This allows importing from the `src` directory.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from robofactor.function_extraction import FunctionInfo, parse_python_source
from robofactor.functional_types import Err, Ok, Result

# --- Generic Type Variables for Pipeline ---
T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")


# ============================================================================
# 1. DATA LAYER (ALGEBRAIC DATA TYPES)
# ============================================================================


@dataclass(frozen=True)
class FileAnalysis:
    """Immutable representation of a source file's content and structure."""

    path: Path
    relative_path: str
    content: str
    structure: tuple[FunctionInfo, ...]


@dataclass(frozen=True)
class FileSummary:
    """The result of summarizing a single file."""

    file_path: str
    summary: str


@dataclass(frozen=True)
class Architecture:
    """The result of the overall architecture analysis."""

    overview: str
    component_breakdown: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class UsageGuide:
    """The result of the usage guide generation."""

    installation: str
    usage: str


@dataclass(frozen=True)
class ProjectContext:
    """Immutable representation of the entire project's state for generation."""

    root: Path
    source_analyses: tuple[FileAnalysis, ...]
    config_files: dict[str, str]  # filename -> content

    def get_config(self, filename: str) -> Result[str, str]:
        """Safely retrieve a configuration file's content."""
        content = self.config_files.get(filename)
        return Ok(content) if content is not None else Err(f"Config '{filename}' not found")


@dataclass(frozen=True)
class GenerationState:
    """
    State passed through the generation pipeline, accumulating results.
    This is the "world" object for our functional pipeline, where each step
    produces a new, updated state.
    """

    project_context: ProjectContext
    console: Console
    llm_config: dspy.LM
    # Intermediate results, added by pipeline steps
    summaries: tuple[FileSummary, ...] = ()
    architecture: Architecture | None = None
    usage: UsageGuide | None = None


# ============================================================================
# 2. SERIALIZATION LAYER
# ============================================================================


def _custom_json_encoder(obj: Any) -> Any:
    """A custom encoder to handle dataclasses, enums, and paths for JSON."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def to_json_string(data: Any) -> str:
    """
    Converts a Python object (including dataclasses) to a JSON string.
    This pure function is essential for passing structured data to the LLM.
    """
    return json.dumps(data, default=_custom_json_encoder, indent=2)


# ============================================================================
# 3. DSPy LAYER (AI MODEL SIGNATURES)
# ============================================================================


class SummarizeFile(dspy.Signature):
    """Summarize a Python file's purpose based on its structure."""

    file_path: str = dspy.InputField(desc="The relative path to the Python file.")
    file_structure: str = dspy.InputField(
        desc="A JSON object describing the functions and classes in the file."
    )
    summary: str = dspy.OutputField(
        desc="A concise, one-paragraph summary of the file's main purpose."
    )


class AnalyzeArchitecture(dspy.Signature):
    """Analyze file summaries to describe the project's architecture."""

    file_summaries: str = dspy.InputField(desc="A JSON array of summaries for all project files.")
    architecture_overview: str = dspy.OutputField(
        desc="A high-level paragraph describing the architecture and data flow."
    )
    component_breakdown: list[dict[str, str]] = dspy.OutputField(
        desc="A list of dicts with 'component' and 'description' keys."
    )


class GenerateUsage(dspy.Signature):
    """Generate installation and usage instructions from config files."""

    pyproject_toml: str = dspy.InputField(desc="Content of pyproject.toml.")
    makefile: str = dspy.InputField(desc="Content of Makefile.")
    cli_help_text: str = dspy.InputField(desc="The --help output from the main CLI.")
    installation_instructions: str = dspy.OutputField(
        desc="Markdown-formatted installation instructions."
    )
    usage_instructions: str = dspy.OutputField(
        desc="Markdown-formatted usage instructions with examples."
    )


class AssembleReadme(dspy.Signature):
    """Assemble a complete README from all generated sections."""

    project_name: str = dspy.InputField()
    project_description: str = dspy.InputField()
    installation: str = dspy.InputField()
    usage: str = dspy.InputField()
    architecture_overview: str = dspy.InputField()
    component_breakdown: str = dspy.InputField(
        desc="A markdown-formatted list of system components."
    )
    readme_content: str = dspy.OutputField(desc="The complete, final README.md content.")


# ============================================================================
# 4. DATA ACQUISITION LAYER
# ============================================================================


def read_file(path: Path) -> Result[str, str]:
    """Pure-functional file reading. Wraps I/O in a Result type."""
    try:
        return Ok(path.read_text(encoding="utf-8"))
    except Exception as e:
        return Err(f"Failed to read {path}: {e}")


def analyze_source_file(path: Path, root: Path) -> Result[FileAnalysis, str]:
    """Reads and parses a single Python file into a FileAnalysis ADT."""
    content_result = read_file(path)
    if isinstance(content_result, Err):
        return content_result
    content = content_result.value

    try:
        structure = tuple(parse_python_source(content, module_name=path.name))
        return Ok(
            FileAnalysis(
                path=path,
                relative_path=str(path.relative_to(root)),
                content=content,
                structure=structure,
            )
        )
    except Exception as e:
        return Err(f"Failed to parse {path}: {e}")


def create_project_context(root: Path) -> Result[ProjectContext, str]:
    """
    Builds the complete, immutable project context by reading and analyzing
    all necessary files from disk. This is the primary I/O-bound operation.
    """
    source_dir = root / "src" / "robofactor"
    py_files = [p for p in source_dir.glob("*.py") if p.name != "__init__.py"]

    analyses: list[FileAnalysis] = []
    for file_path in py_files:
        analysis_result = analyze_source_file(file_path, root)
        if isinstance(analysis_result, Err):
            return analysis_result
        analyses.append(analysis_result.value)

    config_files: dict[str, str] = {}
    required_configs = ("pyproject.toml", "Makefile")
    for filename in required_configs:
        content_result = read_file(root / filename)
        if isinstance(content_result, Err):
            return content_result
        config_files[filename] = content_result.value

    return Ok(ProjectContext(root=root, source_analyses=tuple(analyses), config_files=config_files))


# ============================================================================
# 5. LOGIC LAYER (PIPELINE STEPS)
# ============================================================================

# Each function in this layer represents one major step in the generation process.
# They are pure functions that take the current state and return a new state
# wrapped in a Result, making them perfect for pipeline composition.


def step_summarize_files(state: GenerationState) -> Result[GenerationState, str]:
    """Pipeline step to summarize each source file using an LLM."""
    summarizer = dspy.Predict(SummarizeFile)
    summaries: list[FileSummary] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=state.console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            "Summarizing source files...", total=len(state.project_context.source_analyses)
        )
        for analysis in state.project_context.source_analyses:
            progress.update(task, advance=1, description=f"Summarizing {analysis.relative_path}")
            try:
                prediction = summarizer(
                    file_path=analysis.relative_path,
                    file_structure=to_json_string(analysis.structure),
                )
                summaries.append(
                    FileSummary(file_path=analysis.relative_path, summary=prediction.summary)
                )
            except Exception as e:
                return Err(f"LLM failed to summarize {analysis.relative_path}: {e}")

    # Create a new state with the summaries added by using `replace`.
    return Ok(replace(state, summaries=tuple(summaries)))


def step_analyze_architecture(state: GenerationState) -> Result[GenerationState, str]:
    """Pipeline step to analyze the overall project architecture."""
    state.console.print("Analyzing project architecture...")
    architect = dspy.Predict(AnalyzeArchitecture)
    try:
        prediction = architect(file_summaries=to_json_string(state.summaries))
        architecture = Architecture(
            overview=prediction.architecture_overview,
            component_breakdown=tuple(prediction.component_breakdown),
        )
        return Ok(replace(state, architecture=architecture))
    except Exception as e:
        return Err(f"LLM failed to analyze architecture: {e}")


def step_generate_usage(state: GenerationState) -> Result[GenerationState, str]:
    """Pipeline step to generate the installation and usage guide."""
    state.console.print("Generating usage guide...")
    # This step requires getting help text from the CLI. This impurity is
    # contained here. A 100% pure approach would require mocking, which
    # is overkill for this script.
    try:
        from robofactor.main import app as cli_app
        from typer.testing import CliRunner

        runner = CliRunner()
        help_result = runner.invoke(cli_app, ["--help"])
        cli_help_text = help_result.stdout
    except Exception as e:
        return Err(f"Failed to get CLI help text: {e}")

    pyproject_res = state.project_context.get_config("pyproject.toml")
    makefile_res = state.project_context.get_config("Makefile")
    if isinstance(pyproject_res, Err):
        return pyproject_res
    if isinstance(makefile_res, Err):
        return makefile_res

    usage_generator = dspy.Predict(GenerateUsage)
    try:
        prediction = usage_generator(
            pyproject_toml=pyproject_res.value,
            makefile=makefile_res.value,
            cli_help_text=cli_help_text,
        )
        usage = UsageGuide(
            installation=prediction.installation_instructions, usage=prediction.usage_instructions
        )
        return Ok(replace(state, usage=usage))
    except Exception as e:
        return Err(f"LLM failed to generate usage guide: {e}")


def step_assemble_readme(state: GenerationState) -> Result[str, str]:
    """Final pipeline step to assemble the complete README.md content."""
    state.console.print("Assembling final README...")
    if not state.architecture or not state.usage:
        return Err("Architecture or Usage data is missing for final assembly.")

    component_markdown = "\n".join(
        f"- `{comp['component']}`: {comp['description']}"
        for comp in state.architecture.component_breakdown
    )

    assembler = dspy.Predict(AssembleReadme)
    try:
        # These are hardcoded for this specific project.
        # A more generic script could pull them from the GenerationState.
        prediction = assembler(
            project_name="robofactor",
            project_description="The robot who refactors: /[^_^]\\" ,
            installation=state.usage.installation,
            usage=state.usage.usage,
            architecture_overview=state.architecture.overview,
            component_breakdown=component_markdown,
        )
        return Ok(prediction.readme_content)
    except Exception as e:
        return Err(f"LLM failed to assemble README: {e}")


# ============================================================================
# 6. COMPOSITION LAYER
# ============================================================================


def pipeline_reducer(
    state_result: Result[GenerationState, str],
    step_func: Callable[[GenerationState], Result[GenerationState, str]],
) -> Result[GenerationState, str]:
    """
    A reducer for `functools.reduce` that chains pipeline steps together.
    It stops the pipeline on the first `Err` result, implementing a
    railway-oriented programming pattern.
    """
    if isinstance(state_result, Err):
        return state_result
    return step_func(state_result.value)


def run_generation_pipeline(state: GenerationState) -> Result[str, str]:
    """
    Executes the full README generation pipeline by composing the steps.
    """
    pipeline_steps: list[Callable[[GenerationState], Result[GenerationState, str]]] = [
        step_summarize_files,
        step_analyze_architecture,
        step_generate_usage,
    ]

    # The pipeline builds up the GenerationState through reduction.
    final_state_result = reduce(pipeline_reducer, pipeline_steps, Ok(state))

    # The final step consumes the state to produce the final string.
    if isinstance(final_state_result, Err):
        return final_state_result

    return step_assemble_readme(final_state_result.value)


# ============================================================================
# 7. I/O & CLI LAYER
# ============================================================================


def write_file(path: Path, content: str) -> Result[None, str]:
    """Pure-functional file writing."""
    try:
        path.write_text(content, encoding="utf-8")
        return Ok(None)
    except Exception as e:
        return Err(f"Failed to write to {path}: {e}")


def configure_dspy(model_name: str) -> Result[dspy.LM, str]:
    """Configures the DSPy framework with the specified language model."""
    try:
        llm = dspy.LM(model_name, max_tokens=64000)
        dspy.configure(lm=llm)
        return Ok(llm)
    except Exception as e:
        return Err(f"Failed to configure DSPy with model '{model_name}': {e}")


def main_flow(
    root: Path, output_path: Path, model_name: str, console: Console
) -> Result[None, str]:
    """
    The main workflow orchestrator.
    This function composes all operations, from configuration and data
    acquisition to pipeline execution and final file output.
    """
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*Pydantic serializer warnings.*",
    )
    console.print(f"[dim]1. Configuring LLM: {model_name}...[/dim]")
    llm_result = configure_dspy(model_name)
    if isinstance(llm_result, Err):
        return llm_result

    console.print("[dim]2. Analyzing project structure...[/dim]")
    project_context_result = create_project_context(root)
    if isinstance(project_context_result, Err):
        return project_context_result

    initial_state = GenerationState(
        project_context=project_context_result.value,
        console=console,
        llm_config=llm_result.value,
    )

    console.print("[dim]3. Starting README generation pipeline...[/dim]")
    readme_content_result = run_generation_pipeline(initial_state)
    if isinstance(readme_content_result, Err):
        return readme_content_result

    console.print(f"[dim]4. Writing output to {output_path}...[/dim]")
    return write_file(output_path, readme_content_result.value)


# --- Typer CLI Application ---
app = typer.Typer(
    help="A purely functional README generator for the robofactor project.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def generate(
    output: Path = typer.Option(
        project_root / "README.md",
        "--output",
        "-o",
        help="Path to write the generated README.md file.",
        show_default=True,
    ),
    model: str = typer.Option(
        "gemini/gemini-2.5-pro",
        "--model",
        "-m",
        help="Language model to use for generation.",
        show_default=True,
    ),
) -> None:
    """
    Analyzes the project and generates a comprehensive README.md.
    """
    console = Console()
    console.print("\n[bold cyan]═══ Robofactor README Generator ═══[/bold cyan]\n")

    # Execute the main flow and handle the final result.
    # This is the edge of the application where side-effects (printing to
    # the console, exiting) are handled based on the pure core's output.
    result = main_flow(project_root, output, model, console)

    match result:
        case Ok(_):
            console.print(
                f"\n[bold green]✅ README successfully generated at: {output}[/bold green]"
            )
        case Err(error):
            console.print(f"\n[bold red]❌ Generation failed:[/bold red]\n{error}")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
