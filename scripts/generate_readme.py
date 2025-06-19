#!/usr/bin/env python3
"""
Intelligent README Generator for the robofactor project.

This script leverages DSPy to perform a deep, context-aware analysis of the
project's source code and configuration files. It synthesizes this information
into a comprehensive, well-structured, and professional README.md file.

The architecture is designed to be:
- **Dynamic & Data-Driven**: The README content is generated based on the
  current state of the codebase, `pyproject.toml`, and `Makefile`. It adapts
  as the project evolves without needing manual script updates.
- **Modular & Composable**: The entire generation process is encapsulated within
  a `ReadmeGenerator(dspy.Module)`, which composes smaller, specialized DSPy
  modules for each task (summarization, architecture analysis, etc.).
- **Intelligent**: It goes beyond simple file concatenation by using LMs to
  understand the role of each component, analyze the overall architecture,
  and describe the control flow.
- **Maintainable**: Concerns are separated into distinct layers: data acquisition
  (ProjectAnalyzer), AI logic (DSPy Signatures and Modules), and CLI interaction.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import dspy
import toml
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# --- Setup Project Path ---
# This allows importing from the `src` directory for analysis.
try:
    project_root = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(project_root / "src"))
    from robofactor.function_extraction import FunctionInfo, parse_python_source
    from robofactor.functional_types import CliResult, Err, Ok, Result
    from robofactor.main import app as cli_app
    from robofactor.utils import suppress_pydantic_warnings
except ImportError as e:
    print(
        f"Error: Failed to import project modules. Make sure you run from the project root"
        f" and have installed dependencies.\nDetails: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

# ============================================================================
# 1. DATA STRUCTURES (IMMUTABLE ADTs)
# ============================================================================


@dataclass(frozen=True)
class FileAnalysis:
    """Immutable representation of a source file's content and structure."""

    relative_path: str
    structure: tuple[FunctionInfo, ...]


@dataclass(frozen=True)
class FileSummary:
    """The result of summarizing a single file."""

    file_path: str
    summary: str


@dataclass(frozen=True)
class Architecture:
    """The synthesized understanding of the project's architecture."""

    overview: str
    components: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class UsageGuide:
    """Generated installation and usage instructions."""

    installation: str
    usage: str


@dataclass(frozen=True)
class ProjectContext:
    """Immutable snapshot of the entire project's state for generation."""

    project_name: str
    project_description: str
    source_analyses: tuple[FileAnalysis, ...]
    config_files: dict[str, str]  # filename -> content


# ============================================================================
# 2. DATA ACQUISITION & STATIC ANALYSIS
# ============================================================================


class ProjectAnalyzer:
    """Handles all file system I/O and static analysis of the project."""

    def __init__(self, root: Path, console: Console):
        """
        Initializes the analyzer.

        Args:
            root: The project's root directory.
            console: A Rich console instance for output.
        """
        self.root = root
        self.console = console
        self.source_dir = root / "src" / "robofactor"

    def _read_file(self, path: Path) -> str:
        """Reads a file, raising a FileNotFoundError on failure."""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.console.print(f"[bold red]Error: File not found at {path}[/]")
            raise
        except Exception as e:
            self.console.print(f"[bold red]Error: Failed to read {path}: {e}[/]")
            raise

    def _analyze_source_file(self, path: Path) -> FileAnalysis:
        """Parses a Python file to extract its structure."""
        content = self._read_file(path)
        try:
            structure = tuple(parse_python_source(content, module_name=path.name))
            return FileAnalysis(
                relative_path=str(path.relative_to(self.root)),
                structure=structure,
            )
        except Exception as e:
            self.console.print(f"[bold red]Error: Failed to parse AST for {path}: {e}[/]")
            raise

    def get_cli_help_text(self) -> str:
        """Captures the --help output from the project's Typer CLI."""
        self.console.print("[dim]Capturing CLI help text...[/dim]")
        try:
            from typer.testing import CliRunner

            runner = CliRunner()
            cli_runner_result = runner.invoke(cli_app, ["--help"], catch_exceptions=False)

            # Convert the runner result into our functional Result type
            if cli_runner_result.exit_code == 0:
                result: Result[CliResult, str] = Ok(
                    CliResult(
                        stdout=cli_runner_result.stdout,
                        stderr=cli_runner_result.stderr,
                        exit_code=cli_runner_result.exit_code,
                    )
                )
            else:
                result = Err(
                    f"CLI command failed with exit code {cli_runner_result.exit_code}:\n{cli_runner_result.stderr}"
                )

            # Now, call the requested method, which will raise on Err
            result.raise_for_status()

            # If it didn't raise, we can safely access the value.
            assert isinstance(result, Ok)
            return result.value.stdout

        except Exception as e:
            self.console.print(f"[bold red]Error: Failed to get CLI help text: {e}[/]")
            raise

    def analyze(self) -> tuple[ProjectContext, str]:
        """
        Performs a full analysis of the project.

        Returns:
            A tuple containing the ProjectContext and the CLI help text.
        """
        self.console.print(f"[dim]Analyzing project at: {self.root}[/dim]")

        # Analyze source files
        py_files = [p for p in self.source_dir.glob("*.py") if p.name != "__init__.py"]
        analyses: list[FileAnalysis] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing source files...", total=len(py_files))
            for file_path in py_files:
                progress.update(task, description=f"Parsing {file_path.name}")
                analyses.append(self._analyze_source_file(file_path))
                progress.advance(task)

        # Read config files and project metadata
        config_files: dict[str, str] = {}
        required_configs = ("pyproject.toml", "Makefile")
        for filename in required_configs:
            config_files[filename] = self._read_file(self.root / filename)

        pyproject_data = toml.loads(config_files["pyproject.toml"])
        project_name = pyproject_data.get("project", {}).get("name", "Unknown Project")
        project_desc = pyproject_data.get("project", {}).get(
            "description", "No description found."
        )

        context = ProjectContext(
            project_name=project_name,
            project_description=project_desc,
            source_analyses=tuple(analyses),
            config_files=config_files,
        )

        cli_help_text = self.get_cli_help_text()
        return context, cli_help_text


# ============================================================================
# 3. DSPy SIGNATURES (DECLARATIVE AI TASKS)
# ============================================================================


def _custom_json_encoder(obj: Any) -> Any:
    """A custom encoder to handle dataclasses, enums, and paths for JSON."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value") and not isinstance(obj, type):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def to_json_string(data: Any) -> str:
    """Converts a Python object (including dataclasses) to a JSON string."""
    return json.dumps(data, default=_custom_json_encoder, indent=2)


class SummarizeFile(dspy.Signature):
    """Summarize a Python file's purpose based on its structure."""

    file_path: str = dspy.InputField(desc="The relative path to the Python file.")
    file_structure: str = dspy.InputField(
        desc="A JSON object describing the functions and classes in the file."
    )
    summary: str = dspy.OutputField(
        desc="A concise, one-paragraph summary of the file's main purpose and role in the project."
    )


class SynthesizeArchitecture(dspy.Signature):
    """Analyze file summaries to describe the project's architecture and control flow."""

    project_name: str = dspy.InputField(desc="The name of the project.")
    file_summaries: str = dspy.InputField(desc="A JSON array of summaries for all project files.")
    overview: str = dspy.OutputField(
        desc="A high-level paragraph describing the project's purpose, architecture, and the flow of data/control between key components."
    )
    components: list[dict[str, str]] = dspy.OutputField(
        desc="A list of dictionaries, each with 'component' (file path) and 'description' keys, detailing the role of each file."
    )


class GenerateUsage(dspy.Signature):
    """Generate installation and usage instructions from config files."""

    pyproject_toml: str = dspy.InputField(desc="Content of pyproject.toml.")
    makefile: str = dspy.InputField(desc="Content of Makefile.")
    cli_help_text: str = dspy.InputField(desc="The --help output from the main CLI.")
    installation_instructions: str = dspy.OutputField(
        desc="Markdown-formatted installation instructions, including `uv` and `make` commands."
    )
    usage_instructions: str = dspy.OutputField(
        desc="Markdown-formatted usage instructions with clear examples based on the CLI help text."
    )


class AssembleReadme(dspy.Signature):
    """Assemble a complete, well-structured, and professional README.md from all generated sections."""

    project_name: str = dspy.InputField(desc="The name of the project.")
    project_description: str = dspy.InputField(desc="A one-line description of the project.")
    installation_guide: str = dspy.InputField(desc="Markdown content for the 'Installation' section.")
    usage_guide: str = dspy.InputField(desc="Markdown content for the 'Usage' section, including CLI examples.")
    architecture_overview: str = dspy.InputField(desc="Markdown content for the 'Architecture' section overview.")
    component_breakdown: str = dspy.InputField(
        desc="A markdown-formatted list of system components and their descriptions."
    )
    readme_content: str = dspy.OutputField(
        desc="The complete, final README.md content. It must include a table of contents, all the provided sections, and be formatted professionally."
    )


# ============================================================================
# 4. DSPy README GENERATOR MODULE
# ============================================================================


class ReadmeGenerator(dspy.Module):
    """A DSPy module that orchestrates the entire README generation process."""

    def __init__(self):
        """Initializes the sub-modules for each step of the generation pipeline."""
        super().__init__()
        self.summarizer = dspy.Predict(SummarizeFile, max_tokens=4000)
        self.architect = dspy.ChainOfThought(SynthesizeArchitecture, max_tokens=4000)
        self.usage_writer = dspy.ChainOfThought(GenerateUsage, max_tokens=4000)
        self.assembler = dspy.ChainOfThought(AssembleReadme, max_tokens=8000)

    def forward(
        self, project_context: ProjectContext, cli_help_text: str
    ) -> dspy.Prediction:
        """
        Executes the README generation pipeline.

        Args:
            project_context: The analyzed state of the project.
            cli_help_text: The captured --help output from the CLI.

        Returns:
            A dspy.Prediction object containing the final readme_content and
            intermediate artifacts for inspection.
        """
        # 1. Summarize each source file
        file_summaries = [
            FileSummary(
                file_path=analysis.relative_path,
                summary=self.summarizer(
                    file_path=analysis.relative_path,
                    file_structure=to_json_string(analysis.structure),
                ).summary,
            )
            for analysis in project_context.source_analyses
        ]

        # 2. Synthesize the overall architecture
        arch_prediction = self.architect(
            project_name=project_context.project_name,
            file_summaries=to_json_string(file_summaries),
        )
        architecture = Architecture(
            overview=arch_prediction.overview,
            components=tuple(arch_prediction.components),
        )

        # 3. Generate the usage guide
        usage_prediction = self.usage_writer(
            pyproject_toml=project_context.config_files["pyproject.toml"],
            makefile=project_context.config_files["Makefile"],
            cli_help_text=cli_help_text,
        )
        usage = UsageGuide(
            installation=usage_prediction.installation_instructions,
            usage=usage_prediction.usage_instructions,
        )

        # 4. Assemble the final README
        component_markdown = "\n".join(
            f"- `{comp['component']}`: {comp['description']}"
            for comp in architecture.components
        )

        final_prediction = self.assembler(
            project_name=project_context.project_name,
            project_description=project_context.project_description,
            installation_guide=usage.installation,
            usage_guide=usage.usage,
            architecture_overview=architecture.overview,
            component_breakdown=component_markdown,
        )

        # Return a structured prediction with all artifacts
        return dspy.Prediction(
            summaries=file_summaries,
            architecture=architecture,
            usage=usage,
            readme_content=final_prediction.readme_content,
        )


# ============================================================================
# 5. CLI & MAIN EXECUTION
# ============================================================================

app = typer.Typer(
    help="An intelligent, context-aware README generator for the robofactor project.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def configure_dspy(model_name: str, console: Console) -> None:
    """Configures the DSPy framework with the specified language model."""
    console.print(f"[dim]Configuring LLM: [bold]{model_name}[/bold]...[/dim]")
    try:
        # Use a larger model for generation tasks.
        llm = dspy.LM(model_name)
        dspy.configure(lm=llm)
    except Exception as e:
        console.print(f"[bold red]Error: Failed to configure DSPy with model '{model_name}': {e}[/]")
        raise typer.Exit(code=1)


@app.command()
def generate(
    output: Path = typer.Option(
        project_root / "README.md",
        "--output",
        "-o",
        help="Path to write the generated README.md file.",
        show_default=True,
        writable=True,
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
    suppress_pydantic_warnings()
    console = Console()
    console.print("\n[bold cyan]═══ Robofactor README Generator ═══[/bold cyan]\n")

    try:
        # 1. Configure AI model
        configure_dspy(model, console)

        # 2. Analyze project context
        analyzer = ProjectAnalyzer(project_root, console)
        project_context, cli_help_text = analyzer.analyze()

        # 3. Instantiate and run the generator module
        console.print("[bold blue]Starting README generation pipeline...[/bold blue]")
        readme_generator = ReadmeGenerator()
        with console.status("[bold green]Synthesizing README with DSPy...[/]", spinner="dots"):
            prediction = readme_generator(
                project_context=project_context, cli_help_text=cli_help_text
            )
        console.print("[green]✓ Generation pipeline complete.[/green]")

        # 4. Write the output file
        console.print(f"[dim]Writing output to [bold]{output}[/bold]...[/dim]")
        output.write_text(prediction.readme_content, encoding="utf-8")

    except Exception as e:
        # Catch any exceptions raised during the process
        console.print(f"\n[bold red]❌ An unexpected error occurred:[/bold red]\n{e}")
        raise typer.Exit(code=1)

    console.print(
        f"\n[bold green]✅ README successfully generated at: {output}[/bold green]"
    )


if __name__ == "__main__":
    app()
