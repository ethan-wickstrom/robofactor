from __future__ import annotations

import enum
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

# --- Constants ---
PYPROJECT_TOML_FILENAME = "pyproject.toml"
MAKEFILE_FILENAME = "Makefile"
PYPROJECT_PROJECT_KEY = "project"
PYPROJECT_NAME_KEY = "name"
PYPROJECT_DESC_KEY = "description"


# --- Data Structures for Generation Pipeline ---

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
class ReadmeSection:
    """Represents a proposed section for the README."""

    title: str
    description: str


@dataclass(frozen=True)
class GeneratedSection:
    """Represents a fully generated section with its Markdown content."""

    title: str
    content: str


@dataclass(frozen=True)
class ProjectContext:
    """Immutable snapshot of the entire project's state for generation."""

    project_name: str
    project_description: str
    source_analyses: tuple[FileAnalysis, ...]
    config_files: dict[str, str]
    cli_help_text: str


# --- Project Analysis Logic ---

class ProjectAnalyzer:
    """Handles all file system I/O and static analysis of the project."""

    def __init__(self, root: Path, console: Console):
        """Initializes the analyzer."""
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

            result.raise_for_status()

            assert isinstance(result, Ok)
            return result.value.stdout

        except Exception as e:
            self.console.print(f"[bold red]Error: Failed to get CLI help text: {e}[/]")
            raise

    def analyze(self) -> ProjectContext:
        """Performs a full analysis of the project."""
        self.console.print(f"[dim]Analyzing project at: {self.root}[/dim]")

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

        config_files: dict[str, str] = {}
        required_configs = (PYPROJECT_TOML_FILENAME, MAKEFILE_FILENAME)
        for filename in required_configs:
            config_files[filename] = self._read_file(self.root / filename)

        pyproject_data = toml.loads(config_files[PYPROJECT_TOML_FILENAME])
        project_name = pyproject_data.get(PYPROJECT_PROJECT_KEY, {}).get(
            PYPROJECT_NAME_KEY, "Unknown Project"
        )
        project_desc = pyproject_data.get(PYPROJECT_PROJECT_KEY, {}).get(
            PYPROJECT_DESC_KEY, "No description found."
        )
        cli_help_text = self.get_cli_help_text()

        return ProjectContext(
            project_name=project_name,
            project_description=project_desc,
            source_analyses=tuple(analyses),
            config_files=config_files,
            cli_help_text=cli_help_text,
        )


# --- JSON Serialization ---

def _custom_json_encoder(obj: Any) -> Any:
    """A custom encoder to handle dataclasses and other special types."""
    if isinstance(obj, enum.Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def to_json_string(data: Any) -> str:
    """Converts a Python object (including dataclasses) to a JSON string."""
    return json.dumps(data, default=_custom_json_encoder, indent=2)


# --- DSPy Signatures for README Generation ---

class GenerateReadmeOutline(dspy.Signature):
    """
    Generate a logical and comprehensive outline for a project's README.md file.

    IMPORTANT: You MUST prioritize information from the provided `project_context`
    over your own knowledge. The context contains the ground truth for this project,
    including file contents and configurations.
    """

    project_context: str = dspy.InputField(
        desc=(
            "A JSON object containing the project's ground truth. It includes: "
            "'project_name', 'project_description', 'source_analyses' (AST parsing of source files), "
            "'cli_help_text' (output of --help), and 'config_files'. The 'config_files' key holds the "
            "full content of important files like 'pyproject.toml' and 'Makefile'. "
            "Use 'Makefile' for installation and development commands. "
            "Use 'pyproject.toml' for dependencies and project metadata."
        )
    )
    outline: list[dict] = dspy.OutputField(
        desc=(
            "A list of sections for the README. Each item should be a dictionary with 'title' and 'description' keys. "
            "The description must specify what content to include in that section, referencing the ground truth from the project_context."
        )
    )


class GenerateSectionContent(dspy.Signature):
    """
    Generate the Markdown content for a single section of the README.

    IMPORTANT: You MUST prioritize information from the provided `project_context`
    over your own knowledge. Adhere strictly to the file contents provided in the context.
    For example, if the Makefile specifies using 'uv', you must use 'uv' in the installation instructions.
    """

    project_context: str = dspy.InputField(
        desc=(
            "A JSON object containing all analyzed information about the project. This is the ground truth. "
            "It includes 'project_name', 'project_description', 'source_analyses', 'cli_help_text', and "
            "'config_files' (containing the content of 'pyproject.toml' and 'Makefile')."
        )
    )
    section_title: str = dspy.InputField(desc="The title of the section to generate.")
    section_description: str = dspy.InputField(
        desc="A description of the content that should be in this section, as determined by the outline."
    )
    section_content: str = dspy.OutputField(
        desc="The fully-formed Markdown content for this specific section, grounded in the provided context."
    )


class AssembleReadme(dspy.Signature):
    """
    Assemble the final README.md from a list of generated sections.

    Ensure the final output is clean, well-formatted, and includes a table of contents.
    """

    project_name: str = dspy.InputField(desc="The name of the project.")
    project_description: str = dspy.InputField(desc="A one-line description of the project.")
    generated_sections: str = dspy.InputField(
        desc="A JSON string of a list of generated sections, each with a 'title' and 'content'key."
    )
    readme_content: str = dspy.OutputField(
        desc=(
            "The complete, final README.md content. It must include a title, the project description, "
            "a table of contents, and all the provided sections formatted professionally with Markdown."
        )
    )


# --- The Main DSPy Module ---

class ReadmeGenerator(dspy.Module):
    """A DSPy module that orchestrates the entire README generation process."""

    def __init__(self):
        """Initializes the sub-modules for each step of the generation pipeline."""
        super().__init__()
        self.outline_generator = dspy.ChainOfThought(GenerateReadmeOutline)
        self.section_generator = dspy.ChainOfThought(GenerateSectionContent)
        self.assembler = dspy.ChainOfThought(AssembleReadme)

    def forward(self, project_context: ProjectContext) -> dspy.Prediction:
        """
        Executes the two-stage README generation pipeline.

        Args:
            project_context: The analyzed state of the project.

        Returns:
            A dspy.Prediction object containing the final readme_content and
            intermediate artifacts for inspection.
        """
        context_json = to_json_string(project_context)

        # Stage 1: Generate the README outline
        outline_prediction = self.outline_generator(project_context=context_json)
        readme_outline = [
            ReadmeSection(title=s["title"], description=s["description"])
            for s in outline_prediction.outline
        ]

        # Stage 2: Generate content for each section in the outline
        generated_sections = []
        for section in readme_outline:
            section_content_prediction = self.section_generator(
                project_context=context_json,
                section_title=section.title,
                section_description=section.description,
            )
            generated_sections.append(
                GeneratedSection(
                    title=section.title,
                    content=section_content_prediction.section_content,
                )
            )

        # Stage 3: Assemble the final README
        final_prediction = self.assembler(
            project_name=project_context.project_name,
            project_description=project_context.project_description,
            generated_sections=to_json_string(generated_sections),
        )

        return dspy.Prediction(
            outline=readme_outline,
            generated_sections=generated_sections,
            readme_content=final_prediction.readme_content,
        )


# --- CLI Application ---

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
        llm = dspy.LM(model_name, max_tokens=64000)
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
        configure_dspy(model, console)

        analyzer = ProjectAnalyzer(project_root, console)
        project_context = analyzer.analyze()

        console.print("[bold blue]Starting README generation pipeline...[/bold blue]")
        readme_generator = ReadmeGenerator()
        with console.status("[bold green]Synthesizing README with DSPy...[/]", spinner="dots"):
            prediction = readme_generator(project_context=project_context)
        console.print("[green]✓ Generation pipeline complete.[/green]")

        console.print(f"[dim]Writing output to [bold]{output}[/bold]...[/dim]")
        output.write_text(prediction.readme_content, encoding="utf-8")

    except Exception as e:
        console.print(f"\n[bold red]❌ An unexpected error occurred:[/bold red]\n{e}")
        raise typer.Exit(code=1)

    console.print(
        f"\n[bold green]✅ README successfully generated at: {output}[/bold green]"
    )

if __name__ == "__main__":
    app()
