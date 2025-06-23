#!/usr/bin/env python3
"""
Intelligent README generator for the robofactor project.

This module uses DSPy with Pydantic integration to analyze the project structure
and generate a comprehensive README based on extracted information rather than assumptions.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Protocol

import dspy
import toml
import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path for imports
try:
    project_root = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(project_root / "src"))
    from robofactor.function_extraction import parse_python_source
    from robofactor.main import app as cli_app
    from robofactor.utils import suppress_pydantic_warnings
except ImportError as e:
    print(
        f"Error: Failed to import project modules. Make sure you run from the project root"
        f" and have installed dependencies.\nDetails: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Data Models (Pydantic) ---

class FunctionMetadata(BaseModel):
    """Metadata about a function extracted from source code."""
    name: str
    file_path: str
    docstring: str | None
    is_async: bool
    decorators: list[str]
    parameters: list[str]


class SourceFileAnalysis(BaseModel):
    """Analysis of a single source file."""
    relative_path: str
    functions: list[FunctionMetadata]
    imports: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)


class ProjectMetadata(BaseModel):
    """Basic project metadata from pyproject.toml."""
    name: str
    description: str
    version: str | None = None
    authors: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    dev_dependencies: list[str] = Field(default_factory=list)
    homepage: str | None = None
    repository: str | None = None


class DevelopmentEnvironment(BaseModel):
    """Extracted development environment information."""
    package_manager: str = Field(description="The package manager used (e.g., uv, pip, poetry)")
    install_command: str = Field(description="Command to install the package")
    dev_install_command: str = Field(description="Command to install with dev dependencies")
    available_commands: dict[str, str] = Field(
        default_factory=dict,
        description="Available make/task commands and their descriptions"
    )
    python_version: str | None = None


class ProjectFeatures(BaseModel):
    """High-level features extracted from the project."""
    core_technologies: list[str] = Field(description="Main technologies/libraries used")
    cli_capabilities: list[str] = Field(description="CLI commands and options available")
    key_modules: dict[str, str] = Field(
        description="Key modules and their purposes",
        default_factory=dict
    )
    testing_framework: str | None = None
    code_quality_tools: list[str] = Field(default_factory=list)


class ExtractedContext(BaseModel):
    """Complete extracted context for README generation."""
    metadata: ProjectMetadata
    environment: DevelopmentEnvironment
    features: ProjectFeatures
    source_analyses: list[SourceFileAnalysis]
    cli_help_text: str


class ReadmeSection(BaseModel):
    """A section in the README outline."""
    title: str
    description: str
    priority: int = Field(default=5, ge=1, le=10)


class GeneratedSection(BaseModel):
    """A generated README section with content."""
    title: str
    content: str


# --- Service Interfaces (Dependency Injection) ---

class FileReaderProtocol(Protocol):
    """Protocol for file reading operations."""
    def read_file(self, path: Path) -> str: ...
    def file_exists(self, path: Path) -> bool: ...


class CLIRunnerProtocol(Protocol):
    """Protocol for running CLI commands."""
    def get_help_text(self) -> str: ...


# --- Concrete Service Implementations ---

class FileReader:
    """Handles file system operations."""

    def read_file(self, path: Path) -> str:
        """Read a file's contents."""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            raise

    def file_exists(self, path: Path) -> bool:
        """Check if a file exists."""
        return path.exists()


class CLIRunner:
    """Handles CLI command execution."""

    def get_help_text(self) -> str:
        """Get the CLI help text."""
        try:
            from typer.testing import CliRunner
            runner = CliRunner()
            result = runner.invoke(cli_app, ["--help"], catch_exceptions=False)

            if result.exit_code != 0:
                error_msg = f"CLI failed with exit code {result.exit_code}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            return result.stdout
        except Exception as e:
            logger.error(f"Failed to get CLI help text: {e}")
            raise


# --- Project Analyzer ---

class ProjectAnalyzer:
    """Analyzes project structure and extracts information."""

    def __init__(
        self,
        root: Path,
        file_reader: FileReaderProtocol,
        cli_runner: CLIRunnerProtocol,
        console: Console | None = None
    ):
        self.root = root
        self.file_reader = file_reader
        self.cli_runner = cli_runner
        self.console = console or Console()
        self.source_dir = root / "src" / "robofactor"

    def analyze_source_file(self, path: Path) -> SourceFileAnalysis:
        """Analyze a Python source file."""
        content = self.file_reader.read_file(path)

        try:
            result = parse_python_source(content, module_name=path.name)
            functions = list(result.unwrap())

            # Convert FunctionInfo to our simplified FunctionMetadata
            func_metadata = [
                FunctionMetadata(
                    name=f.name,
                    file_path=str(path.relative_to(self.root)),
                    docstring=f.docstring,
                    is_async=f.is_async,
                    decorators=[d.name for d in f.decorators],
                    parameters=[p.name for p in f.parameters]
                )
                for f in functions
            ]

            return SourceFileAnalysis(
                relative_path=str(path.relative_to(self.root)),
                functions=func_metadata
            )
        except Exception as e:
            logger.error(f"Failed to parse {path}: {e}")
            raise

    def extract_project_metadata(self) -> ProjectMetadata:
        """Extract metadata from pyproject.toml."""
        pyproject_path = self.root / "pyproject.toml"
        content = self.file_reader.read_file(pyproject_path)
        data = toml.loads(content)

        project = data.get("project", {})
        deps = project.get("dependencies", [])
        dev_deps = data.get("dependency-groups", {}).get("dev", [])
        urls = project.get("urls", {})

        return ProjectMetadata(
            name=project.get("name", "Unknown"),
            description=project.get("description", ""),
            version=project.get("version"),
            authors=[a.get("name", "") for a in project.get("authors", [])],
            dependencies=deps,
            dev_dependencies=dev_deps,
            homepage=urls.get("Homepage"),
            repository=urls.get("Repository")
        )

    def analyze_all_source_files(self) -> list[SourceFileAnalysis]:
        """Analyze all Python source files."""
        py_files = [p for p in self.source_dir.glob("*.py") if p.name != "__init__.py"]
        analyses = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing source files...", total=len(py_files))

            for file_path in py_files:
                progress.update(task, description=f"Parsing {file_path.name}")
                analyses.append(self.analyze_source_file(file_path))
                progress.advance(task)

        return analyses

    def get_cli_help(self) -> str:
        """Get CLI help text."""
        self.console.print("[dim]Capturing CLI help text...[/dim]")
        return self.cli_runner.get_help_text()


# --- DSPy Signatures with Pydantic ---

class ExtractPackageManager(dspy.Signature):
    """Extract the package manager and installation commands from project files."""

    makefile_content: str = dspy.InputField(
        desc="Content of the Makefile"
    )
    pyproject_content: str = dspy.InputField(
        desc="Content of pyproject.toml"
    )
    package_manager: str = dspy.OutputField(
        desc="The package manager used (e.g., 'uv', 'pip', 'poetry')"
    )
    install_command: str = dspy.OutputField(
        desc="The exact command to install the package"
    )
    dev_install_command: str = dspy.OutputField(
        desc="The exact command to install with dev dependencies"
    )


class ExtractDevelopmentCommands(dspy.Signature):
    """Extract available development commands from Makefile."""

    makefile_content: str = dspy.InputField(
        desc="Content of the Makefile"
    )
    commands: dict[str, str] = dspy.OutputField(
        desc="Dictionary mapping command names to their descriptions"
    )


class ExtractProjectFeatures(dspy.Signature):
    """Extract key features and technologies from the project."""

    metadata: ProjectMetadata = dspy.InputField()
    source_analyses: list[SourceFileAnalysis] = dspy.InputField()
    cli_help_text: str = dspy.InputField()
    features: ProjectFeatures = dspy.OutputField()


class GenerateReadmeOutline(dspy.Signature):
    """Generate a README outline based on extracted context."""

    context: ExtractedContext = dspy.InputField()
    sections: list[ReadmeSection] = dspy.OutputField(
        desc="List of sections for the README, ordered by priority"
    )


class GenerateSectionContent(dspy.Signature):
    """Generate content for a specific README section."""

    context: ExtractedContext = dspy.InputField()
    section: ReadmeSection = dspy.InputField()
    content: str = dspy.OutputField(
        desc="Markdown content for this section"
    )


class AssembleReadme(dspy.Signature):
    """Assemble the final README from generated sections."""

    project_name: str = dspy.InputField()
    project_description: str = dspy.InputField()
    sections: list[GeneratedSection] = dspy.InputField()
    readme_content: str = dspy.OutputField(
        desc="Complete README.md content with proper formatting"
    )


# --- DSPy Modules ---

class ContextExtractor(dspy.Module):
    """Extracts specific context from project files."""

    def __init__(self):
        super().__init__()
        self.package_extractor = dspy.ChainOfThought(ExtractPackageManager)
        self.commands_extractor = dspy.ChainOfThought(ExtractDevelopmentCommands)
        self.features_extractor = dspy.ChainOfThought(ExtractProjectFeatures)

    def forward(
        self,
        metadata: ProjectMetadata,
        source_analyses: list[SourceFileAnalysis],
        makefile_content: str,
        pyproject_content: str,
        cli_help_text: str,
        python_version: str | None = None
    ) -> ExtractedContext:
        """Extract all context from project files."""

        # Extract package manager and install commands
        pkg_result = self.package_extractor(
            makefile_content=makefile_content,
            pyproject_content=pyproject_content
        )

        # Extract development commands
        cmd_result = self.commands_extractor(
            makefile_content=makefile_content
        )

        # Create development environment
        environment = DevelopmentEnvironment(
            package_manager=pkg_result.package_manager,
            install_command=pkg_result.install_command,
            dev_install_command=pkg_result.dev_install_command,
            available_commands=cmd_result.commands,
            python_version=python_version
        )

        # Extract project features
        features_result = self.features_extractor(
            metadata=metadata,
            source_analyses=source_analyses,
            cli_help_text=cli_help_text
        )

        return ExtractedContext(
            metadata=metadata,
            environment=environment,
            features=features_result.features,
            source_analyses=source_analyses,
            cli_help_text=cli_help_text
        )


class ReadmeGenerator(dspy.Module):
    """Generates README content from extracted context."""

    def __init__(self):
        super().__init__()
        self.outline_generator = dspy.ChainOfThought(GenerateReadmeOutline)
        self.section_generator = dspy.ChainOfThought(GenerateSectionContent)
        self.assembler = dspy.ChainOfThought(AssembleReadme)

    def forward(self, context: ExtractedContext) -> dspy.Prediction:
        """Generate complete README from context."""

        # Generate outline
        outline_result = self.outline_generator(context=context)
        sections = sorted(outline_result.sections, key=lambda s: s.priority)

        # Generate content for each section
        generated_sections = []
        for section in sections:
            section_result = self.section_generator(
                context=context,
                section=section
            )
            generated_sections.append(
                GeneratedSection(
                    title=section.title,
                    content=section_result.content
                )
            )

        # Assemble final README
        final_result = self.assembler(
            project_name=context.metadata.name,
            project_description=context.metadata.description,
            sections=generated_sections
        )

        return dspy.Prediction(
            outline=sections,
            generated_sections=generated_sections,
            readme_content=final_result.readme_content
        )


# --- Main Application ---

app = typer.Typer(
    help="Intelligent README generator for the robofactor project",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def configure_dspy(model_name: str) -> None:
    """Configure DSPy with the specified model."""
    logger.info(f"Configuring DSPy with model: {model_name}")
    try:
        llm = dspy.LM(model_name, max_tokens=64000)
        dspy.configure(lm=llm)
    except Exception as e:
        logger.error(f"Failed to configure DSPy: {e}")
        raise typer.Exit(code=1)


@app.command()
def generate(
    output: Path = typer.Option(
        project_root / "README.md",
        "--output",
        "-o",
        help="Path to write the generated README.md file.",
        writable=True,
    ),
    model: str = typer.Option(
        "gemini/gemini-2.5-pro",
        "--model",
        "-m",
        help="Language model to use for generation.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
) -> None:
    """Analyze the project and generate a comprehensive README."""
    suppress_pydantic_warnings()

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console = Console()
    console.print("\n[bold cyan]═══ Robofactor README Generator ═══[/bold cyan]\n")

    try:
        # Configure DSPy
        configure_dspy(model)

        # Initialize services
        file_reader = FileReader()
        cli_runner = CLIRunner()

        # Analyze project
        console.print("[dim]Analyzing project structure...[/dim]")
        analyzer = ProjectAnalyzer(project_root, file_reader, cli_runner, console)

        # Extract metadata
        metadata = analyzer.extract_project_metadata()
        logger.info(f"Extracted metadata for project: {metadata.name}")

        # Analyze source files
        source_analyses = analyzer.analyze_all_source_files()
        logger.info(f"Analyzed {len(source_analyses)} source files")

        # Get CLI help
        cli_help_text = analyzer.get_cli_help()

        # Read additional files
        makefile_content = file_reader.read_file(project_root / "Makefile")
        pyproject_content = file_reader.read_file(project_root / "pyproject.toml")

        # Read Python version if available
        python_version = None
        python_version_file = project_root / ".python-version"
        if file_reader.file_exists(python_version_file):
            python_version = file_reader.read_file(python_version_file).strip()

        # Extract context
        console.print("[bold blue]Extracting project context...[/bold blue]")
        context_extractor = ContextExtractor()
        context = context_extractor(
            metadata=metadata,
            source_analyses=source_analyses,
            makefile_content=makefile_content,
            pyproject_content=pyproject_content,
            cli_help_text=cli_help_text,
            python_version=python_version
        )

        logger.info(f"Extracted context - Package manager: {context.environment.package_manager}")
        logger.info(f"Available commands: {list(context.environment.available_commands.keys())}")

        # Generate README
        console.print("[bold green]Generating README content...[/bold green]")
        readme_generator = ReadmeGenerator()

        with console.status("[bold green]Synthesizing README with DSPy...[/]", spinner="dots"):
            result = readme_generator(context=context)

        console.print("[green]✓ Generation complete.[/green]")

        # Write output
        console.print(f"[dim]Writing output to [bold]{output}[/bold]...[/dim]")
        output.write_text(result.readme_content, encoding="utf-8")

        console.print(f"\n[bold green]✅ README successfully generated at: {output}[/bold green]")

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        console.print(f"\n[bold red]❌ An error occurred:[/bold red]\n{e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
