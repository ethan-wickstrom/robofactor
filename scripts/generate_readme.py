"""
This script uses DSPy to automatically generate a README.md for the robofactor project.

It performs a multi-stage analysis of the codebase to produce a comprehensive
and well-structured documentation file.
"""

import json
import sys
from pathlib import Path

import dspy
import typer
from rich.console import Console

# Add project root to sys.path to allow imports from the 'robofactor' package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


# --- DSPy Signatures for README Generation ---


class FileSummary(dspy.Signature):
    """Summarize a Python file's purpose, key components, and role in the project."""

    file_path: str = dspy.InputField(desc="The relative path to the Python file.")
    file_content: str = dspy.InputField(desc="The full content of the Python file.")

    summary: str = dspy.OutputField(
        desc="A concise, one-paragraph summary of the file's main purpose and its role within the larger project."
    )
    key_components: list[str] = dspy.OutputField(
        desc="A bulleted list of the most important functions or classes in this file, with a brief (one-sentence) description of each."
    )


class ArchitectureAnalysis(dspy.Signature):
    """Analyze file summaries to describe the overall project architecture."""

    file_summaries: str = dspy.InputField(
        desc="A JSON string of summaries for all files in the project, including file paths, summaries, and key components."
    )

    architecture_overview: str = dspy.OutputField(
        desc="A high-level paragraph describing the project's architecture, data flow, and how the main components interact."
    )
    component_breakdown: list[dict[str, str]] = dspy.OutputField(
        desc="A list of dictionaries, each with 'component' (file path) and 'description' keys, detailing each major part of the system."
    )


class UsageGuide(dspy.Signature):
    """Generate installation and usage instructions from project configuration files."""

    pyproject_toml_content: str = dspy.InputField(desc="The content of the pyproject.toml file.")
    makefile_content: str = dspy.InputField(desc="The content of the Makefile.")
    main_py_content: str = dspy.InputField(desc="The content of the main CLI entrypoint file (main.py).")

    installation_instructions: str = dspy.OutputField(
        desc="Markdown-formatted installation instructions for both production and development, based on the provided files."
    )
    usage_instructions: str = dspy.OutputField(
        desc="Markdown-formatted usage instructions with clear CLI examples derived from the main entrypoint file."
    )


class Readme(dspy.Signature):
    """Assemble a complete, well-formatted README.md from various generated sections."""

    project_name: str = dspy.InputField()
    project_description: str = dspy.InputField()
    installation_instructions: str = dspy.InputField()
    usage_instructions: str = dspy.InputField()
    architecture_overview: str = dspy.InputField()
    component_breakdown: str = dspy.InputField(
        desc="A markdown-formatted string detailing each system component."
    )

    readme_content: str = dspy.OutputField(
        desc="The full, final, well-formatted README.md content, including headers, code blocks, and links."
    )


# --- DSPy Module for README Generation ---


class ReadmeGenerator(dspy.Module):
    """A DSPy program that generates a README for a Python project."""

    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought(FileSummary)
        self.architect = dspy.ChainOfThought(ArchitectureAnalysis)
        self.usage_writer = dspy.ChainOfThought(UsageGuide)
        self.assembler = dspy.ChainOfThought(Readme)

    def forward(self, project_root: Path):
        """Orchestrates the README generation pipeline."""
        console = Console()

        # 1. Summarize each source file
        source_dir = project_root / "src" / "robofactor"
        files_to_summarize = [p for p in source_dir.glob("*.py") if p.name != "__init__.py"]

        summaries = []
        with console.status("[bold green]Summarizing source files...[/bold green]", spinner="dots"):
            for file_path in files_to_summarize:
                console.log(f"Summarizing {file_path.relative_to(project_root)}...")
                content = file_path.read_text(encoding="utf-8")
                result = self.summarizer(
                    file_path=str(file_path.relative_to(project_root)), file_content=content
                )
                summaries.append(
                    {
                        "file_path": str(file_path.relative_to(project_root)),
                        "summary": result.summary,
                        "key_components": result.key_components,
                    }
                )

        # 2. Analyze architecture
        with console.status("[bold green]Analyzing architecture...[/bold green]", spinner="dots"):
            console.log("Synthesizing architecture overview...")
            arch_result = self.architect(file_summaries=json.dumps(summaries, indent=2))

        # 3. Generate Usage Guide
        with console.status("[bold green]Generating usage guide...[/bold green]", spinner="dots"):
            console.log("Analyzing project configuration and entrypoint...")
            main_py = (source_dir / "main.py").read_text(encoding="utf-8")
            pyproject_toml = (project_root / "pyproject.toml").read_text(encoding="utf-8")
            makefile = (project_root / "Makefile").read_text(encoding="utf-8")

            usage_result = self.usage_writer(
                main_py_content=main_py,
                pyproject_toml_content=pyproject_toml,
                makefile_content=makefile,
            )

        # 4. Assemble README
        with console.status("[bold green]Assembling README.md...[/bold green]", spinner="dots"):
            console.log("Generating final README content...")
            project_name = "robofactor"
            project_description = "The robot who refactors: /[^_^]\\"

            component_breakdown_md = "\n".join(
                f"- **`{item['component']}`**: {item['description']}"
                for item in arch_result.component_breakdown
            )

            readme_result = self.assembler(
                project_name=project_name,
                project_description=project_description,
                installation_instructions=usage_result.installation_instructions,
                usage_instructions=usage_result.usage_instructions,
                architecture_overview=arch_result.architecture_overview,
                component_breakdown=component_breakdown_md,
            )

        return readme_result.readme_content


# --- CLI Application ---

app = typer.Typer()


@app.command()
def main(
    output_path: Path = typer.Option(
        project_root / "README.md",
        "--output",
        "-o",
        help="The path to write the generated README.md file.",
    ),
    llm_model: str = typer.Option(
        "gemini/gemini-2.5-pro",
        "--model",
        help="The language model to use for generation.",
    ),
):
    """
    Generates a README.md for the robofactor project using DSPy.
    """
    console = Console()
    console.print("[bold cyan]Robofactor README Generator[/bold cyan]")

    # Configure DSPy
    llm = dspy.LM(llm_model, max_tokens=64000)
    dspy.configure(lm=llm)

    readme_generator = ReadmeGenerator()

    # Run the generator
    readme_content = readme_generator(project_root=project_root)

    # Write the output
    with console.status(f"[bold green]Writing README to {output_path}...[/bold green]"):
        output_path.write_text(readme_content, encoding="utf-8")

    console.print(f"[bold green]✅ README.md successfully generated at {output_path}[/bold green]")


if __name__ == "__main__":
    app()
