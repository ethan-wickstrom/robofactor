"""
Command-line interface for the Resting Agent.

This module provides the main CLI entry point for the Resting Agent, a tool
that autonomously generates RESTful APIs for Laravel applications based on
natural language descriptions.
"""

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..agent import ApiAgent

# Initialize Rich console for better output
console = Console()

# Create the main Typer app
app = typer.Typer(
    name="resting-agent",
    help="🤖 Autonomous REST API generator for Laravel applications",
    add_completion=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Version information
__version__ = "0.1.0"


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        rprint(f"[bold blue]Resting Agent[/bold blue] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.command()
def generate(
    intent: Annotated[
        str,
        typer.Argument(
            help="Natural language description of the API to generate",
            metavar="INTENT",
        ),
    ],
    project_path: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-p",
            help="Path to the Laravel project directory",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="LLM model to use (e.g., gpt-4, claude-3)",
            envvar="RESTING_AGENT_MODEL",
        ),
    ] = "openai/gpt-4o-mini",
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            "-t",
            min=0.0,
            max=2.0,
            help="Model temperature for generation (0.0-2.0)",
        ),
    ] = 0.7,
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens",
            min=1,
            help="Maximum tokens for model generation",
        ),
    ] = 4096,
    cache: Annotated[
        bool,
        typer.Option(
            "--cache/--no-cache",
            help="Enable/disable response caching",
        ),
    ] = True,
    num_retries: Annotated[
        int,
        typer.Option(
            "--retries",
            min=0,
            help="Number of retries on API failures",
        ),
    ] = 3,
    finetuning_model: Annotated[
        str | None,
        typer.Option(
            "--finetuning-model",
            help="Optional fine-tuned model identifier",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show planned actions without executing",
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = False,
) -> None:
    """
    Generate a RESTful API for your Laravel application.

    This command takes a natural language description of the desired API
    functionality and automatically generates all necessary Laravel code
    including models, migrations, controllers, routes, validation, and tests.

    Examples:

        # Generate a blog API
        resting-agent "Create a blog API with posts and comments"

        # Generate with specific model
        resting-agent "User authentication API" --model claude-3-opus

        # Dry run to see planned actions
        resting-agent "Product catalog API" --dry-run
    """
    # Set default for project_path
    if project_path is None:
        project_path = Path.cwd()

    try:
        # Display welcome message
        console.print(
            Panel.fit(
                f"[bold blue]🤖 Resting Agent v{__version__}[/bold blue]\n"
                "[dim]Autonomous REST API Generator for Laravel[/dim]",
                border_style="blue",
            )
        )

        # Show intent interpretation
        console.print(f"\n[bold]Intent:[/bold] {intent}")
        console.print(f"[bold]Project:[/bold] {project_path}")

        if verbose:
            console.print(f"[bold]Model:[/bold] {model}")
            console.print(f"[bold]Temperature:[/bold] {temperature}")
            console.print(f"[bold]Max Tokens:[/bold] {max_tokens}")

        # Create configuration with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Initializing agent...", total=None)

            # Configure DSPy with the language model
            import dspy

            lm = dspy.LM(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                cache=cache,
                num_retries=num_retries,
            )
            dspy.configure(lm=lm, adapter=dspy.ChatAdapter)

            progress.update(task, completed=True)

        # Initialize the agent
        agent = ApiAgent()

        # Check for dry run mode first
        if dry_run:
            console.print("\n[yellow]🚫 Dry run mode enabled[/yellow]")
            console.print("\nThis would generate an API based on your intent.")
            console.print("Run without --dry-run to actually generate the code.")
            raise typer.Exit()

        # Confirm execution
        if not typer.confirm("\nProceed with API generation?"):
            console.print("[yellow]Generation cancelled[/yellow]")
            raise typer.Exit()

        # Execute the API generation using the forward method
        console.print("\n[bold cyan]🚀 Generating API...[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating API components...", total=None)

            # Call the forward method which handles the entire generation process
            agent.forward(intent=intent, project_path=str(project_path))

            progress.update(task, completed=True)

        # Show summary
        console.print(
            Panel(
                f"[bold]Generation Complete![/bold]\n\n"
                f"📁 Project: {project_path}\n"
                f"✅ API has been generated successfully!",
                title="[bold green]Summary[/bold green]",
                border_style="green",
            )
        )

        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  1. Run [cyan]php artisan migrate[/cyan] to create database tables")
        console.print("  2. Run [cyan]php artisan test[/cyan] to verify the API")
        console.print("  3. Check [cyan]routes/api.php[/cyan] for your new endpoints")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Generation interrupted by user[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"\n[red]❌ Error: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def version() -> None:
    """Show version information."""
    console.print(
        Panel(
            f"[bold blue]🤖 Resting Agent[/bold blue]\n\n"
            f"Version: [green]{__version__}[/green]\n"
            f"Python: [yellow]{sys.version.split()[0]}[/yellow]\n"
            f"Homepage: [link]https://github.com/yourusername/resting-agent[/link]",
            title="About",
            border_style="blue",
        )
    )


@app.command()
def models() -> None:
    """List supported LLM models."""
    models_info = {
        "OpenAI": [
            ("gpt-4", "Most capable, best for complex APIs"),
            ("gpt-4-turbo", "Faster GPT-4 variant"),
            ("gpt-3.5-turbo", "Fast and cost-effective"),
        ],
        "Anthropic": [
            ("claude-3-opus", "Most capable Claude model"),
            ("claude-3-sonnet", "Balanced performance"),
            ("claude-3-haiku", "Fast and efficient"),
        ],
        "Local": [
            ("ollama/codellama", "Code-optimized local model"),
            ("ollama/mistral", "General purpose local model"),
        ],
    }

    console.print("[bold]Supported LLM Models:[/bold]\n")

    for provider, models in models_info.items():
        console.print(f"[bold cyan]{provider}:[/bold cyan]")
        for model, description in models:
            console.print(f"  • [green]{model}[/green] - {description}")
        console.print()


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
