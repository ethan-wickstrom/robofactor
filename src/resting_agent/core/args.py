"""
Command-line argument processing and validation for the Resting Agent.

This module provides functions to validate and transform raw CLI inputs into
strongly-typed configuration objects. It follows the IPO (Input-Process-Output)
model and maintains separation between CLI concerns and business logic.
"""

from pathlib import Path

import typer
from dspy import Adapter
from pydantic import ValidationError

from .config import AppConfig, ModelConfig
from .types import PositiveInt, Temperature


def _format_validation_errors(validation_error: ValidationError) -> str:
    """
    Format Pydantic validation errors into user-friendly CLI messages.

    Args:
        validation_error: The Pydantic ValidationError to format.

    Returns:
        A formatted string with bullet points for each error.
    """
    error_lines = []
    for error in validation_error.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        error_lines.append(f"  • {field}: {message}")

    return "\n".join(error_lines)


def _normalize_model_identifier(model: str) -> str:
    """
    Normalize a model identifier for consistent processing.

    Applies the Robustness Principle: be liberal in what you accept.

    Args:
        model: The raw model identifier string.

    Returns:
        A normalized model identifier.
    """
    # Strip whitespace and normalize slashes
    normalized = model.strip()

    # Handle common variations (e.g., "gpt-4" vs "openai/gpt-4")
    if "/" not in normalized and normalized.startswith(("gpt", "claude", "llama")):
        # Infer provider for common model prefixes
        if normalized.startswith("gpt"):
            normalized = f"openai/{normalized}"
        elif normalized.startswith("claude"):
            normalized = f"anthropic/{normalized}"

    return normalized


def _validate_project_path(path_str: str) -> Path:
    """
    Validate that a project path exists and is a directory.

    Args:
        path_str: The path string to validate.

    Returns:
        A resolved Path object.

    Raises:
        typer.BadParameter: If the path doesn't exist or isn't a directory.
    """
    path = Path(path_str).resolve()

    if not path.exists():
        raise typer.BadParameter(f"Project path does not exist: {path}")

    if not path.is_dir():
        raise typer.BadParameter(f"Project path must be a directory, not a file: {path}")

    # Check if it's a Laravel project (has artisan file)
    artisan_path = path / "artisan"
    if not artisan_path.exists():
        typer.echo(
            f"⚠️  Warning: No 'artisan' file found in {path}. This may not be a Laravel project.",
            err=True,
        )

    return path


def _create_model_config(
    *,
    model: str,
    temperature: Temperature,
    max_tokens: PositiveInt,
    cache: bool,
    num_retries: PositiveInt,
    finetuning_model: str | None,
) -> ModelConfig:
    """
    Process raw CLI inputs into a validated ModelConfig instance.

    This pure function represents the 'Process' step in the IPO model. It takes
    primitive types, applies normalization and business rules, and returns
    a validated, immutable configuration object.

    Args:
        model: Model identifier (e.g., "gpt-4", "openai/gpt-4").
        temperature: Sampling temperature for the model.
        max_tokens: Maximum tokens for model generation.
        cache: Whether to enable response caching.
        num_retries: Number of retries on API failures.
        finetuning_model: Optional fine-tuned model identifier.

    Returns:
        A validated ModelConfig instance.

    Raises:
        typer.BadParameter: If validation fails, with user-friendly error messages.
    """
    try:
        # Apply normalization to model identifier
        normalized_model = _normalize_model_identifier(model)

        # Construct the configuration data
        model_data = {
            "model": normalized_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cache": cache,
            "num_retries": num_retries,
            "finetuning_model": finetuning_model.strip() if finetuning_model else None,
        }

        # Let Pydantic handle validation according to defined constraints
        return ModelConfig(**model_data)

    except ValidationError as e:
        # Provide clear, actionable error messages
        error_details = _format_validation_errors(e)
        raise typer.BadParameter(f"Invalid model configuration:\n{error_details}") from e


def create_app_config(
    *,
    adapter: Adapter,
    project_path: str,
    model: str,
    temperature: Temperature,
    max_tokens: PositiveInt,
    cache: bool,
    num_retries: PositiveInt,
    finetuning_model: str | None,
) -> AppConfig:
    """
    Construct a validated AppConfig from CLI arguments and runtime objects.

    This function serves as the primary interface between the CLI and the
    application's configuration system. It orchestrates validation, normalization,
    and assembly of the final configuration object.

    Args:
        adapter: A runtime-instantiated DSPy Adapter for LLM interactions.
        project_path: Path to the Laravel project directory.
        model: LLM model identifier (e.g., "gpt-4", "claude-3").
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens for generation.
        cache: Whether to enable response caching.
        num_retries: Number of retries on API failures.
        finetuning_model: Optional fine-tuned model identifier.

    Returns:
        A fully validated and immutable AppConfig instance.

    Raises:
        typer.BadParameter: If any validation fails.
    """
    # Validate the project path
    validated_path = _validate_project_path(project_path)

    # Create and validate model configuration
    model_cfg = _create_model_config(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        num_retries=num_retries,
        finetuning_model=finetuning_model,
    )

    # Assemble the final configuration
    try:
        return AppConfig(
            adapter=adapter,
            project_path=str(validated_path),
            model_cfg=model_cfg,
        )
    except ValidationError as e:
        # Handle any remaining validation errors
        error_details = _format_validation_errors(e)
        raise typer.BadParameter(f"Invalid application configuration:\n{error_details}") from e
