from typing import Optional

import typer
from pydantic import ValidationError
from dspy import Adapter

from .config import AppConfig, ModelConfig
from .types import PositiveInt, Temperature

def _create_model_config(
    *,
    model: str,
    temperature: Temperature,
    max_tokens: PositiveInt,
    cache: bool,
    num_retries: PositiveInt,
    finetuning_model: Optional[str],
) -> ModelConfig:
    """
    A pure function to process raw inputs into a validated ModelConfig.

    This function represents the 'Process' step in the IPO model. It takes
    primitive types, enforces business rules via Pydantic models, and returns
    a validated, immutable data structure.

    Raises:
        typer.BadParameter: If validation fails, wrapping Pydantic's
                            ValidationError for CLI-friendly output.
    """
    try:
        # The Robustness Principle: be liberal in what you accept.
        # We strip whitespace and lowercase the model identifier before validation.
        model_data = {
            "model": model.strip(),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cache": cache,
            "num_retries": num_retries,
            "finetuning_model": finetuning_model,
        }

        # Pydantic handles the strict validation against the defined constraints.
        # This respects the DRY principle by reusing validation from config.py.
        return ModelConfig(**model_data)
    except ValidationError as e:
        # Translate a domain-specific error into a user-facing CLI error.
        error_messages = "\n".join(
            f"  - {err['loc'][0]}: {err['msg']}" for err in e.errors()
        )
        raise typer.BadParameter(
            f"Invalid model configuration provided:\n{error_messages}"
        )

def create_app_config(
    *,
    adapter: Adapter,
    project_path: str,
    model: str,
    temperature: Temperature,
    max_tokens: PositiveInt,
    cache: bool,
    num_retries: PositiveInt,
    finetuning_model: Optional[str],
) -> AppConfig:
    """
    Constructs the final AppConfig from CLI arguments and runtime objects.

    This function acts as the primary interface for the CLI module. It orchestrates
    the creation of the ModelConfig and assembles the final, immutable AppConfig.
    This separation ensures that the main CLI logic in `cli.py` remains clean and
    focused on command-line operations.

    Args:
        adapter: A runtime-instantiated DSPy Adapter (not from CLI).
        project_path: The required project path from the CLI.
        model: The required model identifier from the CLI.
        temperature: Model temperature.
        max_tokens: Model maximum tokens.
        cache: Flag to enable or disable caching.
        num_retries: Number of retries on failure.
        finetuning_model: Optional identifier for a finetuning model.

    Returns:
        A fully validated and immutable AppConfig instance.
    """
    # 1. Delegate the complex part of validation to the pure helper function.
    model_cfg = _create_model_config(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        num_retries=num_retries,
        finetuning_model=finetuning_model,
    )

    # 2. Assemble the final configuration object.
    # Pydantic validates the final AppConfig structure as well.
    try:
        return AppConfig(
            adapter=adapter,
            project_path=project_path,
            model_cfg=model_cfg,
        )
    except ValidationError as e:
        # This would typically catch issues with project_path or the adapter.
        raise typer.BadParameter(f"Invalid application configuration:\n{e}")
