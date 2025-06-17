from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional

from dspy import Adapter, LM
from dspy.utils.callback import BaseCallback
from pydantic import BaseModel, ConfigDict, Field, computed_field
from .types import Temperature, PositiveInt

if TYPE_CHECKING:
    from dspy.clients.provider import Provider


class ImmutableModel(BaseModel):
    """Base class for immutable Pydantic models."""
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        validate_default=True,
    )


class ModelConfig(ImmutableModel):
    """
    Configuration for DSPy language models.
    """
    # Required
    model: str = Field(
        ...,
        min_length=3,
        description="Model identifier (e.g., 'openai/gpt-4o-mini')",
    )

    # Core parameters
    temperature: Temperature = 0.0
    max_tokens: PositiveInt = 4096

    # Behavior flags
    cache: bool = True
    num_retries: PositiveInt = 3

    # Extensions
    callbacks: list[BaseCallback] = Field(default_factory=list)
    provider: Optional["Provider"] = None
    finetuning_model: Optional[str] = None
    launch_kwargs: dict[str, Any] = Field(default_factory=dict)
    train_kwargs: dict[str, Any] = Field(default_factory=dict)


class AppConfig(ImmutableModel):
    """
    Application configuration with lazy LM instantiation.
    """
    adapter: Adapter
    project_path: str
    model_cfg: ModelConfig

    @computed_field
    @cached_property
    def lm(self) -> LM:
        """Lazy-load the language model from configuration."""
        return LM(**self.model_cfg.model_dump(), model_type="chat")
