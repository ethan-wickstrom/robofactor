from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic.functional_validators import field_validator

from .constants import ActionType


@dataclass
class ExecutionResult:
    """Represents the outcome of an executed action."""
    success: bool
    message: str
    data: Any | None = None

class Action(BaseModel):
    """A single, discrete step in the execution plan."""
    action_type: ActionType
    path: str = Field(default="", description="File path for file operations or command for execution.")
    content_description: str = Field(default="", description="Natural language description of the content to generate.")
    context_files: list[str] = Field(default_factory=list, description="List of files to read for context before execution.")
    dependencies: list[int] = Field(default_factory=list, description="Indices of actions that must complete before this one.")

    @field_validator('path')
    @classmethod
    def validate_path(cls, v: str, values: Any) -> str:
        """Ensure path is provided for relevant action types."""
        action_type = values.data.get('action_type')
        if action_type == ActionType.RUN_COMMAND and not v:
            raise ValueError("A command must be provided for the 'run_command' action.")
        return v

class Plan(BaseModel):
    """A structured execution plan composed of a sequence of actions."""
    actions: list[Action] = Field(alias='plan', description="The ordered list of actions to execute.")
    description: str = Field(default="", description="A high-level summary of the plan's objective.")

    def validate_dependencies(self) -> bool:
        """
        Validates that all dependency indices in the plan are valid.
        An action can only depend on actions that appear before it in the list.
        """
        for i, action in enumerate(self.actions):
            if any(dep >= i for dep in action.dependencies):
                return False
        return True
