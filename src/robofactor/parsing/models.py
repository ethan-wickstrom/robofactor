import enum
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ..json.types import JSON


class TestCase(BaseModel):
    """Represents a single test case with positional and keyword args and expected output."""

    args: JSON = Field()
    kwargs: JSON = Field()
    expected_output: JSON


class CodeQualityScores(BaseModel):
    """Holds various code quality metrics."""

    linting_score: float
    complexity_score: float
    typing_score: float
    docstring_score: float
    linting_issues: Sequence[str] = Field(default_factory=Sequence)


class ParameterKind(enum.Enum):
    """Enumeration for the different kinds of function parameters."""

    POSITIONAL_ONLY = "positional_only"
    POSITIONAL_OR_KEYWORD = "positional_or_keyword"
    VAR_POSITIONAL = "var_positional"
    KEYWORD_ONLY = "keyword_only"
    VAR_KEYWORD = "var_keyword"


@dataclass(frozen=True)
class Parameter:
    """
    Represents a function parameter with its name, kind, and optional details.

    Attributes:
        name: The name of the parameter.
        kind: The kind of the parameter (e.g., positional-only).
        annotation: The type annotation as a string, if present.
        default: The default value as a string, if present.
    """

    name: str
    kind: ParameterKind
    annotation: str | None = None
    default: str | None = None


@dataclass(frozen=True)
class Decorator:
    """
    Represents a function decorator.

    Attributes:
        name: The name of the decorator.
        args: A tuple of arguments passed to the decorator, as strings.
    """

    name: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class FunctionContext:
    """Represents the context where a function is defined (base class)."""

    pass


@dataclass(frozen=True)
class ModuleContext(FunctionContext):
    """
    Represents a function defined at the module level.

    Attributes:
        module_name: The name of the module.
    """

    module_name: str


@dataclass(frozen=True)
class ClassContext(FunctionContext):
    """
    Represents a function defined within a class.

    Attributes:
        class_name: The name of the class.
        parent_context: The context in which the class is defined.
    """

    class_name: str
    parent_context: FunctionContext


@dataclass(frozen=True)
class NestedContext(FunctionContext):
    """
    Represents a function defined within another function.

    Attributes:
        parent_function: The name of the enclosing function.
        parent_context: The context of the enclosing function.
    """

    parent_function: str
    parent_context: FunctionContext
