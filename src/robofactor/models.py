from .json.types import JSON
from collections.abc import Sequence, Mapping
from pydantic import BaseModel, Field


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
    linting_issues: Sequence[str] = Field(default_factory=list)
