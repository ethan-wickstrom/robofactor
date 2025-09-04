from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodeQualityScores:
    """Holds various code quality metrics.

    Separated into a shared types module to avoid import cycles between
    analysis and evaluation when using runtime type checking (beartype).
    """

    linting_score: float
    complexity_score: float
    typing_score: float
    docstring_score: float
    linting_issues: list[str]


__all__ = ["CodeQualityScores"]

