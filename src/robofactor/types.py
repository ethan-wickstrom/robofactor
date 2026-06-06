from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import dspy
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from dspy.adapters.types.code import Code as _DSPyCode

    type PythonCode = _DSPyCode[Literal["python"]]
else:
    PythonCode = dspy.Code[Literal["python"]]


class OpportunityCategory(StrEnum):
    """Categories describing the primary focus of a refactoring opportunity."""

    PERFORMANCE = "PERFORMANCE"
    READABILITY = "READABILITY"
    STRUCTURE = "STRUCTURE"
    DOCUMENTATION = "DOCUMENTATION"
    TYPING = "TYPING"
    ERROR_HANDLING = "ERROR_HANDLING"
    TESTING = "TESTING"
    ROBUSTNESS = "ROBUSTNESS"
    MAINTAINABILITY = "MAINTAINABILITY"
    CLEANLINESS = "CLEANLINESS"
    OTHER = "OTHER"

    @classmethod
    def _missing_(cls, value: object) -> OpportunityCategory | None:
        """Handle alternative category names and normalize input."""
        if not isinstance(value, str):
            return None

        normalized = value.strip().replace("-", "_").replace(" ", "_").upper()
        return next((m for m in cls if m.value == normalized), None) or {
            "ERRORHANDLING": cls.ERROR_HANDLING,
            "ERRORHANDLNG": cls.ERROR_HANDLING,
            "ROBUST": cls.ROBUSTNESS,
            "MAINTAIN": cls.MAINTAINABILITY,
        }.get(normalized)


class RefactoringOpportunity(BaseModel):
    """Actionable refactoring opportunity surfaced during code analysis."""

    category: OpportunityCategory = OpportunityCategory.OTHER
    description: str
    expected_benefit: str | None = None


class CodeAnalysisReport(BaseModel):
    """Structured report summarising the code's intent and refactoring leads."""

    purpose: str
    complexity: str
    dependencies: list[str] = Field(default_factory=list)
    summary: str
    opportunities: list[RefactoringOpportunity] = Field(default_factory=list)


class PlanStepFocus(StrEnum):
    """Focus area targeted by a refactoring step."""

    STRUCTURE = "structure"
    MAINTAINABILITY = "maintainability"
    ROBUSTNESS = "robustness"
    READABILITY = "readability"
    TYPING = "typing"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object) -> PlanStepFocus | None:
        """Handle alternative focus names and normalize input."""
        if not isinstance(value, str):
            return None

        normalized = value.strip().replace("-", "_").replace(" ", "_").lower()
        return next((m for m in cls if m.value == normalized), None) or {
            "ROBUST": cls.ROBUSTNESS,
            "ERROR_HANDLING": cls.ROBUSTNESS,
            "ERRORHANDLING": cls.ROBUSTNESS,
            "MAINTAIN": cls.MAINTAINABILITY,
            "DOCS": cls.DOCUMENTATION,
        }.get(normalized.upper())


class PlanStep(BaseModel):
    """A single refactoring action."""

    description: str
    focus: PlanStepFocus = PlanStepFocus.OTHER
    success_criteria: str | None = None


class RefactoringPlanModel(BaseModel):
    """High-level refactoring objective with ordered execution steps."""

    objective: str
    steps: list[PlanStep] = Field(default_factory=list)


class RefactoredArtifact(BaseModel):
    """Result of applying a refactoring plan."""

    code: PythonCode
    explanation: str
    key_changes: list[str] = Field(default_factory=list)


class LintingReport(BaseModel):
    """Linting quality assessment and related issues."""

    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)


class ComplexityReport(BaseModel):
    """Cyclomatic complexity assessment."""

    score: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class TypingReport(BaseModel):
    """Type annotation coverage report."""

    score: float = Field(ge=0.0, le=1.0)


class DocumentationReport(BaseModel):
    """Docstring coverage report."""

    score: float = Field(ge=0.0, le=1.0)


class QualityMetrics(BaseModel):
    """Aggregated code quality metrics."""

    linting: LintingReport
    complexity: ComplexityReport
    typing: TypingReport
    documentation: DocumentationReport


class RecommendationPriority(StrEnum):
    """Priority describing urgency of acting on a recommendation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def _missing_(cls, value: object) -> RecommendationPriority | None:
        """Handle priority normalization."""
        return (
            next((m for m in cls if m.value == value.strip().lower()), None)
            if isinstance(value, str)
            else None
        )


class AssessmentRecommendation(BaseModel):
    """Actionable recommendation produced during final assessment."""

    area: Literal["linting", "complexity", "typing", "documentation", "functional", "general"] = (
        "general"
    )
    message: str
    priority: RecommendationPriority = RecommendationPriority.MEDIUM


class QualityAssessment(BaseModel):
    """Engineer-facing assessment accompanying the quantitative score."""

    outcome: Literal["safe_to_apply", "needs_revision", "unsafe"]
    rationale: str
    risks: list[str] = Field(default_factory=list)


__all__ = [
    "AssessmentRecommendation",
    "CodeAnalysisReport",
    "ComplexityReport",
    "DocumentationReport",
    "LintingReport",
    "OpportunityCategory",
    "PlanStep",
    "PlanStepFocus",
    "PythonCode",
    "QualityAssessment",
    "QualityMetrics",
    "RecommendationPriority",
    "RefactoredArtifact",
    "RefactoringOpportunity",
    "RefactoringPlanModel",
    "TypingReport",
]
