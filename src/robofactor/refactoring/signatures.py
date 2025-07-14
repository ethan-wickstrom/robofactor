"""
Defines the Pydantic data models and DSPy Signatures for the refactoring process.

This module specifies the structured inputs and outputs for each step of the
AI-powered refactoring pipeline, including analysis, planning, implementation,
and evaluation.
"""

from collections.abc import Sequence

import dspy
from pydantic import BaseModel, Field, field_validator, model_validator

from robofactor.common.code_extraction import extract_python_code


# --- Pydantic Models ---
class AnalysisOutput(BaseModel):
    """Structured analysis of Python code functionality and improvement opportunities."""

    analysis: str = Field(
        description="Concise summary of functionality, complexity, and dependencies"
    )
    refactoring_opportunities: Sequence[str] = Field(
        description="Actionable bullet points for refactoring"
    )


class PlanOutput(BaseModel):
    """Step-by-step refactoring execution plan."""

    refactoring_summary: str = Field(description="High-level refactoring objective")
    plan_steps: Sequence[str] = Field(description="Sequential actions to achieve refactoring")


class ImplementationOutput(BaseModel):
    """Final refactored code with change explanations."""

    refactored_code: str = Field(
        description="PEP8-compliant Python code with type hints and docstrings"
    )
    implementation_explanation: str = Field(description="Rationale for implemented changes")

    @field_validator("refactored_code")
    @classmethod
    def extract_from_markdown(cls, v: str) -> str:
        """Extracts Python code from a markdown code block."""
        return extract_python_code(v)


class EvaluationOutput(BaseModel):
    """Holistic assessment of refactoring quality."""

    final_score: float = Field(description="Weighted quality score (0.0-1.0)", ge=0.0, le=1.0)
    final_suggestion: str = Field(description="Improvement recommendations or approval")

    @model_validator(mode="after")
    def validate_score_precision(self) -> "EvaluationOutput":
        """Rounds the final score to two decimal places."""
        if isinstance(self.final_score, float):
            self.final_score = round(self.final_score, 2)
        return self


# --- DSPy Signatures ---
class CodeAnalysis(dspy.Signature):
    """
    Analyze Python code for functionality and improvement areas.

    **Instruction**: You are an expert code analyst. Your task is to thoroughly
    examine the provided Python code snippet. Identify its core functionality,
    dependencies, and complexity. Then, suggest concrete, actionable refactoring
    opportunities. The analysis should be concise, and the opportunities should be
    clear and directly implementable.
    """

    code_snippet: str = dspy.InputField(desc="Python code to analyze")
    analysis: AnalysisOutput = dspy.OutputField()


class RefactoringPlan(dspy.Signature):
    """
    Create a refactoring plan based on code analysis.

    **Instruction**: You are a senior software architect. Based on the provided code
    and its analysis, create a high-level refactoring plan. Define a clear
    objective for the refactoring and then break it down into a sequence of
    specific, logical steps. The plan should be easy to follow and lead to a
    measurably better version of the code.
    """

    code_snippet: str = dspy.InputField(desc="Original Python code")
    analysis: str = dspy.InputField(desc="Code analysis summary")
    plan: PlanOutput = dspy.OutputField()


class RefactoredCode(dspy.Signature):
    """
    Generate refactored code from an execution plan.

    **Instruction**: You are a world-class Python programmer. Your task is to
    implement the refactoring plan for the given code. The final code must be
    100% PEP8 compliant, include comprehensive docstrings (Google-style),
    and have full type hints. Provide a clear explanation of the changes you made
    and why. The refactored code must be enclosed in a single Python markdown block.
    """

    original_code: str = dspy.InputField(desc="Unmodified source code")
    refactoring_summary: str = dspy.InputField(desc="Refactoring objective")
    plan_steps: Sequence[str] = dspy.InputField(desc="Step-by-step refactoring actions")
    implementation: ImplementationOutput = dspy.OutputField()


class FinalEvaluation(dspy.Signature):
    """
    Assess refactored code quality with quantitative metrics.

    **Instruction**: You are a quality assurance automation bot. Evaluate the
    refactored code based on the provided quality and functional scores.
    Your assessment must result in a final score between 0.0 and 1.0,
    where 1.0 is a perfect refactoring. Provide a concluding suggestion,
    either approving the code or recommending specific further improvements.
    The final score should be a weighted average of the inputs.
    """

    code_snippet: str = dspy.InputField(desc="Refactored Python code")
    quality_scores: str = dspy.InputField(desc="JSON quality metrics")
    functional_score: float = dspy.InputField(desc="Test pass rate (0.0-1.0)")
    evaluation: EvaluationOutput = dspy.OutputField()
