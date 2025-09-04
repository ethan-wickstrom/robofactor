import logging
from typing import Literal

import dspy

FAILURE_SCORE = 0.0
TRAINING_DATA_FILE = "training.json"
logger = logging.getLogger(__name__)


class CodeAnalysis(dspy.Signature):
    """
    Analyze Python code for its purpose, complexity, and dependencies.
    Identify actionable refactoring opportunities and summarize the findings.
    """
    code_snippet: dspy.Code[Literal["python"]] = dspy.InputField(desc="Python code to analyze")
    analysis: str = dspy.OutputField(description="Summary of functionality, complexity, and dependencies")
    refactoring_opportunities: list[str] = dspy.OutputField(
        description="Actionable bullet points for refactoring"
    )


class RefactoringPlan(dspy.Signature):
    """
    Formulate a high-level refactoring goal and sequential action plan based on given code and analysis.
    Return a clear summary and an ordered list of steps.
    """
    code_snippet: dspy.Code[Literal["python"]] = dspy.InputField(desc="Original Python code")
    analysis: str = dspy.InputField(desc="Code analysis summary")
    refactoring_summary: str = dspy.OutputField(description="High-level refactoring objective")
    plan_steps: list[str] = dspy.OutputField(description="Sequential actions to achieve refactoring")


class RefactoredCode(dspy.Signature):
    """
    Apply the refactoring plan to produce improved code.
    Output PEP8-compliant Python with type hints, docstrings, and a rationale for the changes.
    """
    original_code: dspy.Code[Literal["python"]] = dspy.InputField(desc="Unmodified source code")
    refactoring_summary: str = dspy.InputField(desc="Refactoring objective")
    plan_steps: list[str] = dspy.InputField(desc="Step-by-step refactoring actions")
    refactored_code: dspy.Code[Literal["python"]] = dspy.OutputField(
        description="PEP8-compliant Python code with type hints and docstrings"
    )
    implementation_explanation: str = dspy.OutputField(
        description="Rationale for implemented changes"
    )


class FinalEvaluation(dspy.Signature):
    """
    Assess the refactored code using quantitative metrics and test results.
    Provide a weighted quality score and actionable final suggestions.
    """
    code_snippet: dspy.Code[Literal["python"]] = dspy.InputField(desc="Refactored Python code")
    quality_scores: str = dspy.InputField(desc="JSON quality metrics")
    functional_score: float = dspy.InputField(desc="Test pass rate (0.0-1.0)")
    final_score: float = dspy.OutputField(
        description="Weighted quality score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    final_suggestion: str = dspy.OutputField(
        description="Improvement recommendations or approval"
    )


class CodeRefactor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(CodeAnalysis)
        self.planner = dspy.Predict(RefactoringPlan)
        self.implementer = dspy.Predict(RefactoredCode)
        self.evaluator = dspy.Predict(FinalEvaluation)

    def forward(self, code_snippet: str) -> dspy.Prediction:
        analysis_result = self.analyzer(code_snippet=code_snippet)
        plan_result = self.planner(
            code_snippet=code_snippet,
            analysis=analysis_result.analysis.analysis
        )
        impl_result = self.implementer(
            original_code=code_snippet,
            refactoring_summary=plan_result.plan.refactoring_summary,
            plan_steps=plan_result.plan.plan_steps,
        )
        eval_result = self.evaluator(
            code_snippet=impl_result.implementation.refactored_code,
            quality_scores=impl_result.implementation.quality_scores,
            functional_score=impl_result.implementation.functional_score,
        )
        return dspy.Prediction(
            analysis=analysis_result.analysis.analysis,
            refactoring_opportunities=analysis_result.analysis.refactoring_opportunities,
            refactoring_summary=plan_result.plan.refactoring_summary,
            plan_steps=plan_result.plan.plan_steps,
            refactored_code=impl_result.implementation.refactored_code,
            implementation_explanation=impl_result.implementation.implementation_explanation,
            final_score=eval_result.final_score,
            final_suggestion=eval_result.final_suggestion,
        )
