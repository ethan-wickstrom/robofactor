import logging
from collections.abc import Sequence

import dspy
from pydantic import BaseModel, Field, field_validator, model_validator
from returns.result import Result, Success

from . import analysis, evaluation
from .evaluation import EvaluationResult

# --- Constants ---
FAILURE_SCORE = 0.0
logger = logging.getLogger(__name__)


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
        return analysis.extract_python_code(v)


class EvaluationOutput(BaseModel):
    """Holistic assessment of refactoring quality."""

    final_score: float = Field(description="Weighted quality score (0.0-1.0)", ge=0.0, le=1.0)
    final_suggestion: str = Field(description="Improvement recommendations or approval")

    @model_validator(mode="after")
    def validate_score_precision(self) -> "EvaluationOutput":
        if isinstance(self.final_score, float):
            self.final_score = round(self.final_score, 2)
        return self


# --- DSPy Signatures ---
class CodeAnalysis(dspy.Signature):
    """Analyze Python code for functionality and improvement areas."""

    code_snippet: str = dspy.InputField(desc="Python code to analyze")
    analysis: AnalysisOutput = dspy.OutputField()


class RefactoringPlan(dspy.Signature):
    """Create refactoring plan based on code analysis."""

    code_snippet: str = dspy.InputField(desc="Original Python code")
    analysis: str = dspy.InputField(desc="Code analysis summary")
    plan: PlanOutput = dspy.OutputField()


class RefactoredCode(dspy.Signature):
    """Generate refactored code from execution plan."""

    original_code: str = dspy.InputField(desc="Unmodified source code")
    refactoring_summary: str = dspy.InputField(desc="Refactoring objective")
    plan_steps: Sequence[str] = dspy.InputField(desc="Step-by-step refactoring actions")
    implementation: ImplementationOutput = dspy.OutputField()


class FinalEvaluation(dspy.Signature):
    """Assess refactored code quality with quantitative metrics."""

    code_snippet: str = dspy.InputField(desc="Refactored Python code")
    quality_scores: str = dspy.InputField(desc="JSON quality metrics")
    functional_score: float = dspy.InputField(desc="Test pass rate (0.0-1.0)")
    evaluation: EvaluationOutput = dspy.OutputField()


# --- DSPy Modules ---
class CodeRefactor(dspy.Module):
    """Orchestrates code analysis, planning, and refactoring."""

    def __init__(self):
        super().__init__()
        self.analyzer: dspy.Module = dspy.Predict(CodeAnalysis)
        self.planner: dspy.Module = dspy.Predict(RefactoringPlan)
        self.implementer: dspy.Module = dspy.Predict(RefactoredCode)

    def forward(self, code_snippet: str) -> dspy.Prediction:
        analysis_result = self.analyzer(code_snippet=code_snippet)
        plan_result = self.planner(
            code_snippet=code_snippet, analysis=analysis_result.analysis.analysis
        )
        impl_result = self.implementer(
            original_code=code_snippet,
            refactoring_summary=plan_result.plan.refactoring_summary,
            plan_steps=plan_result.plan.plan_steps,
        )

        return dspy.Prediction(
            analysis=analysis_result.analysis.analysis,
            refactoring_opportunities=analysis_result.analysis.refactoring_opportunities,
            refactoring_summary=plan_result.plan.refactoring_summary,
            plan_steps=plan_result.plan.plan_steps,
            refactored_code=impl_result.implementation.refactored_code,
            implementation_explanation=impl_result.implementation.implementation_explanation,
        )


class RefactoringEvaluator(dspy.Module):
    """Evaluates refactored code through automated checks and LLM assessment."""

    def __init__(self):
        super().__init__()
        self.evaluator: dspy.Module = dspy.Predict(FinalEvaluation)

    def _handle_evaluation_success(
        self, eval_data: EvaluationResult, refactored_code: str
    ) -> float:
        """Process successful programmatic evaluation."""
        functional_score = (
            eval_data.functional_check.passed_tests / eval_data.functional_check.total_tests
            if eval_data.functional_check.total_tests > 0
            else 1.0
        )

        try:
            llm_evaluation = self.evaluator(
                code_snippet=refactored_code,
                quality_scores=eval_data.quality_scores.model_dump_json(),
                functional_score=functional_score,
            )
            return llm_evaluation.evaluation.final_score
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}", exc_info=True)
            return FAILURE_SCORE

    def forward(self, original_example: dspy.Example, prediction: dspy.Prediction) -> float:
        refactored_code = getattr(prediction, "refactored_code", "")
        if not refactored_code:
            logger.warning("Evaluation aborted: Missing refactored code")
            return FAILURE_SCORE

        code_to_evaluate = analysis.extract_python_code(refactored_code)
        if not code_to_evaluate:
            logger.warning("Evaluation aborted: Empty code extraction")
            return FAILURE_SCORE

        test_cases = getattr(original_example, "test_cases", [])
        eval_result: Result[EvaluationResult, str] = evaluation.evaluate_refactored_code(
            code_to_evaluate, test_cases
        )

        if isinstance(eval_result, Success):
            return self._handle_evaluation_success(eval_result.unwrap(), code_to_evaluate)
        else:
            logger.warning(f"Programmatic evaluation failed: {eval_result.failure()}")
            return FAILURE_SCORE
