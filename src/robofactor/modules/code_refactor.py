import dspy

from robofactor import analysis
from robofactor.data.models import TestCase
from robofactor.signatures.code_analysis import CodeAnalysis
from robofactor.signatures.final_evaluation import FinalEvaluation
from robofactor.signatures.refactored_code import RefactoredCode
from robofactor.signatures.refactoring_plan import RefactoringPlan
from robofactor.types import (
    ComplexityReport,
    DocumentationReport,
    EvaluationRecommendation,
    LintingReport,
    PythonCode,
    QualityMetrics,
    QualityVerdict,
    RecommendationPriority,
    TypingReport,
)


def _compute_functional_score(refactored_code: PythonCode, test_cases: list[TestCase]) -> float:
    """Compute test pass rate for refactored code."""
    if not test_cases:
        return 0.0
    is_valid, func_name, _ = analysis.check_syntax(refactored_code)
    if not is_valid or not func_name:
        return 0.0
    passed = analysis.check_functional_correctness(refactored_code, func_name, test_cases)
    return passed / len(test_cases)


def _create_syntax_error_prediction(
    analysis_report: object,
    plan: object,
    artifact: object,
    error_message: str,
) -> dspy.Prediction:
    """Create prediction for syntax-invalid code."""
    return dspy.Prediction(
        analysis_report=analysis_report,
        plan=plan,
        artifact=artifact,
        quality_metrics=QualityMetrics(
            linting=LintingReport(score=0.0, issues=[f"Syntax Error: {error_message}"]),
            complexity=ComplexityReport(score=0.0, warnings=[]),
            typing=TypingReport(score=0.0),
            documentation=DocumentationReport(score=0.0),
        ),
        final_score=0.0,
        verdict=QualityVerdict(
            decision="reject",
            rationale=f"Generated code has syntax errors: {error_message}",
            risks=["Code cannot be executed", "Invalid Python syntax"],
        ),
        recommendations=[
            EvaluationRecommendation(
                area="functional",
                message=f"Fix syntax error: {error_message}",
                priority=RecommendationPriority.HIGH,
            )
        ],
    )


class CodeRefactor(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.analyzer = dspy.Predict(CodeAnalysis)
        self.planner = dspy.Predict(RefactoringPlan)
        self.implementer = dspy.Predict(RefactoredCode)
        self.evaluator = dspy.Predict(FinalEvaluation)

    def forward(
        self, code_snippet: PythonCode, test_cases: list[TestCase] | None = None
    ) -> dspy.Prediction:
        analysis_result = self.analyzer(code_snippet=code_snippet)
        plan_result = self.planner(code_snippet=code_snippet, analysis=analysis_result.report)
        impl_result = self.implementer(original_code=code_snippet, plan=plan_result.plan)

        # Validate syntax early to prevent downstream errors
        is_valid, _, error_msg = analysis.check_syntax(impl_result.artifact.code)
        if not is_valid:
            return _create_syntax_error_prediction(
                analysis_result.report, plan_result.plan, impl_result.artifact, error_msg or ""
            )

        quality_metrics = analysis.check_code_quality(impl_result.artifact.code)
        functional_score = _compute_functional_score(impl_result.artifact.code, test_cases or [])
        eval_result = self.evaluator(
            code_snippet=impl_result.artifact.code,
            quality_metrics=quality_metrics,
            functional_score=functional_score,
        )

        return dspy.Prediction(
            analysis_report=analysis_result.report,
            plan=plan_result.plan,
            artifact=impl_result.artifact,
            quality_metrics=quality_metrics,
            final_score=eval_result.final_score,
            verdict=eval_result.verdict,
            recommendations=eval_result.recommendations,
        )
