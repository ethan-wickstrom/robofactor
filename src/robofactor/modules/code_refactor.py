import dspy
from returns.result import Failure, Success

from robofactor.checks import BehaviorTest
from robofactor.refactor_check import (
    check_candidate_code,
    check_code_syntax,
    quality_metrics_for_code,
)
from robofactor.signatures.code_analysis import CodeAnalysis
from robofactor.signatures.final_assessment import FinalAssessment
from robofactor.signatures.refactored_code import RefactoredCode
from robofactor.signatures.refactoring_plan import RefactoringPlan
from robofactor.types import (
    AssessmentRecommendation,
    CodeAnalysisReport,
    ComplexityReport,
    DocumentationReport,
    LintingReport,
    PythonCode,
    QualityAssessment,
    QualityMetrics,
    RecommendationPriority,
    RefactoredArtifact,
    RefactoringPlanModel,
    TypingReport,
)


def _compute_functional_score(
    refactored_code: PythonCode, behavior_tests: tuple[BehaviorTest, ...]
) -> float:
    """Compute test pass rate for refactored code."""
    if not behavior_tests:
        return 0.0

    result = check_candidate_code(refactored_code, behavior_tests)
    if isinstance(result, Success):
        return result.unwrap().behavior.pass_rate
    if isinstance(result, Failure):
        return 0.0
    raise AssertionError(f"unexpected candidate check result: {result!r}")


def _create_syntax_error_prediction(
    analysis_report: CodeAnalysisReport,
    plan: RefactoringPlanModel,
    artifact: RefactoredArtifact,
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
        assessment=QualityAssessment(
            outcome="unsafe",
            rationale=f"Generated code has syntax errors: {error_message}",
            risks=["Code cannot be executed", "Invalid Python syntax"],
        ),
        recommendations=[
            AssessmentRecommendation(
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
        self.assessor = dspy.Predict(FinalAssessment)

    def forward(
        self, code_snippet: PythonCode, behavior_tests: tuple[BehaviorTest, ...]
    ) -> dspy.Prediction:
        analysis_result = self.analyzer(code_snippet=code_snippet)
        plan_result = self.planner(code_snippet=code_snippet, analysis=analysis_result.report)
        impl_result = self.implementer(original_code=code_snippet, plan=plan_result.plan)

        match check_code_syntax(impl_result.artifact.code):
            case Failure(error_msg):
                return _create_syntax_error_prediction(
                    analysis_result.report, plan_result.plan, impl_result.artifact, error_msg
                )
            case Success(_):
                pass

        match check_candidate_code(impl_result.artifact.code, behavior_tests):
            case Success(result):
                quality_metrics = result.quality
                functional_score = result.behavior.pass_rate if behavior_tests else 0.0
            case Failure():
                quality_metrics = quality_metrics_for_code(impl_result.artifact.code)
                functional_score = 0.0

        assessment_result = self.assessor(
            code_snippet=impl_result.artifact.code,
            quality_metrics=quality_metrics,
            functional_score=functional_score,
        )

        return dspy.Prediction(
            analysis_report=analysis_result.report,
            plan=plan_result.plan,
            artifact=impl_result.artifact,
            quality_metrics=quality_metrics,
            final_score=assessment_result.final_score,
            assessment=assessment_result.assessment,
            recommendations=assessment_result.recommendations,
        )
