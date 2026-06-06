import dspy

from robofactor.types import (
    AssessmentRecommendation,
    PythonCode,
    QualityAssessment,
    QualityMetrics,
)


class FinalAssessment(dspy.Signature):
    """
    Assess the refactored code using quantitative metrics and test results.
    Provide a weighted quality score and structured recommendations.
    """

    code_snippet: PythonCode = dspy.InputField(desc="Refactored Python code")
    quality_metrics: QualityMetrics = dspy.InputField(desc="Structured quality metrics")
    functional_score: float = dspy.InputField(desc="Test pass rate (0.0-1.0)")
    final_score: float = dspy.OutputField(
        description="Weighted quality score (0.0-1.0)", ge=0.0, le=1.0
    )
    assessment: QualityAssessment = dspy.OutputField(description="Overall qualitative assessment")
    recommendations: list[AssessmentRecommendation] = dspy.OutputField(
        description="Actionable follow-up recommendations"
    )
