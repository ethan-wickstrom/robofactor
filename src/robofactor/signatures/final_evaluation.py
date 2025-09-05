from typing import Literal

import dspy


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
