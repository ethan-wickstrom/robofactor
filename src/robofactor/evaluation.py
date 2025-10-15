"""Railway-oriented evaluation pipeline for refactored code."""

from typing import NamedTuple

from returns.result import Result, safe

from . import analysis
from .data import models
from .types import PythonCode, QualityMetrics


class FunctionalCheckResult(NamedTuple):
    """Result of functional correctness testing."""

    passed_tests: int
    total_tests: int


class EvaluationResult(NamedTuple):
    """Complete evaluation results for refactored code."""

    code: PythonCode
    func_name: str
    quality_metrics: QualityMetrics
    functional_check: FunctionalCheckResult


def evaluate_refactored_code(
    code: PythonCode, tests: list[models.TestCase]
) -> Result[EvaluationResult, str]:
    """Evaluate refactored code through syntax, quality, and functional checks.

    Uses railway-oriented programming: any step failure short-circuits the pipeline.

    Returns:
        Success with EvaluationResult or Failure with error message.
    """
    return analysis.check_syntax(code).bind(
        lambda func_name: safe(lambda: analysis.check_code_quality(code))()
        .alt(lambda e: f"Quality check failed: {e}")
        .bind(
            lambda quality: safe(
                lambda: analysis.check_functional_correctness(code, func_name, tests)
            )()
            .alt(lambda e: f"Functional check failed: {e}")
            .map(
                lambda passed: EvaluationResult(
                    code=code,
                    func_name=func_name,
                    quality_metrics=quality,
                    functional_check=FunctionalCheckResult(
                        passed_tests=passed,
                        total_tests=len(tests),
                    ),
                )
            )
        )
    )


__all__ = [
    "EvaluationResult",
    "FunctionalCheckResult",
    "evaluate_refactored_code",
]
