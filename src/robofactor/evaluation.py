"""
Data models and core logic for evaluating refactored code.

This module defines the structures for test cases, quality scores, and
evaluation results, and contains the pure function for performing the evaluation.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import dspy
from pydantic import BaseModel, Field

from . import analysis_utils


class TestCase(BaseModel):
    """A single, executable test case for a function."""

    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    expected_output: Any


class CodeQualityScores(BaseModel):
    """Holds various code quality metrics."""

    linting_score: float
    complexity_score: float
    typing_score: float
    docstring_score: float
    linting_issues: list[str] = Field(default_factory=list)


class SyntaxCheckResult(NamedTuple):
    """Encapsulates the result of a syntax check."""

    is_valid: bool
    func_name: str | None
    error_message: str | None


class FunctionalCheckResult(NamedTuple):
    """Encapsulates the result of functional correctness tests."""

    passed_tests: int
    total_tests: int


class EvaluationResult(NamedTuple):
    """Holds all evaluation results for a piece of refactored code."""

    code: str
    syntax_check: SyntaxCheckResult
    quality_scores: CodeQualityScores | None
    functional_check: FunctionalCheckResult | None


def evaluate_refactoring(prediction: dspy.Prediction, example: dspy.Example) -> EvaluationResult:
    """
    Performs a full evaluation of the refactored code without any I/O.

    Args:
        prediction: The dspy.Prediction object containing the refactored code.
        example: The dspy.Example object containing test cases.

    Returns:
        An EvaluationResult object with all analysis data.
    """
    code = analysis_utils._extract_python_code(prediction.refactored_code)
    is_valid, func_name, err = analysis_utils.check_syntax(code)
    syntax_result = SyntaxCheckResult(is_valid, func_name, err)

    if not is_valid:
        return EvaluationResult(
            code=code,
            syntax_check=syntax_result,
            quality_scores=None,
            functional_check=None,
        )

    raw_tests = example.get("test_cases")
    tests = [TestCase(**tc) for tc in raw_tests] if raw_tests else []

    if not tests:  # Module-level refactoring without specific tests
        quality = analysis_utils.check_code_quality(code)
        functional_result = FunctionalCheckResult(passed_tests=0, total_tests=0)
    else:  # Function-level refactoring with tests
        if not func_name:
            err_msg = "Tests provided, but no function found in code snippet."
            return EvaluationResult(
                code=code,
                syntax_check=SyntaxCheckResult(is_valid, None, err_msg),
                quality_scores=None,
                functional_check=None,
            )

        quality = analysis_utils.check_code_quality(code, func_name)
        passed_count = analysis_utils.check_functional_correctness(code, func_name, tests)
        functional_result = FunctionalCheckResult(passed_count, len(tests))

    return EvaluationResult(
        code=code,
        syntax_check=syntax_result,
        quality_scores=quality,
        functional_check=functional_result,
    )
