"""
Data models and core logic for evaluating refactored code.

This module defines the structures for test cases, quality scores, and
evaluation results, and contains the pure function for performing the evaluation.
It leverages the 'returns' library for robust, type-safe error handling using
a railway-oriented programming approach.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from pydantic import BaseModel, Field
from returns.result import Failure, Result, Success, do, safe

from . import analysis


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


class FunctionalCheckResult(NamedTuple):
    """Encapsulates the result of functional correctness tests."""

    passed_tests: int
    total_tests: int


class EvaluationResult(NamedTuple):
    """Holds all successful evaluation results for a piece of refactored code."""

    code: str
    func_name: str
    quality_scores: CodeQualityScores
    functional_check: FunctionalCheckResult


def _check_syntax(code: str) -> Result[str, str]:
    """
    Checks for valid Python syntax and returns the function name if valid.

    This function wraps the analysis function to convert its tuple-based
    output into a `Result` monad, which is more suitable for functional
    pipelines.
    """
    is_valid, func_name, err = analysis.check_syntax(code)
    if not is_valid or not func_name:
        return Failure(f"Syntax Check Failed: {err or 'No function found.'}")
    return Success(func_name)


@safe
def _check_quality(code: str, func_name: str) -> CodeQualityScores:
    """
    Checks code quality and returns the scores.

    The `@safe` decorator automatically wraps this function's execution in a
    `Result` container, capturing any exceptions as a `Failure`.
    """
    return analysis.check_code_quality(code, func_name)


@safe
def _check_functional_correctness(
    code: str, func_name: str, tests: list[TestCase]
) -> FunctionalCheckResult:
    """
    Runs functional tests and returns the pass rate.

    The `@safe` decorator captures any exceptions during test execution.
    """
    if not tests:
        return FunctionalCheckResult(passed_tests=0, total_tests=0)

    passed_tests = analysis.check_functional_correctness(code, func_name, tests)
    return FunctionalCheckResult(passed_tests=passed_tests, total_tests=len(tests))


@do(Result[EvaluationResult, str])
def evaluate_refactored_code(
    code: str, tests: list[TestCase]
) -> Result[EvaluationResult, str]:
    """
    Performs a full evaluation of the refactored code.

    This function orchestrates a pipeline of checks (syntax, quality, functional)
    using a declarative, railway-oriented approach with `returns`'s `do` notation.
    If any step fails, the entire pipeline short-circuits and returns the error.

    Args:
        code: The refactored Python code to evaluate.
        tests: A list of test cases to verify functional correctness.

    Returns:
        A `Result` container:
        - `Success(EvaluationResult)` if all checks pass.
        - `Failure(str)` with a descriptive error message if any check fails.
    """
    func_name = yield _check_syntax(code)

    quality_scores = yield _check_quality(code, func_name).alt(
        lambda e: f"Quality Check Failed: {e}"
    )

    functional_check = yield _check_functional_correctness(code, func_name, tests).alt(
        lambda e: f"Functional Check Failed: {e}"
    )

    return EvaluationResult(
        code=code,
        func_name=func_name,
        quality_scores=quality_scores,
        functional_check=functional_check,
    )
