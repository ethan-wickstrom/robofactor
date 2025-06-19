"""
Data models and core logic for evaluating refactored code.

This module defines the structures for test cases, quality scores, and
evaluation results, and contains the pure function for performing the evaluation.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from pydantic import BaseModel, Field

from . import analysis
from .functional_types import Err, Ok, Result


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
    """Checks for valid Python syntax and returns the function name if valid."""
    is_valid, func_name, err = analysis.check_syntax(code)
    if not is_valid or not func_name:
        return Err(f"Syntax Check Failed: {err or 'No function found.'}")
    return Ok(func_name)


def _check_quality(code: str, func_name: str) -> Result[CodeQualityScores, str]:
    """Checks code quality and returns the scores if successful."""
    try:
        quality = analysis.check_code_quality(code, func_name)
        return Ok(quality)
    except Exception as e:
        return Err(f"Quality Check Failed: {e}")


def _check_functional_correctness(
    code: str, func_name: str, tests: list[TestCase]
) -> Result[FunctionalCheckResult, str]:
    """Runs functional tests and returns the pass rate if successful."""
    if not tests:
        return Ok(FunctionalCheckResult(passed_tests=0, total_tests=0))

    try:
        passed_tests = analysis.check_functional_correctness(code, func_name, tests)
        return Ok(FunctionalCheckResult(passed_tests, len(tests)))
    except Exception as e:
        return Err(f"Functional Check Failed: {e}")


def evaluate_refactored_code(
    code: str, tests: list[TestCase]
) -> Result[EvaluationResult, str]:
    """
    Performs a full evaluation of the refactored code.

    This function orchestrates a pipeline of checks (syntax, quality, functional)
    and returns a comprehensive result. It uses a functional, railway-oriented
    approach with the `Result` type to handle potential failures at each stage.

    Args:
        code: The refactored Python code to evaluate.
        tests: A list of test cases to verify functional correctness.

    Returns:
        - Ok(EvaluationResult) if all checks pass.
        - Err(str) with a descriptive error message if any check fails.
    """
    syntax_result = _check_syntax(code)
    if isinstance(syntax_result, Err):
        return syntax_result
    func_name = syntax_result.value

    quality_result = _check_quality(code, func_name)
    if isinstance(quality_result, Err):
        return quality_result

    functional_result = _check_functional_correctness(code, func_name, tests)
    if isinstance(functional_result, Err):
        return functional_result

    return Ok(
        EvaluationResult(
            code=code,
            func_name=func_name,
            quality_scores=quality_result.value,
            functional_check=functional_result.value,
        )
    )
