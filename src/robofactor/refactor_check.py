from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from robofactor.checks import (
    BehaviorTest,
    CheckReport,
    CheckSummary,
    QualityReport,
    check_candidate,
    check_refactor,
    parse_function_signature,
    python_quality,
)
from robofactor.types import (
    ComplexityReport,
    DocumentationReport,
    LintingReport,
    PythonCode,
    QualityMetrics,
    TypingReport,
)


@dataclass(frozen=True, kw_only=True)
class CheckedRefactor:
    code: PythonCode
    function_name: str
    quality: QualityMetrics
    behavior: CheckSummary


def check_candidate_code(
    code: PythonCode,
    behavior_tests: Sequence[BehaviorTest],
) -> Result[CheckedRefactor, str]:
    return _checked_refactor(code, check_candidate(code.code, behavior_tests))


def check_refactored_code(
    source_code: str,
    refactored_code: PythonCode,
    behavior_tests: Sequence[BehaviorTest],
) -> Result[CheckedRefactor, str]:
    result = check_refactor(source_code, refactored_code.code, behavior_tests)
    match result:
        case Failure(reason):
            return Failure(reason)
        case Success(report) if report.passed:
            return Success(_checked_refactor_from_report(refactored_code, report))
        case Success(report):
            failures = "; ".join(failure.message for failure in report.failures[:3])
            return Failure(f"Behavior preservation failed: {failures}")
    raise AssertionError(f"unexpected refactor check result: {result!r}")


def check_code_syntax(code: PythonCode) -> Result[str, str]:
    return parse_function_signature(code.code).map(lambda signature: signature.name)


def quality_metrics_for_code(code: PythonCode) -> QualityMetrics:
    signature = parse_function_signature(code.code).unwrap()
    return quality_metrics_from_report(python_quality(code.code, signature.name))


def quality_metrics_from_report(quality: QualityReport) -> QualityMetrics:
    return QualityMetrics(
        linting=LintingReport(
            score=1.0 if quality.ruff_passed else 0.0,
            issues=list(quality.ruff_issues),
        ),
        complexity=ComplexityReport(
            score=quality.complexity_score,
            warnings=list(quality.complexity_issues),
        ),
        typing=TypingReport(score=quality.typing_score),
        documentation=DocumentationReport(score=quality.docstring_score),
    )


def _checked_refactor(
    code: PythonCode,
    result: Result[CheckReport, str],
) -> Result[CheckedRefactor, str]:
    match result:
        case Failure(reason):
            return Failure(reason)
        case Success(report):
            return Success(_checked_refactor_from_report(code, report))
    raise AssertionError(f"unexpected candidate check result: {result!r}")


def _checked_refactor_from_report(code: PythonCode, report: CheckReport) -> CheckedRefactor:
    return CheckedRefactor(
        code=code,
        function_name=report.function_name,
        quality=quality_metrics_from_report(report.quality),
        behavior=report.explicit_behavior,
    )


__all__ = [
    "CheckedRefactor",
    "check_candidate_code",
    "check_code_syntax",
    "check_refactored_code",
    "quality_metrics_for_code",
    "quality_metrics_from_report",
]
