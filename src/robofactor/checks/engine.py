from __future__ import annotations

from collections.abc import Sequence

from returns.result import Failure, Result, Success

from robofactor.checks.behavior import (
    comparison_failures,
    explicit_behavior_failures,
    generated_comparison_failures,
)
from robofactor.checks.model import BehaviorTest, CheckFailure, CheckReport, CheckSummary
from robofactor.checks.quality import python_quality, suppression_warnings
from robofactor.checks.signature import parse_function_signature


def check_candidate(code: str, tests: Sequence[BehaviorTest]) -> Result[CheckReport, str]:
    signature_result = parse_function_signature(code)
    match signature_result:
        case Failure(reason):
            return Failure(reason)
        case Success(signature):
            pass

    suppression_failures = tuple(
        CheckFailure(check="suppression_comment", case_id="candidate", message=warning)
        for warning in suppression_warnings(code)
    )
    behavior_failures = explicit_behavior_failures(code, signature.name, tuple(tests))
    failures = (*behavior_failures, *suppression_failures)
    return Success(
        CheckReport(
            passed=not failures,
            function_name=signature.name,
            signature=signature,
            explicit_behavior=_summary(len(tests), behavior_failures),
            source_comparisons=CheckSummary(passed=0, failed=0, total=0),
            generated_comparisons=CheckSummary(passed=0, failed=0, total=0),
            quality=python_quality(code, signature.name),
            failures=failures,
            warnings=(),
        )
    )


def check_refactor(
    source_code: str,
    refactored_code: str,
    tests: Sequence[BehaviorTest],
) -> Result[CheckReport, str]:
    source_signature_result = parse_function_signature(source_code)
    refactored_signature_result = parse_function_signature(refactored_code)
    match (source_signature_result, refactored_signature_result):
        case (Failure(reason), _):
            return Failure(f"source signature could not be read: {reason}")
        case (_, Failure(reason)):
            return Failure(f"refactored signature could not be read: {reason}")
        case (Success(source_signature), Success(refactored_signature)):
            pass

    if source_signature != refactored_signature:
        return Failure(
            "function signature changed: "
            f"expected {source_signature.render()}, got {refactored_signature.render()}"
        )

    behavior_tests = tuple(tests)
    explicit_failures = explicit_behavior_failures(
        refactored_code,
        refactored_signature.name,
        behavior_tests,
    )
    comparison_failures_ = comparison_failures(
        source_code,
        refactored_code,
        refactored_signature.name,
        behavior_tests,
        check="source_comparison",
    )
    generated_failures = generated_comparison_failures(
        source_code,
        refactored_code,
        refactored_signature.name,
        behavior_tests,
    )
    suppression_failures = tuple(
        CheckFailure(check="suppression_comment", case_id="refactored", message=warning)
        for warning in suppression_warnings(refactored_code)
    )
    failures = (
        *explicit_failures,
        *comparison_failures_,
        *generated_failures,
        *suppression_failures,
    )
    generated_total = 1 if behavior_tests else 0
    generated_failed = len(generated_failures)
    return Success(
        CheckReport(
            passed=not failures,
            function_name=refactored_signature.name,
            signature=refactored_signature,
            explicit_behavior=_summary(len(behavior_tests), explicit_failures),
            source_comparisons=_summary(len(behavior_tests), comparison_failures_),
            generated_comparisons=CheckSummary(
                passed=generated_total - generated_failed,
                failed=generated_failed,
                total=generated_total,
            ),
            quality=python_quality(refactored_code, refactored_signature.name),
            failures=failures,
            warnings=(),
        )
    )


def _summary(total: int, failures: tuple[CheckFailure, ...]) -> CheckSummary:
    failed = len(failures)
    return CheckSummary(passed=total - failed, failed=failed, total=total)
