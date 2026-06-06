from pathlib import Path

from returns.result import Failure, Success

from robofactor.checks import BehaviorTest, check_candidate, check_refactor
from robofactor.refactoring import AppliedChange, ApplyFailed, apply_change, inspect_rope_file


def test_check_refactor_passes_when_behavior_and_signature_match() -> None:
    source = "def add(left, right=0):\n    total = left + right\n    return total\n"
    refactored = "def add(left, right=0):\n    return left + right\n"
    tests = (BehaviorTest(case_id="add.public", args=(2,), kwargs={"right": 3}, expected_output=5),)

    result = check_refactor(source, refactored, tests)

    assert isinstance(result, Success)
    report = result.unwrap()
    assert report.passed
    assert report.explicit_behavior.passed == 1
    assert report.source_comparisons.passed == 1
    assert report.generated_comparisons.total == 1


def test_check_refactor_blocks_signature_changes() -> None:
    source = "def add(left, right=0):\n    return left + right\n"
    refactored = "def add(left):\n    return left\n"

    result = check_refactor(source, refactored, ())

    assert isinstance(result, Failure)
    assert "signature changed" in result.failure()


def test_check_refactor_generates_counterexamples_near_behavior_tests() -> None:
    source = "def clamp(value):\n    return value if value >= 0 else 0\n"
    refactored = "def clamp(value):\n    return value\n"
    tests = (BehaviorTest(case_id="clamp.public", args=(5,), kwargs={}, expected_output=5),)

    result = check_refactor(source, refactored, tests)

    assert isinstance(result, Success)
    report = result.unwrap()
    assert not report.passed
    assert report.explicit_behavior.passed == 1
    assert report.generated_comparisons.failed == 1
    assert report.failures[0].check == "generated_comparison"


def test_check_candidate_reports_behavior_and_quality() -> None:
    code = (
        "def add(left: int, right: int) -> int:\n"
        '    """Return the sum."""\n'
        "    return left + right\n"
    )
    tests = (BehaviorTest(case_id="add.public", args=(1, 2), kwargs={}, expected_output=3),)

    result = check_candidate(code, tests)

    assert isinstance(result, Success)
    report = result.unwrap()
    assert report.passed
    assert report.explicit_behavior.passed == 1
    assert report.quality.ruff_passed


def test_apply_change_writes_only_after_checks_pass(tmp_path: Path) -> None:
    source_path = tmp_path / "calculator.py"
    source_path.write_text("def add(left, right):\n    return left + right\n", encoding="utf-8")
    proposed = "def add(left, right):\n    return left - right\n"
    tests = (BehaviorTest(case_id="add.public", args=(4, 2), kwargs={}, expected_output=6),)

    result = apply_change(source_path, proposed, tests)

    assert isinstance(result, ApplyFailed)
    assert result.reason == "checks_failed"
    assert (
        source_path.read_text(encoding="utf-8")
        == "def add(left, right):\n    return left + right\n"
    )

    safe_result = apply_change(
        source_path, "def add(left, right):\n    return left + right\n", tests
    )

    assert isinstance(safe_result, AppliedChange)
    assert safe_result.changed_file == source_path


def test_inspect_rope_file_reports_module_name(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    module = package / "calculator.py"
    module.write_text("def add(left, right):\n    return left + right\n", encoding="utf-8")

    result = inspect_rope_file(tmp_path, module)

    assert isinstance(result, Success)
    report = result.unwrap()
    assert report.module_name == "package.calculator"
    assert report.resource_path == "package/calculator.py"
