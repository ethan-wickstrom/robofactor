from returns.result import Failure, Success

from robofactor.data.models import TestCase
from robofactor.evaluation import evaluate_refactored_code


def test_evaluate_refactored_code_success_with_passing_tests():
    code = (
        "def add(a: int, b: int) -> int:\n"
        "    \"\"\"Return the sum of two integers.\"\"\"\n"
        "    return a + b\n"
    )
    tests = [
        TestCase(args=[1, 2], kwargs={}, expected_output=3),
        TestCase(args=[-1, 1], kwargs={}, expected_output=0),
    ]
    res = evaluate_refactored_code(code, tests)
    assert isinstance(res, Success)
    out = res.unwrap()
    assert out.func_name == "add"
    assert out.functional_check.passed_tests == 2
    assert out.functional_check.total_tests == 2


def test_evaluate_refactored_code_syntax_failure():
    bad_code = "x = 1"  # no top-level function
    res = evaluate_refactored_code(bad_code, [])
    assert isinstance(res, Failure)
    assert "Syntax Check Failed" in res.failure()

