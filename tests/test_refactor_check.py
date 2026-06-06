import dspy
from returns.result import Failure, Success

from robofactor.checks import BehaviorTest
from robofactor.refactor_check import check_candidate_code
from robofactor.types import PythonCode


def _python_code(source: str) -> PythonCode:
    return dspy.Code(code=source)


def test_check_candidate_code_success_with_passing_tests() -> None:
    code = _python_code(
        "def add(a: int, b: int) -> int:\n"
        '    """Return the sum of two integers."""\n'
        "    return a + b\n"
    )
    tests = (
        BehaviorTest(case_id="add.positive", args=(1, 2), kwargs={}, expected_output=3),
        BehaviorTest(case_id="add.zero", args=(-1, 1), kwargs={}, expected_output=0),
    )

    result = check_candidate_code(code, tests)

    assert isinstance(result, Success)
    checked = result.unwrap()
    assert checked.function_name == "add"
    assert checked.behavior.passed == 2
    assert checked.behavior.total == 2


def test_check_candidate_code_reports_syntax_failure() -> None:
    result = check_candidate_code(_python_code("x = 1"), ())

    assert isinstance(result, Failure)
    assert "top-level function" in result.failure()
