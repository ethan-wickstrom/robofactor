import dspy

from robofactor.checks import BehaviorTest
from robofactor.modules.code_refactor import _compute_functional_score
from robofactor.types import PythonCode


def _python_code(source: str) -> PythonCode:
    return dspy.Code(code=source)


def test_compute_functional_score_all_pass():
    """Test that functional score is 1.0 when all tests pass."""
    code = _python_code(
        'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n'
    )
    behavior_tests = (
        BehaviorTest(case_id="add.positive", args=(1, 2), kwargs={}, expected_output=3),
        BehaviorTest(case_id="add.zero", args=(0, 0), kwargs={}, expected_output=0),
        BehaviorTest(case_id="add.mixed", args=(-1, 1), kwargs={}, expected_output=0),
    )
    score = _compute_functional_score(code, behavior_tests)
    assert score == 1.0


def test_compute_functional_score_partial_pass():
    """Test that functional score is proportional to pass rate."""
    code = _python_code(
        'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b + 1\n'
    )
    behavior_tests = (
        BehaviorTest(case_id="add.positive", args=(1, 2), kwargs={}, expected_output=3),
        BehaviorTest(case_id="add.negative", args=(0, -1), kwargs={}, expected_output=-1),
    )
    score = _compute_functional_score(code, behavior_tests)
    assert score == 0.0


def test_compute_functional_score_no_tests():
    """Test that score is 0.0 when no tests provided."""
    code = _python_code("def foo(): pass")
    score = _compute_functional_score(code, ())
    assert score == 0.0


def test_compute_functional_score_invalid_syntax():
    """Test that score is 0.0 for invalid syntax."""
    code = _python_code("def foo(")
    behavior_tests = (BehaviorTest(case_id="foo.empty", args=(), kwargs={}, expected_output=None),)
    score = _compute_functional_score(code, behavior_tests)
    assert score == 0.0
