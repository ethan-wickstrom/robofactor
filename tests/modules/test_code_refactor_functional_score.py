from robofactor.data.models import TestCase
from robofactor.modules.code_refactor import _compute_functional_score


def test_compute_functional_score_all_pass():
    """Test that functional score is 1.0 when all tests pass."""
    code = 'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n'
    test_cases = [
        TestCase(args=[1, 2], kwargs={}, expected_output=3),
        TestCase(args=[0, 0], kwargs={}, expected_output=0),
        TestCase(args=[-1, 1], kwargs={}, expected_output=0),
    ]
    score = _compute_functional_score(code, test_cases)
    assert score == 1.0


def test_compute_functional_score_partial_pass():
    """Test that functional score is proportional to pass rate."""
    code = (
        "def add(a: int, b: int) -> int:\n"
        '    """Add two numbers."""\n'
        "    return a + b + 1\n"  # Intentionally wrong
    )
    test_cases = [
        TestCase(args=[1, 2], kwargs={}, expected_output=3),  # Fails: returns 4
        TestCase(args=[0, -1], kwargs={}, expected_output=-1),  # Fails: returns 0
    ]
    score = _compute_functional_score(code, test_cases)
    assert score == 0.0


def test_compute_functional_score_no_tests():
    """Test that score is 0.0 when no tests provided."""
    code = "def foo(): pass"
    score = _compute_functional_score(code, [])
    assert score == 0.0


def test_compute_functional_score_invalid_syntax():
    """Test that score is 0.0 for invalid syntax."""
    code = "def foo("  # Incomplete
    test_cases = [TestCase(args=[], kwargs={}, expected_output=None)]
    score = _compute_functional_score(code, test_cases)
    assert score == 0.0
