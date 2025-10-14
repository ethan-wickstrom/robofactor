from __future__ import annotations

import json
import textwrap

import dspy

from robofactor.data import models
from robofactor.types import PythonCode


def _build_execution_script(func_name: str, test_case: models.TestCase) -> str:
    """Build Python script to execute function with test case arguments."""
    return textwrap.dedent(
        f"""
        import json

        args = json.loads('''{json.dumps(test_case.args)}''')
        kwargs = json.loads('''{json.dumps(test_case.kwargs)}''')

        result = {func_name}(*args, **kwargs)
        print(json.dumps(result))
        """
    )


def check_functional_correctness(
    code: PythonCode | str, func_name: str, test_cases: list[models.TestCase]
) -> int:
    """Execute test cases against code in sandboxed interpreter, return pass count."""
    if not test_cases:
        return 0

    source = code.code if isinstance(code, dspy.Code) else code

    def _run_test(interp: dspy.PythonInterpreter, test: models.TestCase) -> bool:
        try:
            actual_json = interp.execute(_build_execution_script(func_name, test))
            actual = json.loads(actual_json)
            expected = json.loads(json.dumps(test.expected_output))
            return actual == expected
        except Exception:
            return False

    with dspy.PythonInterpreter() as interp:
        interp.execute(source)
        return sum(_run_test(interp, test) for test in test_cases)
