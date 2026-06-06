from __future__ import annotations

import json
import os
import textwrap
from typing import assert_never, cast

import dspy
from hypothesis import HealthCheck, Phase, find, settings
from hypothesis import strategies as st
from hypothesis.errors import NoSuchExample
from hypothesis.strategies import SearchStrategy

from robofactor.checks.model import (
    BehaviorTest,
    CallOutcome,
    CheckFailure,
    CheckName,
    Raised,
    Returned,
)
from robofactor.json_value import Json

GENERATED_COMPARISON_SETTINGS = settings(
    max_examples=50,
    derandomize=True,
    database=None,
    deadline=None,
    phases=(Phase.generate, Phase.shrink),
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.filter_too_much),
)


def explicit_behavior_failures(
    code: str,
    function_name: str,
    tests: tuple[BehaviorTest, ...],
) -> tuple[CheckFailure, ...]:
    if not tests:
        return ()

    with _python_interpreter() as interpreter:
        interpreter.execute(code)
        return tuple(
            failure
            for test in tests
            if (
                failure := _explicit_behavior_failure(
                    interpreter,
                    function_name,
                    test,
                )
            )
        )


def comparison_failures(
    source_code: str,
    refactored_code: str,
    function_name: str,
    tests: tuple[BehaviorTest, ...],
    *,
    check: CheckName,
) -> tuple[CheckFailure, ...]:
    if not tests:
        return ()

    with (
        _python_interpreter() as source_interpreter,
        _python_interpreter() as refactored_interpreter,
    ):
        source_interpreter.execute(source_code)
        refactored_interpreter.execute(refactored_code)
        return tuple(
            failure
            for test in tests
            if (
                failure := _comparison_failure(
                    source_interpreter,
                    refactored_interpreter,
                    function_name,
                    test,
                    check=check,
                )
            )
        )


def generated_comparison_failures(
    source_code: str,
    refactored_code: str,
    function_name: str,
    tests: tuple[BehaviorTest, ...],
) -> tuple[CheckFailure, ...]:
    if not tests:
        return ()

    with (
        _python_interpreter() as source_interpreter,
        _python_interpreter() as refactored_interpreter,
    ):
        source_interpreter.execute(source_code)
        refactored_interpreter.execute(refactored_code)
        try:
            failing_test = find(
                _generated_test_strategy(tests),
                lambda test: (
                    _comparison_failure(
                        source_interpreter,
                        refactored_interpreter,
                        function_name,
                        test,
                        check="generated_comparison",
                    )
                    is not None
                ),
                settings=GENERATED_COMPARISON_SETTINGS,
            )
        except NoSuchExample:
            return ()

        failure = _comparison_failure(
            source_interpreter,
            refactored_interpreter,
            function_name,
            failing_test,
            check="generated_comparison",
        )
        assert failure is not None
        return (failure,)


def _explicit_behavior_failure(
    interpreter: dspy.PythonInterpreter,
    function_name: str,
    test: BehaviorTest,
) -> CheckFailure | None:
    actual = _call_function(interpreter, function_name, test)
    match actual:
        case Returned(value=value) if value == test.expected_output:
            return None
        case Returned(value=value):
            return CheckFailure(
                check="explicit_behavior",
                case_id=test.case_id,
                message=f"expected {test.expected_output!r}, got {value!r}",
                refactored=actual,
            )
        case Raised(reason=reason):
            return CheckFailure(
                check="explicit_behavior",
                case_id=test.case_id,
                message=reason,
                refactored=actual,
            )
        case _:
            assert_never(actual)


def _comparison_failure(
    source_interpreter: dspy.PythonInterpreter,
    refactored_interpreter: dspy.PythonInterpreter,
    function_name: str,
    test: BehaviorTest,
    *,
    check: CheckName,
) -> CheckFailure | None:
    source = _call_function(source_interpreter, function_name, test)
    refactored = _call_function(refactored_interpreter, function_name, test)
    if source == refactored:
        return None

    return CheckFailure(
        check=check,
        case_id=test.case_id,
        message="source and refactored outputs differ",
        source=source,
        refactored=refactored,
    )


def _call_function(
    interpreter: dspy.PythonInterpreter,
    function_name: str,
    test: BehaviorTest,
) -> CallOutcome:
    try:
        output = interpreter.execute(_call_script(function_name, test))
        return Returned(value=cast("Json", json.loads(output)))
    except json.JSONDecodeError as error:
        return Raised(reason=f"could not decode JSON output: {error}")
    except Exception as error:
        # Sandbox boundary: user/model code exceptions are behavior outcomes.
        return Raised(reason=str(error))


def _call_script(function_name: str, test: BehaviorTest) -> str:
    args = json.dumps(test.args)
    kwargs = json.dumps(test.kwargs)
    return textwrap.dedent(
        f"""
        import json

        args = json.loads({args!r})
        kwargs = json.loads({kwargs!r})
        result = {function_name}(*args, **kwargs)
        print(json.dumps(result))
        """
    )


def _python_interpreter() -> dspy.PythonInterpreter:
    os.environ.setdefault("DENO_NO_PACKAGE_JSON", "1")
    return dspy.PythonInterpreter()


def _generated_test_strategy(tests: tuple[BehaviorTest, ...]) -> SearchStrategy[BehaviorTest]:
    return st.one_of(*tuple(_test_strategy(test) for test in tests))


def _test_strategy(test: BehaviorTest) -> SearchStrategy[BehaviorTest]:
    args = st.tuples(*tuple(_json_strategy(value) for value in test.args))
    kwargs = st.fixed_dictionaries(
        {name: _json_strategy(value) for name, value in test.kwargs.items()}
    )
    return st.builds(
        BehaviorTest,
        case_id=st.just(f"{test.case_id}.generated"),
        args=args,
        kwargs=kwargs,
        expected_output=st.just(test.expected_output),
        labels=st.just((*test.labels, "generated")),
    )


def _json_strategy(value: Json) -> SearchStrategy[Json]:
    match value:
        case None:
            return st.none()
        case bool():
            return st.booleans()
        case int():
            return st.integers(min_value=value - 10, max_value=value + 10)
        case float():
            return st.floats(
                min_value=value - 100.0,
                max_value=value + 100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        case str():
            return st.one_of(
                st.just(value),
                st.text(min_size=0, max_size=max(8, min(len(value) + 4, 24))),
            )
        case list():
            if not value:
                return st.just([])
            return st.lists(
                st.one_of(*tuple(_json_strategy(item) for item in value)),
                min_size=0,
                max_size=min(len(value) + 2, 5),
            )
        case dict():
            return st.fixed_dictionaries({key: _json_strategy(item) for key, item in value.items()})
        case _:
            assert_never(value)
