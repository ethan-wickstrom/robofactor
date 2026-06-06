from pathlib import Path
from typing import TypeIs

import dspy
from returns.result import Failure, Result

from ..checks import BehaviorTest
from ..json_value import Json
from ..utils import load_json
from ._internal.collectors import collect
from ._internal.parsers import BasicParser, DictParser, ListParser

_TRAINING_DATA_FILE = Path(__file__).parent / "training.json"
type BehaviorTestFields = tuple[list[Json], dict[str, Json], Json]


def _is_json(_value: Json) -> TypeIs[Json]:
    return True


def _is_json_dict(value: Json) -> TypeIs[dict[str, Json]]:
    return isinstance(value, dict)


def _is_json_list(value: Json) -> TypeIs[list[Json]]:
    return isinstance(value, list)


def _is_string(value: Json) -> TypeIs[str]:
    return isinstance(value, str)


def _collect_examples(data: Json) -> Result[list[dspy.Example], str]:
    if not isinstance(data, list):
        return Failure("Expected training data to be a list.")
    return collect(data, _create_dspy_example_parser())


def _create_dspy_example_parser() -> DictParser[dspy.Example]:
    """Create a parser for converting raw JSON data to dspy.Example objects."""
    return DictParser(
        field_parsers={
            "code_snippet": BasicParser(type_check=_is_string, type_name="string"),
            "behavior_tests": ListParser(_create_behavior_test_parser()),
        },
        constructor=lambda code_snippet, behavior_tests: dspy.Example(
            code_snippet=code_snippet,
            behavior_tests=_behavior_tests_from_fields(behavior_tests),
        ).with_inputs("code_snippet", "behavior_tests"),
    )


def _create_behavior_test_parser() -> DictParser[BehaviorTestFields]:
    return DictParser(
        field_parsers={
            "args": BasicParser(type_check=_is_json_list, type_name="list"),
            "kwargs": BasicParser(type_check=_is_json_dict, type_name="dict"),
            "expected_output": BasicParser(type_check=_is_json, type_name="json"),
        },
        constructor=lambda args, kwargs, expected_output: (args, kwargs, expected_output),
    )


def _behavior_tests_from_fields(
    behavior_test_fields: list[BehaviorTestFields],
) -> tuple[BehaviorTest, ...]:
    return tuple(
        BehaviorTest(
            case_id=f"training-{index}",
            args=tuple(args),
            kwargs=dict(kwargs),
            expected_output=expected_output,
        )
        for index, (args, kwargs, expected_output) in enumerate(behavior_test_fields)
    )


def get_examples() -> Result[list[dspy.Example], str]:
    """Load and parse training examples from JSON file."""
    return load_json(_TRAINING_DATA_FILE).bind(_collect_examples)


__all__ = ["get_examples"]
