from pathlib import Path

import dspy
from returns.result import Result

from ..utils import load_json
from ._internal.collectors import collect
from ._internal.parsers import BasicParser, DictParser, ListParser
from .models import TestCase

_TRAINING_DATA_FILE = Path(__file__).parent / "training.json"


def _create_dspy_example_parser() -> DictParser[dspy.Example]:
    """Create a parser for converting raw JSON data to dspy.Example objects."""
    return DictParser(
        field_parsers={
            "code_snippet": BasicParser(
                type_check=lambda x: isinstance(x, str), type_name="string"
            ),
            "test_cases": ListParser(
                DictParser(
                    field_parsers={
                        "args": BasicParser(
                            type_check=lambda x: isinstance(x, list), type_name="list"
                        ),
                        "kwargs": BasicParser(
                            type_check=lambda x: isinstance(x, dict), type_name="dict"
                        ),
                        "expected_output": BasicParser(type_check=lambda x: True, type_name="any"),
                    },
                    constructor=TestCase,
                )
            ),
        },
        constructor=lambda code_snippet, test_cases: dspy.Example(
            code_snippet=code_snippet, test_cases=test_cases
        ).with_inputs("code_snippet", "test_cases"),
    )


def get_examples() -> Result[list[dspy.Example], str]:
    """Load and parse training examples from JSON file."""
    return load_json(_TRAINING_DATA_FILE).bind(
        lambda data: collect(data, _create_dspy_example_parser())
    )


__all__ = ["get_examples"]
