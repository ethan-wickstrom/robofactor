import json
from collections.abc import Sequence
from logging import getLogger
from pathlib import Path
from typing import TypeGuard, cast

import dspy

from ..app.config import TRAINING_DATA_FILE
from ..parsing.models import TestCase
from ..json.is_json_list import is_json_list
from ..json.types import JSON, JSONObject

FAILURE_SCORE = 0.0
logger = getLogger(__name__)


def is_training_item(x: JSON) -> TypeGuard[JSONObject]:
    return (
        isinstance(x, dict)
        and "code_snippet" in x
        and isinstance(x["code_snippet"], str)
        and (
            "test_cases" not in x
            or (
                isinstance(x["test_cases"], Sequence)
                and all(isinstance(tc, dict) for tc in x["test_cases"])
            )
        )
    )


def load_training_data() -> list[dspy.Example]:
    data_path = Path(__file__).parent / TRAINING_DATA_FILE
    try:
        # CAST the untyped json.loads → JSON
        raw = cast(JSON, json.loads(data_path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        logger.error(f"Training data file not found: {data_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in training data: {e}")
        return []

    # NARROW to actual Sequence[JSON]
    if not is_json_list(raw):
        logger.error(f"Expected top-level array, got {type(raw).__name__}")
        return []

    items: list[dspy.Example] = []
    for idx, entry in enumerate(raw):
        if not is_training_item(entry):
            logger.error(f"Invalid training entry at index {idx}: {entry!r}")
            continue
        code = entry["code_snippet"]
        raw_tcs = entry.get("test_cases", [])
        tcs = cast(Sequence[JSONObject], raw_tcs)
        items.append(
            dspy.Example(
                code_snippet=code,
                test_cases=[TestCase(**tc) for tc in tcs],
            ).with_inputs("code_snippet")
        )
    return items
