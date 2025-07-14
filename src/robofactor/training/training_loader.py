from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TypeGuard

import dspy
from pydantic import ValidationError

from robofactor.app.config import TRAINING_DATA_FILE
from robofactor.json.types import JSON, JSONObject

from .models import TrainingEntry, TrainingSetAdapter

FAILURE_SCORE = 0.0
logger = logging.getLogger(__name__)


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


def is_json_object(x: JSON) -> TypeGuard[JSONObject]:
    return isinstance(x, dict)


def load_training_data() -> list[dspy.Example]:
    """Return the validated training set as a list of DSPy ``Example`` objects."""

    try:
        raw_text: str = TRAINING_DATA_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("Training data file not found: %s", TRAINING_DATA_FILE)
        return []

    try:
        entries: list[TrainingEntry] = TrainingSetAdapter.validate_json(raw_text)
    except ValidationError as exc:
        logger.error("Invalid JSON in training data: %s", exc)
        return []

    return [
        dspy.Example(
            code_snippet=entry.code_snippet,
            test_cases=list(entry.test_cases),  # Already validated TestCase models
        ).with_inputs("code_snippet")
        for entry in entries
    ]
