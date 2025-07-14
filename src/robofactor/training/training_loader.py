from __future__ import annotations

import logging

import dspy
from pydantic import ValidationError

from robofactor.app.config import TRAINING_DATA_FILE

from .models import TrainingEntry, TrainingSetAdapter

FAILURE_SCORE = 0.0
logger = logging.getLogger(__name__)


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
