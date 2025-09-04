from pathlib import Path

import dspy
from returns.result import Result

from ..utils import load_json
from ._internal.collectors import collect_examples

_TRAINING_DATA_FILE = "training.json"


def get_examples() -> Result[list[dspy.Example], str]:
    return load_json(Path(_TRAINING_DATA_FILE)).bind(collect_examples)


__all__ = ["get_examples"]
