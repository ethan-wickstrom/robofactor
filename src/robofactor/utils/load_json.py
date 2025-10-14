import json
from pathlib import Path

from returns.result import Result, safe

from ..types import Json


def load_json(file_path: Path) -> Result[Json, str]:
    """Parse JSON file into list of dictionaries."""
    return safe(lambda: json.load(file_path.open("r", encoding="utf-8")))().alt(str)
