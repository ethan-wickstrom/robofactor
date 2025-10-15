import json
from pathlib import Path

from returns.result import safe

from ..types import Json


@safe
def load_json(file_path: Path) -> Json:
    """Parse JSON file and return parsed data."""
    with file_path.open(encoding="utf-8") as f:
        return json.load(f)
