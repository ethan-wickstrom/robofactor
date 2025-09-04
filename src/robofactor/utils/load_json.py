import json
from pathlib import Path
from typing import Any

from returns.result import Result, safe


def load_json(file_path: Path) -> Result[list[dict[str, Any]], str]:
    """Parse JSON file into list of dictionaries."""
    return safe(lambda: json.load(file_path.open("r", encoding="utf-8")))().alt(str)
