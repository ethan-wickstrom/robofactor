import json
from pathlib import Path

from returns.result import Failure, Result, Success

from ..json_value import Json


def load_json(file_path: Path) -> Result[Json, str]:
    try:
        with file_path.open(encoding="utf-8") as f:
            return Success(json.load(f))
    except (OSError, json.JSONDecodeError) as error:
        return Failure(str(error))
