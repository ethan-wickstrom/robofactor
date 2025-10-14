from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

from robofactor import config
from robofactor.types import LintDiagnostic


def run_ruff_json(target: Path) -> list[LintDiagnostic]:
    result = subprocess.run(
        ["ruff", "check", "--output-format", "json", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    return cast(list[LintDiagnostic], json.loads(result.stdout) if result.stdout else [])


def format_lint_issue(rec: LintDiagnostic) -> str:
    location = rec.get("location") or {}
    return (
        f"{rec.get('filename', '')}:{location.get('row', 0)}:{location.get('column', 0)} "
        f"{rec.get('code', '')} {rec.get('message', '')}"
    )


def is_complexity_issue(rec: LintDiagnostic) -> bool:
    return rec.get("code") == config.FLAKE8_COMPLEXITY_CODE
