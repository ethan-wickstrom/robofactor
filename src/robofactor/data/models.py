from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class TestCase:
    """Represents an executable test case for a function."""

    args: list[Any]
    kwargs: dict[str, Any]
    expected_output: Any


__all__ = ["TestCase"]
