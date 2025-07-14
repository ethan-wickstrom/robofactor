"""
Single source of truth for JSON validation utilities.
"""

from collections.abc import Sequence
from typing import TypeGuard

from ..json.types import JSON, JSONObject


def is_json_object(x: JSON) -> TypeGuard[JSONObject]:
    """Type guard for JSON object validation."""
    return isinstance(x, dict)


def is_json_list(x: JSON) -> TypeGuard[Sequence[JSON]]:
    """Type guard for JSON list validation."""
    return isinstance(x, Sequence) and not isinstance(x, str)


def is_training_item(x: JSON) -> TypeGuard[JSONObject]:
    """
    Type guard for training data validation.

    Validates that the input is a JSON object with required training data structure.
    """
    return (
        is_json_object(x)
        and "code_snippet" in x
        and isinstance(x["code_snippet"], str)
        and (
            "test_cases" not in x
            or (
                isinstance(x["test_cases"], Sequence)
                and all(is_json_object(tc) for tc in x["test_cases"])
            )
        )
    )
