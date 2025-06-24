from collections.abc import Sequence
from typing import TypeGuard

from .types import JSON


def is_json_list(x: JSON) -> TypeGuard[Sequence[JSON]]:
    return isinstance(x, Sequence)
