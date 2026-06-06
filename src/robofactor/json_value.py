from __future__ import annotations

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]

__all__ = ["Json"]
