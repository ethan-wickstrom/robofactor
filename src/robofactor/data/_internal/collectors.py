from returns.result import Result

from robofactor.data._internal.parsers import ListParser, Parser
from robofactor.json_value import Json


def collect[T](raw_items: list[Json], item_parser: Parser[T]) -> Result[list[T], str]:
    """Parse every raw item with the same parser."""
    return ListParser(item_parser).parse(raw_items)


__all__ = ["collect"]
