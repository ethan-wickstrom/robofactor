from typing import Any

from returns.result import Failure, Result, Success

from robofactor.data._internal.parsers import ListParser, Parser, parse


def collect[T](raw_items: list[Any], item_parser: Parser[T]) -> Result[list[T], str]:
    """Generic collection function that works with any parser.

    This is more composable than specific collection functions as it can
    work with any parser type, not just dspy.Example parsers.
    """
    return parse(raw_items, ListParser(item_parser))


def collect_with_context[T](
    raw_items: list[Any], item_parser: Parser[T], error_context: str = "item"
) -> Result[list[T], str]:
    """Collect with enhanced error context for better debugging."""
    for i, raw_item in enumerate(raw_items):
        if isinstance(result := parse(raw_item, item_parser), Failure):
            return Failure(f"Error parsing {error_context} {i}: {result.failure()}")
    return parse(raw_items, ListParser(item_parser))


def collect_partial[T](raw_items: list[Any], item_parser: Parser[T]) -> tuple[list[T], list[str]]:
    """Collect items, returning both successes and failures.

    Unlike the other collect functions, this doesn't fail fast but
    processes all items and returns both successful parses and error messages.
    """
    successes = []
    errors = []
    for i, raw_item in enumerate(raw_items):
        match parse(raw_item, item_parser):
            case Success(value):
                successes.append(value)
            case Failure(error):
                errors.append(f"Item {i}: {error}")
    return (successes, errors)


__all__ = ["collect", "collect_partial", "collect_with_context"]
