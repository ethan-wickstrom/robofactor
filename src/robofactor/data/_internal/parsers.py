from collections.abc import Callable
from typing import Any, Protocol

from returns.result import Failure, Result, Success


class Parser[T](Protocol):
    """Protocol for all parsers."""

    def parse(self, raw_data: Any) -> Result[T, str]:
        """Parse raw data into type T."""
        ...


class BasicParser[T]:
    """Basic parser for primitive types with validation."""

    def __init__(
        self,
        type_check: Callable[[Any], bool],
        type_name: str,
        validator: Callable[[T], Result[T, str]] | None = None,
    ):
        self.type_check = type_check
        self.type_name = type_name
        self.validator = validator or (lambda x: Success(x))

    def parse(self, raw_data: Any) -> Result[T, str]:
        """Parse and validate raw data."""
        if not self.type_check(raw_data):
            return Failure(f"Expected {self.type_name}, got {type(raw_data).__name__}")

        return self.validator(raw_data)


class ListParser[T]:
    """Parser for lists with element validation."""

    def __init__(self, element_parser: Parser[T]):
        self.element_parser = element_parser

    def parse(self, raw_data: Any) -> Result[list[T], str]:
        """Parse list with element validation."""
        if not isinstance(raw_data, list):
            return Failure(f"Expected list, got {type(raw_data).__name__}")

        parsed_elements = []
        for i, element in enumerate(raw_data):
            result = self.element_parser.parse(element)
            if isinstance(result, Failure):
                return Failure(f"Error parsing element {i}: {result.failure()}")
            parsed_elements.append(result.unwrap())

        return Success(parsed_elements)


class DictParser[T]:
    """Parser for dictionaries with field validation."""

    def __init__(self, field_parsers: dict[str, Parser[Any]], constructor: Callable[..., T]):
        self.field_parsers = field_parsers
        self.constructor = constructor

    def parse(self, raw_data: Any) -> Result[T, str]:
        """Parse dictionary into structured object."""
        if not isinstance(raw_data, dict):
            return Failure(f"Expected dict, got {type(raw_data).__name__}")

        parsed_fields = {}
        for field_name, parser in self.field_parsers.items():
            field_value = raw_data.get(field_name)
            if field_value is None:
                return Failure(f"Missing required field: {field_name}")

            result = parser.parse(field_value)
            if isinstance(result, Failure):
                return Failure(f"Error parsing field '{field_name}': {result.failure()}")

            parsed_fields[field_name] = result.unwrap()

        try:
            return Success(self.constructor(**parsed_fields))
        except Exception as e:
            return Failure(f"Failed to construct object: {e!s}")


class OptionalParser[T]:
    """Parser for optional fields with default values."""

    def __init__(self, inner_parser: Parser[T], default: T):
        self.inner_parser = inner_parser
        self.default = default

    def parse(self, raw_data: Any) -> Result[T, str]:
        """Parse with fallback to default value."""
        if raw_data is None:
            return Success(self.default)
        return self.inner_parser.parse(raw_data)


class TransformParser[T, U]:
    """Parser that transforms parsed data using a function."""

    def __init__(self, inner_parser: Parser[T], transform: Callable[[T], Result[U, str]]):
        self.inner_parser = inner_parser
        self.transform = transform

    def parse(self, raw_data: Any) -> Result[U, str]:
        """Parse and transform the result."""
        return self.inner_parser.parse(raw_data).bind(self.transform)


def parse[T](raw_data: Any, parser: Parser[T]) -> Result[T, str]:
    """Universal parse function that works with any parser."""
    return parser.parse(raw_data)


__all__ = [
    "BasicParser",
    "DictParser",
    "ListParser",
    "OptionalParser",
    "Parser",
    "TransformParser",
    "parse",
]
