from collections.abc import Callable, Mapping
from typing import Protocol, TypeIs

from returns.result import Failure, Result, Success

from robofactor.json_value import Json


def _valid_value[T](value: T) -> Result[T, str]:
    return Success(value)


class Parser[T](Protocol):
    """Parser boundary for JSON input."""

    def parse(self, raw_data: Json) -> Result[T, str]:
        """Return typed data or a parse failure reason."""
        ...


class BasicParser[T: Json]:
    """Validate a scalar value with an optional domain validator."""

    def __init__(
        self,
        type_check: Callable[[Json], TypeIs[T]],
        type_name: str,
        validator: Callable[[T], Result[T, str]] | None = None,
    ):
        self.type_check = type_check
        self.type_name = type_name
        self.validator = validator or _valid_value

    def parse(self, raw_data: Json) -> Result[T, str]:
        """Return the scalar value after type and domain validation."""
        if not self.type_check(raw_data):
            return Failure(f"Expected {self.type_name}, got {type(raw_data).__name__}")

        return self.validator(raw_data)


class ListParser[T]:
    """Parse a list by applying one parser to each element."""

    def __init__(self, element_parser: Parser[T]):
        self.element_parser = element_parser

    def parse(self, raw_data: Json) -> Result[list[T], str]:
        """Return parsed elements or the first element failure."""
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
    """Parse a mapping into a structured object."""

    def __init__(self, field_parsers: Mapping[str, Parser[object]], constructor: Callable[..., T]):
        self.field_parsers = field_parsers
        self.constructor = constructor

    def parse(self, raw_data: Json) -> Result[T, str]:
        """Return the constructed object or the first field failure."""
        if not isinstance(raw_data, dict):
            return Failure(f"Expected dict, got {type(raw_data).__name__}")

        parsed_fields: dict[str, object] = {}
        for field_name, parser in self.field_parsers.items():
            if field_name not in raw_data:
                return Failure(f"Missing required field: {field_name}")

            field_value = raw_data[field_name]
            result = parser.parse(field_value)
            if isinstance(result, Failure):
                return Failure(f"Error parsing field '{field_name}': {result.failure()}")

            parsed_fields[field_name] = result.unwrap()

        try:
            return Success(self.constructor(**parsed_fields))
        except (TypeError, ValueError) as error:
            return Failure(f"Failed to construct object: {error!s}")


__all__ = [
    "BasicParser",
    "DictParser",
    "ListParser",
    "Parser",
]
