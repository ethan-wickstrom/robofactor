"""
Defines functional Algebraic Data Types (ADTs) for robust and type-safe error handling.

This module introduces a generic `Result` type, which is a sum type representing either
a success (`Ok`) or a failure (`Err`). Using `Result` allows functions to make
potential failures an explicit part of their type signature, eliminating the need
for imperative try/except blocks and preventing hidden exceptions.

This approach aligns with functional programming principles by:
- Enforcing explicit handling of all possible outcomes at the type level.
- Improving code clarity and predictability.
- Making functions more composable and easier to reason about.

For more information on the concept, see related implementations in languages like
Rust (`std::result::Result`) or Scala (`Either`).
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, Union

# Type variables for the generic Result type
# T represents the type of the success value.
# E represents the type of the error value.
T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class CliResult:
    """Represents the successful result of a CLI command."""

    stdout: str
    stderr: str
    exit_code: int


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Represents a successful outcome containing a value."""

    value: T

    def raise_for_status(self) -> None:
        """Does nothing for a successful result."""
        pass


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Represents a failure outcome containing an error."""

    error: E

    def raise_for_status(self) -> None:
        """Raises a RuntimeError for a failure result."""
        raise RuntimeError(str(self.error))


# The Result type is a union of Ok and Err, representing either success or failure.
Result = Union[Ok[T], Err[E]]