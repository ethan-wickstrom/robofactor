from returns.result import Failure, Success

from robofactor.data._internal.collectors import (
    collect,
    collect_partial,
    collect_with_context,
)
from robofactor.data._internal.parsers import BasicParser


def test_collect_success_and_failure() -> None:
    int_parser = BasicParser[int](type_check=lambda x: isinstance(x, int), type_name="int")

    ok = collect([1, 2, 3], int_parser)
    assert isinstance(ok, Success)
    assert ok.unwrap() == [1, 2, 3]

    bad = collect([1, "x", 3], int_parser)
    assert isinstance(bad, Failure)
    assert "element 1" in bad.failure()


def test_collect_with_context_enhances_error_message() -> None:
    int_parser = BasicParser[int](type_check=lambda x: isinstance(x, int), type_name="int")
    res = collect_with_context([1, "x"], int_parser, error_context="example")
    assert isinstance(res, Failure)
    assert "example 1" in res.failure()


def test_collect_partial_returns_both_success_and_errors() -> None:
    int_parser = BasicParser[int](type_check=lambda x: isinstance(x, int), type_name="int")
    successes, errors = collect_partial([1, "x", 3, "y"], int_parser)

    assert successes == [1, 3]
    assert len(errors) == 2
    assert errors[0].startswith("Item 1:") and errors[1].startswith("Item 3:")

