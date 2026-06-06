from typing import TypeIs

from returns.result import Failure, Success

from robofactor.data._internal.collectors import collect
from robofactor.data._internal.parsers import BasicParser
from robofactor.json_value import Json


def _is_int(value: Json) -> TypeIs[int]:
    return isinstance(value, int)


def test_collect_success_and_failure() -> None:
    int_parser = BasicParser[int](type_check=_is_int, type_name="int")

    ok = collect([1, 2, 3], int_parser)
    assert isinstance(ok, Success)
    assert ok.unwrap() == [1, 2, 3]

    bad = collect([1, "x", 3], int_parser)
    assert isinstance(bad, Failure)
    assert "element 1" in bad.failure()
