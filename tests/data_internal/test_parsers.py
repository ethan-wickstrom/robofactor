from returns.result import Failure, Success

from robofactor.data._internal.parsers import BasicParser, DictParser, ListParser


def test_basic_parser_success_and_failure() -> None:
    is_int = BasicParser[int](type_check=lambda x: isinstance(x, int), type_name="int")
    ok = is_int.parse(3)
    err = is_int.parse("nope")

    assert isinstance(ok, Success)
    assert ok.unwrap() == 3

    assert isinstance(err, Failure)
    assert "Expected int" in err.failure()


def test_list_parser_success_and_element_error() -> None:
    is_int = BasicParser[int](type_check=lambda x: isinstance(x, int), type_name="int")
    list_parser = ListParser(is_int)

    ok = list_parser.parse([1, 2, 3])
    assert isinstance(ok, Success)
    assert ok.unwrap() == [1, 2, 3]

    bad = list_parser.parse([1, "x", 3])
    assert isinstance(bad, Failure)
    # Contains index and inner parser error
    msg = bad.failure()
    assert "element 1" in msg and "Expected int" in msg

    not_list = list_parser.parse({})
    assert isinstance(not_list, Failure)
    assert "Expected list" in not_list.failure()


def test_dict_parser_success_and_missing_field() -> None:
    a_parser = BasicParser[int](type_check=lambda x: isinstance(x, int), type_name="int")
    b_parser = BasicParser[str](type_check=lambda x: isinstance(x, str), type_name="string")

    parser = DictParser(
        field_parsers={"a": a_parser, "b": b_parser},
        constructor=lambda a, b: (a, b),
    )

    ok = parser.parse({"a": 7, "b": "hi"})
    assert isinstance(ok, Success)
    assert ok.unwrap() == (7, "hi")

    missing = parser.parse({"a": 7})
    assert isinstance(missing, Failure)
    assert "Missing required field: b" in missing.failure()

