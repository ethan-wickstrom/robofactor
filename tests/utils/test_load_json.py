from pathlib import Path

from returns.result import Failure, Success

from robofactor.utils.load_json import load_json


def test_load_json_success(tmp_path: Path):
    p = tmp_path / "ok.json"
    p.write_text('[{"a": 1}, {"b": 2}]', encoding="utf-8")
    res = load_json(p)
    assert isinstance(res, Success)
    assert res.unwrap() == [{"a": 1}, {"b": 2}]


def test_load_json_failure(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text('{"a": 1,}', encoding="utf-8")  # trailing comma -> invalid JSON
    res = load_json(p)
    assert isinstance(res, Failure)
    assert "JSON" in res.failure() or "Expecting" in res.failure()

