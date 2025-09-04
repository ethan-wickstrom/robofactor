from returns.result import Failure, Success

from robofactor.data import examples


def test_get_examples_reads_training_file_success():
    res = examples.get_examples()
    assert isinstance(res, Success)
    exs = res.unwrap()
    assert isinstance(exs, list) and len(exs) > 0
    first = exs[0]
    # dspy.Example stores data like a mapping
    assert hasattr(first, "code_snippet")
    assert hasattr(first, "test_cases")


def test_get_examples_propagates_load_failure(monkeypatch):
    class FakeFailure:
        def bind(self, _):
            return Failure("boom")

    monkeypatch.setattr(examples, "load_json", lambda *_args, **_kwargs: FakeFailure())
    res = examples.get_examples()
    assert isinstance(res, Failure)
    assert res.failure() == "boom"

