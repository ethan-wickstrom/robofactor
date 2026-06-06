from returns.result import Failure, Success

from robofactor.checks import BehaviorTest
from robofactor.data import examples


def test_get_examples_reads_training_file_success():
    res = examples.get_examples()
    assert isinstance(res, Success)
    exs = res.unwrap()
    assert isinstance(exs, list) and len(exs) > 0
    first = exs[0]
    assert hasattr(first, "code_snippet")
    assert hasattr(first, "behavior_tests")
    assert set(first.inputs().keys()) == {"code_snippet", "behavior_tests"}
    assert isinstance(first.behavior_tests[0], BehaviorTest)
    assert first.behavior_tests[0].case_id == "training-0"


def test_get_examples_propagates_load_failure(monkeypatch):
    class FakeFailure:
        def bind(self, _):
            return Failure("boom")

    monkeypatch.setattr(examples, "load_json", lambda *_args, **_kwargs: FakeFailure())
    res = examples.get_examples()
    assert isinstance(res, Failure)
    assert res.failure() == "boom"
