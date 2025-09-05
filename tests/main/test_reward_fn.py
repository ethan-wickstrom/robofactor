from types import SimpleNamespace

from returns.result import Failure

import robofactor.main as main_mod


def test_reward_fn_returns_zero_on_examples_failure(monkeypatch):
    monkeypatch.setattr(main_mod.examples, "get_examples", lambda: Failure("fail"))
    score = main_mod._reward_fn({"code_snippet": "print('x')"}, SimpleNamespace())
    assert score == 0.0

