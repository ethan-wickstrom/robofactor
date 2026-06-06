import dspy
import pytest
from returns.result import Failure

import robofactor.main as main_mod


def test_reward_fn_fails_when_examples_cannot_load(monkeypatch) -> None:
    monkeypatch.setattr(main_mod.examples, "get_examples", lambda: Failure("fail"))

    with pytest.raises(ValueError, match="Training examples unavailable"):
        main_mod._reward_fn({"code_snippet": "print('x')"}, dspy.Prediction())
