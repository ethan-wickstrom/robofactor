"""Pydantic models for the training data JSON format.

These models ensure type-safe deserialization of the training dataset and
eliminate the reliance on ``json.loads`` returning ``Any``. They also provide
single-source-of-truth schemas that other parts of the codebase can reference,
keeping the system DRY and explicit.
"""

from collections.abc import Sequence

from pydantic import BaseModel, Field, TypeAdapter

from ..parsing.models import TestCase


class TrainingEntry(BaseModel):
    """A single training data record.

    Attributes
    ----------
    code_snippet
        The Python source code which DSPy will analyse and refactor.
    test_cases
        A possibly empty sequence of :class:`robofactor.parsing.models.TestCase`
        instances that validate the behaviour of ``code_snippet``.
    """

    code_snippet: str = Field(..., min_length=1)
    test_cases: Sequence[TestCase] = Field(default_factory=tuple)


#: A pre-configured adapter that can validate a *top-level* JSON array of
#: :class:`TrainingEntry` objects.  We declare the adapter at module import
#: so that the heavy schema compilation happens once and can be reused by
#: every call to :pyfunc:`robofactor.training.training_loader.load_training_data`.
TrainingSetAdapter: TypeAdapter[list[TrainingEntry]] = TypeAdapter(list[TrainingEntry])
