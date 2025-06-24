"""
The evaluation package is responsible for assessing the quality and correctness
of refactored code. It includes a multi-stage pipeline that performs syntax,
quality, and functional checks to ensure that AI-generated code is safe,
reliable, and adheres to best practices.
"""
from . import checkers, pipeline
from .pipeline import evaluate_refactored_code, EvaluationResult

__all__ = ["evaluate_refactored_code", "EvaluationResult", "checkers", "pipeline"]
