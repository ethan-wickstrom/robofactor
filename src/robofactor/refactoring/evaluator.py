"""
Defines the DSPy module for evaluating refactored code.
"""

import logging

import dspy
from returns.result import Result, Success

from ..evaluation import EvaluationResult, evaluate_refactored_code
from ..parsing import analysis
from .signatures import FinalEvaluation

# --- Constants ---
FAILURE_SCORE = 0.0
logger = logging.getLogger(__name__)


class RefactoringEvaluator(dspy.Module):
    """Evaluates refactored code through automated checks and LLM assessment."""

    def __init__(self) -> None:
        """Initializes the evaluator module."""
        super().__init__()
        self.evaluator: dspy.Module = dspy.Predict(FinalEvaluation)

    def _handle_evaluation_success(
        self, eval_data: EvaluationResult, refactored_code: str
    ) -> float:
        """
        Process a successful programmatic evaluation by sending results to an LLM.

        Args:
            eval_data: The structured results from the programmatic checks.
            refactored_code: The code that was evaluated.

        Returns:
            The final score from the LLM assessment, or a failure score.
        """
        functional_score = (
            eval_data.functional_check.passed_tests / eval_data.functional_check.total_tests
            if eval_data.functional_check.total_tests > 0
            else 1.0
        )

        try:
            llm_evaluation = self.evaluator(
                code_snippet=refactored_code,
                quality_scores=eval_data.quality_scores.model_dump_json(),
                functional_score=functional_score,
            )
            return llm_evaluation.evaluation.final_score
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}", exc_info=True)
            return FAILURE_SCORE

    def forward(self, original_example: dspy.Example, prediction: dspy.Prediction) -> float:
        """
        Executes the full evaluation pipeline for a refactoring prediction.

        This function serves as the metric for the DSPy teleprompter. It first
        runs programmatic checks and then uses an LLM for a final assessment.

        Args:
            original_example: The original data point, containing test cases.
            prediction: The output from the CodeRefactor module.

        Returns:
            A final score between 0.0 and 1.0.
        """
        refactored_code = getattr(prediction, "refactored_code", "")
        if not refactored_code:
            logger.warning("Evaluation aborted: Missing refactored code")
            return FAILURE_SCORE

        code_to_evaluate = analysis.extract_python_code(refactored_code)
        if not code_to_evaluate:
            logger.warning("Evaluation aborted: Empty code extraction")
            return FAILURE_SCORE

        test_cases = getattr(original_example, "test_cases", [])
        eval_result: Result[EvaluationResult, str] = evaluate_refactored_code(
            code_to_evaluate, test_cases
        )

        if isinstance(eval_result, Success):
            return self._handle_evaluation_success(eval_result.unwrap(), code_to_evaluate)

        logger.warning(f"Programmatic evaluation failed: {eval_result.failure()}")
        return FAILURE_SCORE
