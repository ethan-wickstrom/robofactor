import json
import logging

import dspy
from returns.result import Failure, Success

from robofactor import analysis
from robofactor.data import examples
from robofactor.signatures.code_analysis import CodeAnalysis
from robofactor.signatures.final_evaluation import FinalEvaluation
from robofactor.signatures.refactored_code import RefactoredCode
from robofactor.signatures.refactoring_plan import RefactoringPlan

FAILURE_SCORE = 0.0
TRAINING_DATA_FILE = "training.json"
logger = logging.getLogger(__name__)


class CodeRefactor(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.analyzer = dspy.Predict(CodeAnalysis)
        self.planner = dspy.Predict(RefactoringPlan)
        self.implementer = dspy.Predict(RefactoredCode)
        self.evaluator = dspy.Predict(FinalEvaluation)

    @staticmethod
    def _quality_scores_json(code: str) -> str:
        """Compute quality metrics and return a JSON string.

        Falls back to an empty JSON object on failure to keep the evaluator robust.
        """
        try:
            is_valid, func_name, _ = analysis.check_syntax(code)
            # Use provided function name when available; quality can still run without it.
            quality = analysis.check_code_quality(code, func_name if is_valid else None)

            # Aggregate quality as a simple average of available scalar scores.
            scalars = [
                quality.linting_score,
                quality.complexity_score,
                quality.typing_score,
                quality.docstring_score,
            ]
            aggregate = sum(scalars) / len(scalars) if scalars else 0.0

            payload = {
                "linting_score": quality.linting_score,
                "complexity_score": quality.complexity_score,
                "typing_score": quality.typing_score,
                "docstring_score": quality.docstring_score,
                "aggregate_quality": aggregate,
                "linting_issues": quality.linting_issues,
            }
            return json.dumps(payload, ensure_ascii=False)
        except Exception as exc:  # Keep model resilient during optimization runs
            logger.debug("quality scoring failed: %s", exc)
            return "{}"

    @staticmethod
    def _functional_score(original_code: str, refactored_code: str) -> float:
        """Compute functional score based on training-set tests.

        - Looks up tests by matching the original code snippet in the training set.
        - Returns pass_rate = passed/total when tests exist.
        - Returns 1.0 when the example has no tests.
        - Returns 0.0 on failure (no match, invalid syntax, or unexpected error).
        """
        try:
            trainset_result = examples.get_examples()
            match trainset_result:
                case Success(trainset):
                    example = next((ex for ex in trainset if ex.code_snippet == original_code), None)
                    if example is None:
                        return 0.0
                    tests = getattr(example, "test_cases", []) or []
                    if not tests:
                        return 1.0

                    is_valid, func_name, _ = analysis.check_syntax(refactored_code)
                    if not is_valid or not func_name:
                        return 0.0
                    passed = analysis.check_functional_correctness(refactored_code, func_name, tests)
                    return passed / len(tests)
                case Failure(_):
                    return 0.0
        except Exception as exc:  # Defensive: never break the module flow
            logger.debug("functional scoring failed: %s", exc)
            return 0.0
        return 0.0

    def forward(self, code_snippet: str) -> dspy.Prediction:
        analysis_result = self.analyzer(code_snippet=code_snippet)
        plan_result = self.planner(
            code_snippet=code_snippet,
            analysis=analysis_result.analysis,
        )
        impl_result = self.implementer(
            original_code=code_snippet,
            refactoring_summary=plan_result.refactoring_summary,
            plan_steps=plan_result.plan_steps,
        )
        eval_result = self.evaluator(
            code_snippet=impl_result.refactored_code,
            quality_scores=self._quality_scores_json(impl_result.refactored_code),
            functional_score=self._functional_score(
                original_code=code_snippet, refactored_code=impl_result.refactored_code
            ),
        )
        return dspy.Prediction(
            analysis=analysis_result.analysis,
            refactoring_opportunities=analysis_result.refactoring_opportunities,
            refactoring_summary=plan_result.refactoring_summary,
            plan_steps=plan_result.plan_steps,
            refactored_code=impl_result.refactored_code,
            implementation_explanation=impl_result.implementation_explanation,
            final_score=eval_result.final_score,
            final_suggestion=eval_result.final_suggestion,
        )
