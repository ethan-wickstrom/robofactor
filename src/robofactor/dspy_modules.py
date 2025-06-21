"""
DSPy modules, Pydantic models, and data loading for the refactoring agent.
"""

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

import dspy
from pydantic import BaseModel, Field, field_validator
from returns.result import Failure, Result, Success

from . import analysis, evaluation
from .evaluation import EvaluationResult, TestCase

# --- Pydantic Models for Structured Outputs ---


class AnalysisOutput(BaseModel):
    """A structured analysis of a Python code snippet."""

    analysis: str = Field(
        description="A concise summary of the code's functionality, complexity, and dependencies."
    )
    refactoring_opportunities: List[str] = Field(
        description="A bulleted list of specific, actionable refactoring opportunities."
    )


class PlanOutput(BaseModel):
    """A step-by-step plan to refactor Python code."""

    refactoring_summary: str = Field(description="A high-level summary of the refactoring goal.")
    plan_steps: List[str] = Field(
        description="A detailed, step-by-step list of actions to refactor the code."
    )


class ImplementationOutput(BaseModel):
    """The final, refactored code and an explanation of the changes."""

    refactored_code: str = Field(
        description="The final, PEP8-compliant, refactored Python code block with type hints and docstrings.",
    )
    implementation_explanation: str = Field(
        description="A brief explanation of how the plan was implemented."
    )

    @field_validator("refactored_code")
    @classmethod
    def extract_from_markdown(cls, v: str) -> str:
        """Ensures the output is raw Python code, stripping markdown fences."""
        return analysis._extract_python_code(v)


class EvaluationOutput(BaseModel):
    """A final, holistic evaluation of the refactored code."""

    final_score: float = Field(
        description="A final, holistic score from 0.0 to 1.0, weighting functional correctness most heavily.",
        ge=0.0,
        le=1.0,
    )
    final_suggestion: str = Field(
        description="A final suggestion for improvement or a confirmation of readiness."
    )


# --- DSPy Signatures with Nested Pydantic Models ---


class CodeAnalysis(dspy.Signature):
    """Analyze Python code for its purpose, complexity, and areas for improvement."""

    code_snippet: str = dspy.InputField(desc="The Python code to be analyzed.")
    analysis: AnalysisOutput = dspy.OutputField()


class RefactoringPlan(dspy.Signature):
    """Create a step-by-step plan to refactor Python code based on an analysis."""

    code_snippet: str = dspy.InputField(desc="The original Python code snippet.")
    analysis: str = dspy.InputField(desc="The analysis of the code snippet.")
    plan: PlanOutput = dspy.OutputField()


class RefactoredCode(dspy.Signature):
    """Generate refactored Python code based on a plan."""

    original_code: str = dspy.InputField(desc="The original, un-refactored Python code.")
    refactoring_summary: str = dspy.InputField(desc="The high-level goal of the refactoring.")
    plan_steps: List[str] = dspy.InputField(desc="The step-by-step plan to apply.")
    implementation: ImplementationOutput = dspy.OutputField()


class FinalEvaluation(dspy.Signature):
    """Evaluate the refactored code based on quantitative scores and provide a final assessment."""

    code_snippet: str = dspy.InputField(desc="The refactored code being evaluated.")
    quality_scores: str = dspy.InputField(
        desc="A JSON object of quantitative scores (linting, complexity, typing, docstrings)."
    )
    functional_score: float = dspy.InputField(
        desc="A score from 0.0 to 1.0 indicating test pass rate."
    )
    evaluation: EvaluationOutput = dspy.OutputField()


# --- DSPy Modules ---


class CodeRefactor(dspy.Module):
    """A module that analyzes, plans, and refactors Python code."""

    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(CodeAnalysis)
        self.planner = dspy.Predict(RefactoringPlan)
        self.implementer = dspy.Predict(RefactoredCode)

    def forward(self, code_snippet: str) -> dspy.Prediction:
        """Orchestrates the analysis, planning, and implementation steps."""
        analysis_result = self.analyzer(code_snippet=code_snippet)
        plan_result = self.planner(
            code_snippet=code_snippet, analysis=analysis_result.analysis.analysis
        )
        impl_result = self.implementer(
            original_code=code_snippet,
            refactoring_summary=plan_result.plan.refactoring_summary,
            plan_steps=plan_result.plan.plan_steps,
        )

        return dspy.Prediction(
            analysis=analysis_result.analysis.analysis,
            refactoring_opportunities=analysis_result.analysis.refactoring_opportunities,
            refactoring_summary=plan_result.plan.refactoring_summary,
            plan_steps=plan_result.plan.plan_steps,
            refactored_code=impl_result.implementation.refactored_code,
            implementation_explanation=impl_result.implementation.implementation_explanation,
        )


class RefactoringEvaluator(dspy.Module):
    """
    A module to evaluate refactored code using a pipeline of programmatic checks
    and a final LLM judgment.
    """

    def __init__(self):
        super().__init__()
        self.evaluator = dspy.Predict(FinalEvaluation)

    def forward(
        self, original_example: dspy.Example, prediction: dspy.Prediction, trace: Optional[Any] = None
    ) -> float:
        """
        Evaluates the refactored code through a robust, multi-stage pipeline.

        Returns 0.0 on any failure to signal the optimizer, otherwise returns the LLM's score.
        """
        refactored_code = getattr(prediction, "refactored_code", None)
        if not refactored_code:
            logging.warning("Evaluation failed: No refactored code found.")
            return 0.0

        code_to_evaluate = analysis._extract_python_code(refactored_code)
        if not code_to_evaluate:
            logging.warning("Evaluation failed: Extracted code is empty.")
            return 0.0

        tests = getattr(original_example, "test_cases", [])

        programmatic_result: Result[EvaluationResult, str] = evaluation.evaluate_refactored_code(
            code_to_evaluate, tests
        )

        match programmatic_result:
            case Success(eval_data):
                quality_scores = eval_data.quality_scores
                functional_check = eval_data.functional_check
                functional_score = (
                    functional_check.passed_tests / functional_check.total_tests
                    if functional_check.total_tests > 0
                    else 1.0
                )

                try:
                    llm_evaluation = self.evaluator(
                        code_snippet=code_to_evaluate,
                        quality_scores=quality_scores.model_dump_json(),
                        functional_score=functional_score,
                    )
                    return llm_evaluation.evaluation.final_score
                except Exception as e:
                    logging.error(f"LLM-based evaluation failed: {e}", exc_info=True)
                    return 0.0

            case Failure(error_message):
                logging.warning(f"Programmatic evaluation failed: {error_message}")
                return 0.0


# --- Training Data ---


def get_training_data() -> List[dspy.Example]:
    """
    Loads training examples from an external JSON file.
    Decouples training data from application logic for better maintainability.
    """
    data_path = Path(__file__).parent / "training_data.json"
    try:
        with data_path.open("r", encoding="utf-8") as f:
            training_json = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Could not load or parse training data from {data_path}: {e}")
        return []

    examples = []
    for item in training_json:
        test_cases = [TestCase(**tc) for tc in item.get("test_cases", [])]
        example = dspy.Example(
            code_snippet=item["code_snippet"],
            test_cases=test_cases,
        ).with_inputs("code_snippet")
        examples.append(example)

    return examples
