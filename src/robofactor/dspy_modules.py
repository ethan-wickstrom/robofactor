"""
DSPy modules, Pydantic models, and data loading for the refactoring agent.
"""

import json
from pathlib import Path
from typing import List

import dspy
from pydantic import BaseModel, Field, field_validator

from . import analysis_utils
from .evaluation import CodeQualityScores, TestCase
from .functional_types import Err, Ok, Result


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
        return analysis_utils._extract_python_code(v)


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


def _evaluate_syntax(code: str) -> Result[str, str]:
    """Checks for valid Python syntax and returns the function name if valid."""
    is_valid, func_name, err = analysis_utils.check_syntax(code)
    if not is_valid or not func_name:
        return Err(f"Syntax Check Failed: {err or 'No function found.'}")
    return Ok(func_name)


def _evaluate_quality(code: str, func_name: str) -> Result[CodeQualityScores, str]:
    """Checks code quality and returns the scores if successful."""
    try:
        quality = analysis_utils.check_code_quality(code, func_name)
        return Ok(quality)
    except Exception as e:
        return Err(f"Quality Check Failed: {e}")


def _evaluate_functional_correctness(
    code: str, func_name: str, tests: List[TestCase]
) -> Result[float, str]:
    """Runs functional tests and returns the pass rate if successful."""
    if not tests:
        return Ok(1.0)  # No tests to run, so functionally perfect by default.

    try:
        passed_tests = analysis_utils.check_functional_correctness(code, func_name, tests)
        score = (passed_tests / len(tests)) if tests else 1.0
        return Ok(score)
    except Exception as e:
        return Err(f"Functional Check Failed: {e}")


class RefactoringEvaluator(dspy.Module):
    """
    A module to evaluate refactored code using a pipeline of programmatic checks
    and a final LLM judgment, with robust, functional error handling.
    """

    def __init__(self):
        super().__init__()
        self.evaluator = dspy.Predict(FinalEvaluation)

    def forward(
        self, original_example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> float:
        """
        Evaluates the refactored code. Returns 0.0 on any programmatic failure
        to signal the optimizer, otherwise returns the LLM's final score.
        """
        code = prediction.refactored_code
        test_cases_raw = original_example.get("test_cases", [])
        tests: List[TestCase] = test_cases_raw if isinstance(test_cases_raw, list) and test_cases_raw is not None else []

        # 1. Evaluate Syntax
        syntax_result = _evaluate_syntax(code)
        if isinstance(syntax_result, Err):
            return 0.0
        func_name = syntax_result.value

        # 2. Evaluate Code Quality
        quality_result = _evaluate_quality(code, func_name)
        if isinstance(quality_result, Err):
            return 0.0
        quality_scores = quality_result.value

        # 3. Evaluate Functional Correctness
        functional_result = _evaluate_functional_correctness(code, func_name, tests)
        if isinstance(functional_result, Err):
            return 0.0
        functional_score = functional_result.value

        # 4. Final LLM-based Evaluation (if all programmatic checks pass)
        try:
            eval_result = self.evaluator(
                code_snippet=code,
                quality_scores=quality_scores.model_dump_json(),
                functional_score=functional_score,
            )
            return eval_result.evaluation.final_score
        except Exception:
            return 0.0


# --- Training Data ---


def get_training_data() -> List[dspy.Example]:
    """
    Loads training examples from an external JSON file.
    Decouples training data from application logic for better maintainability.
    """
    data_path = Path(__file__).parent / "training_data.json"
    if not data_path.exists():
        return []

    with data_path.open("r", encoding="utf-8") as f:
        training_json = json.load(f)

    examples = []
    for item in training_json:
        test_cases = [TestCase(**tc) for tc in item.get("test_cases", [])]
        example = dspy.Example(
            code_snippet=item["code_snippet"],
            test_cases=test_cases,
        ).with_inputs("code_snippet")
        examples.append(example)

    return examples
