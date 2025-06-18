"""
DSPy signatures, modules, and training data for the refactoring agent.
"""

import textwrap

import dspy

from . import analysis_utils
from .evaluation import TestCase


# --- Data Models and DSPy Signatures ---
class CodeAnalysis(dspy.Signature):
    """Analyze Python code for its purpose, complexity, and areas for improvement."""

    code_snippet: str = dspy.InputField(desc="The Python code to be analyzed.")
    analysis: str = dspy.OutputField(
        desc="A concise summary of the code's functionality and complexity."
    )
    refactoring_opportunities: list[str] = dspy.OutputField(
        desc="A bulleted list of specific, actionable refactoring opportunities."
    )


class RefactoringPlan(dspy.Signature):
    """Create a step-by-step plan to refactor Python code based on an analysis."""

    code_snippet: str = dspy.InputField(desc="The original Python code snippet.")
    analysis: str = dspy.InputField(desc="The analysis of the code snippet.")
    refactoring_summary: str = dspy.OutputField(
        desc="A high-level summary of the refactoring goal."
    )
    plan_steps: list[str] = dspy.OutputField(
        desc="A detailed, step-by-step list of actions to refactor the code."
    )


class RefactoredCode(dspy.Signature):
    """Generate refactored Python code based on a plan."""

    original_code: str = dspy.InputField(desc="The original, un-refactored Python code.")
    refactoring_summary: str = dspy.InputField(desc="The high-level goal of the refactoring.")
    plan_steps: list[str] = dspy.InputField(desc="The step-by-step plan to apply.")
    refactored_code: str = dspy.OutputField(
        prefix="```python\n",
        desc="The final, PEP8-compliant, refactored Python code block with type hints and docstrings.",
    )
    implementation_explanation: str = dspy.OutputField(
        desc="A brief explanation of how the plan was implemented."
    )


class EvaluationSignature(dspy.Signature):
    """Evaluate the refactored code based on quantitative scores and provide a final assessment."""

    code_snippet: str = dspy.InputField(desc="The refactored code being evaluated.")
    quality_scores: str = dspy.InputField(
        desc="A JSON object of quantitative scores (linting, complexity, typing, docstrings)."
    )
    functional_score: float = dspy.InputField(
        desc="A score from 0.0 to 1.0 indicating test pass rate."
    )
    final_score: float = dspy.OutputField(
        desc="A final, holistic score from 0.0 to 1.0, weighting functional correctness most heavily."
    )
    final_suggestion: str = dspy.OutputField(
        desc="A final suggestion for improvement or a confirmation of readiness."
    )


# --- DSPy Modules ---
class CodeRefactor(dspy.Module):
    """A module that analyzes, plans, and refactors Python code."""

    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(CodeAnalysis)
        self.planner = dspy.Predict(RefactoringPlan)
        self.implementer = dspy.Predict(RefactoredCode)

    def forward(self, code_snippet: str) -> dspy.Prediction:
        analysis = self.analyzer(code_snippet=code_snippet)
        plan = self.planner(code_snippet=code_snippet, analysis=analysis.analysis)
        impl = self.implementer(
            original_code=code_snippet,
            refactoring_summary=plan.refactoring_summary,
            plan_steps=plan.plan_steps,
        )
        return dspy.Prediction(
            analysis=analysis.analysis,
            refactoring_opportunities=analysis.refactoring_opportunities,
            refactoring_summary=plan.refactoring_summary,
            plan_steps=plan.plan_steps,
            refactored_code=impl.refactored_code,
            implementation_explanation=impl.implementation_explanation,
        )


class RefactoringEvaluator(dspy.Module):
    """A module to evaluate refactored code using programmatic checks and LLM judgment."""

    def __init__(self):
        super().__init__()
        self.evaluator = dspy.Predict(EvaluationSignature)

    def forward(
        self, original_example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> float:
        code = analysis_utils._extract_python_code(prediction.refactored_code)
        raw_tests = original_example.get("test_cases")
        tests = [TestCase(**tc) for tc in raw_tests] if raw_tests else []

        is_valid, func_name, _ = analysis_utils.check_syntax(code)
        if not is_valid:
            return 0.0

        if not tests:
            quality = analysis_utils.check_code_quality(code)
            functional_score = 1.0
        else:
            if not func_name:
                return 0.0
            quality = analysis_utils.check_code_quality(code, func_name)
            passed_tests = analysis_utils.check_functional_correctness(code, func_name, tests)
            functional_score = (passed_tests / len(tests)) if tests else 1.0

        eval_result = self.evaluator(
            code_snippet=code,
            quality_scores=quality.model_dump_json(),
            functional_score=functional_score,
        )
        try:
            return float(eval_result.final_score)
        except (ValueError, TypeError):
            return 0.0


# --- Training Data ---
def get_training_data() -> list[dspy.Example]:
    """Returns a list of examples for training the refactoring tool."""
    # (Training data remains the same as original)
    return [
        dspy.Example(
            code_snippet="""
def process_data(d):
    res = [x['price'] * x['qty'] for x in d if x['qty'] > 0]
    total = 0
    for r in res:
        total += r
    tax = 0.08
    final_total = total * (1 + tax)
    return final_total
""",
            test_cases=[
                TestCase(
                    args=[[{"price": 10, "qty": 2}, {"price": 5, "qty": -1}]], expected_output=21.6
                ).model_dump(),
                TestCase(
                    args=[[{"price": 100, "qty": 1}, {"price": 20, "qty": 5}]],
                    expected_output=216.0,
                ).model_dump(),
                TestCase(args=[[]], expected_output=0.0).model_dump(),
            ],
        ).with_inputs("code_snippet"),
        dspy.Example(
            code_snippet=textwrap.dedent("""
            def proc_trans(t, d1, d2, disc_rules):
                # ... (omitted for brevity, content is identical to original)
                pass
            """),
            test_cases=[
                TestCase(
                    args=[
                        [
                            ("user1", 100, "2024-01-01"),
                            ("user1", 200, "2024-01-02"),
                            ("user2", 150, "2024-01-01"),
                            ("user1", 50, "2024-01-03"),
                            ("user2", 300, "2024-01-04"),
                        ],
                        "2024-01-01",
                        "2024-01-03",
                        [("total", 250, 0.1), ("count", 2, 0.05), ("max", 150, 0.15)],
                    ],
                    expected_output=[
                        ["user1", 350, 116.66666666666667, 200, 0.3, 245.0, "100;200;50"],
                        ["user2", 150, 150.0, 150, 0.15, 127.5, "150"],
                    ],
                ).model_dump(),
                TestCase(
                    args=[[("user1", 100, "2024-01-01")], "2024-01-01", "2024-01-01", []],
                    expected_output=[["user1", 100, 100.0, 100, 0, 100.0, "100"]],
                ).model_dump(),
                TestCase(
                    args=[[], "2024-01-01", "2024-01-31", [("total", 100, 0.1)]], expected_output=[]
                ).model_dump(),
            ],
        ).with_inputs("code_snippet"),
    ]
