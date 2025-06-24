"""
Defines the core DSPy module for the code refactoring pipeline.
"""

import dspy

from .signatures import CodeAnalysis, RefactoringPlan, RefactoredCode


class CodeRefactor(dspy.Module):
    """Orchestrates code analysis, planning, and refactoring."""

    def __init__(self) -> None:
        """Initializes the multi-stage refactoring module."""
        super().__init__()
        self.analyzer: dspy.Module = dspy.Predict(CodeAnalysis)
        self.planner: dspy.Module = dspy.Predict(RefactoringPlan)
        self.implementer: dspy.Module = dspy.Predict(RefactoredCode)

    def forward(self, code_snippet: str) -> dspy.Prediction:
        """
        Executes the analysis, planning, and implementation pipeline.

        Args:
            code_snippet: The Python code to be refactored.

        Returns:
            A dspy.Prediction object containing the full trace of the
            refactoring process, from analysis to final code.
        """
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
