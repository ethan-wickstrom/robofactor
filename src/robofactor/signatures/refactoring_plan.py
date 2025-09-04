from typing import Literal

import dspy


class RefactoringPlan(dspy.Signature):
    """
    Formulate a high-level refactoring goal and sequential action plan based on given code and analysis.
    Return a clear summary and an ordered list of steps.
    """
    code_snippet: dspy.Code[Literal["python"]] = dspy.InputField(desc="Original Python code")
    analysis: str = dspy.InputField(desc="Code analysis summary")
    refactoring_summary: str = dspy.OutputField(description="High-level refactoring objective")
    plan_steps: list[str] = dspy.OutputField(description="Sequential actions to achieve refactoring")
