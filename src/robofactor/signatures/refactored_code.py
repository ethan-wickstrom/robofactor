from typing import Literal

import dspy


class RefactoredCode(dspy.Signature):
    """
    Apply the refactoring plan to produce improved code.
    Output PEP8-compliant Python with type hints, docstrings, and a rationale
    for the changes.
    """
    original_code: dspy.Code[Literal["python"]] = dspy.InputField(desc="Unmodified source code")
    refactoring_summary: str = dspy.InputField(desc="Refactoring objective")
    plan_steps: list[str] = dspy.InputField(desc="Step-by-step refactoring actions")
    refactored_code: dspy.Code[Literal["python"]] = dspy.OutputField(
        description="PEP8-compliant Python code with type hints and docstrings"
    )
    implementation_explanation: str = dspy.OutputField(
        description="Rationale for implemented changes"
    )