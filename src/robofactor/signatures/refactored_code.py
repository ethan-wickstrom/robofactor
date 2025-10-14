import dspy

from robofactor.types import PythonCode, RefactoredArtifact, RefactoringPlanModel


class RefactoredCode(dspy.Signature):
    """
    Apply the refactoring plan to produce improved code.
    Output PEP8-compliant Python with type hints, docstrings, and a rationale
    for the changes.
    """

    original_code: PythonCode = dspy.InputField(desc="Unmodified source code")
    plan: RefactoringPlanModel = dspy.InputField(desc="Structured refactoring plan")
    artifact: RefactoredArtifact = dspy.OutputField(
        description="Refactored code and supporting notes"
    )
