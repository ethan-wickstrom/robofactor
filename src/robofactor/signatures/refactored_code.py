import dspy

from robofactor.types import PythonCode, RefactoredArtifact, RefactoringPlanModel


class RefactoredCode(dspy.Signature):
    """
    Apply the refactoring plan to produce improved Python code.

    CRITICAL CONSTRAINTS:
    1. PRESERVE ALL FUNCTIONALITY - Code must behave identically to the original
    2. NO NEW CLASSES - Do not introduce classes unless they existed in original
    3. NO EMPTY DEFINITIONS - Every class, function, loop must have a body
    4. SYNTACTICALLY VALID - Code must parse without SyntaxError or IndentationError
    5. KEEP STRUCTURE - Functions stay functions, don't restructure into classes/modules

    Focus on: type hints, docstrings, readability, removing duplication, better names.
    """

    original_code: PythonCode = dspy.InputField(desc="Unmodified source code")
    plan: RefactoringPlanModel = dspy.InputField(desc="Structured refactoring plan")
    artifact: RefactoredArtifact = dspy.OutputField(
        description="Refactored code with explanation and key changes"
    )
