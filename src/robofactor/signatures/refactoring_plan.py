import dspy

from robofactor.types import CodeAnalysisReport, PythonCode, RefactoringPlanModel


class RefactoringPlan(dspy.Signature):
    """
    Formulate a high-level refactoring goal and sequential action plan based on given code and analysis.
    Return a clear summary and an ordered list of steps.
    """

    code_snippet: PythonCode = dspy.InputField(desc="Original Python code")
    analysis: CodeAnalysisReport = dspy.InputField(desc="Structured code analysis details")
    plan: RefactoringPlanModel = dspy.OutputField(
        description="Refactoring objective and ordered steps"
    )
