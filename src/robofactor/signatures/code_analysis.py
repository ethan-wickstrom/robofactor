import dspy

from robofactor.types import CodeAnalysisReport, PythonCode


class CodeAnalysis(dspy.Signature):
    """
    Analyze Python code for its purpose, complexity, and dependencies.
    Identify actionable refactoring opportunities and summarize the findings.
    """

    code_snippet: PythonCode = dspy.InputField(desc="Python code to analyze")
    report: CodeAnalysisReport = dspy.OutputField(description="Structured analysis findings")
