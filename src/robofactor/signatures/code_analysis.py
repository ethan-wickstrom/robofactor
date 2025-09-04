from typing import Literal

import dspy


class CodeAnalysis(dspy.Signature):
    """
    Analyze Python code for its purpose, complexity, and dependencies.
    Identify actionable refactoring opportunities and summarize the findings.
    """
    code_snippet: dspy.Code[Literal["python"]] = dspy.InputField(desc="Python code to analyze")
    analysis: str = dspy.OutputField(description="Summary of functionality, complexity, and dependencies")
    refactoring_opportunities: list[str] = dspy.OutputField(
        description="Actionable bullet points for refactoring"
    )
