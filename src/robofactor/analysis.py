import ast
import tempfile
from pathlib import Path

import dspy

from linting.ruff_tool import format_lint_issue, is_complexity_issue, run_ruff_json
from testing import check_functional_correctness as _testing_check_functional_correctness
from type_checking.metrics import docstring_and_typing_scores

from . import config
from .data import models
from .types import (
    ComplexityReport,
    DocumentationReport,
    LintingReport,
    PythonCode,
    QualityMetrics,
    TypingReport,
)


def check_syntax(code: PythonCode | str) -> tuple[bool, str | None, str | None]:
    """Check for valid Python syntax and top-level function definition."""
    source = code.code if isinstance(code, dspy.Code) else code
    try:
        tree = ast.parse(source)
        if func_node := next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None):
            return (True, func_node.name, None)
        return (False, None, "No top-level function definition found.")
    except SyntaxError as e:
        return (False, None, f"Syntax Error: {e}")


def check_code_quality(code: PythonCode | str, func_name: str | None = None) -> QualityMetrics:
    """Analyze Python code quality using ruff and AST metrics."""
    source = code.code if isinstance(code, dspy.Code) else code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(source)
        tmp_path = Path(tmp.name)

    try:
        records = run_ruff_json(tmp_path)

        complexity_issues = [format_lint_issue(rec) for rec in records if is_complexity_issue(rec)]
        linting_issues = [format_lint_issue(rec) for rec in records if not is_complexity_issue(rec)]

        complexity_score = 0.0 if complexity_issues else 1.0
        linting_score = max(
            0.0, 1.0 - (config.LINTING_PENALTY_PER_ISSUE * len(linting_issues))
        )

        docstring_score, typing_score = docstring_and_typing_scores(
            ast.parse(source), func_name
        )

        return QualityMetrics(
            linting=LintingReport(score=linting_score, issues=linting_issues),
            complexity=ComplexityReport(score=complexity_score, warnings=complexity_issues),
            typing=TypingReport(score=typing_score),
            documentation=DocumentationReport(score=docstring_score),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def check_functional_correctness(
    code: PythonCode | str, func_name: str, test_cases: list[models.TestCase]
) -> int:
    """Execute test cases against code in sandboxed interpreter, return pass count."""
    return _testing_check_functional_correctness(code, func_name, test_cases)
