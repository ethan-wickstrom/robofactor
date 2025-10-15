"""Pure functional code analysis."""

import ast

from returns.result import Failure, Result, Success

from .data import models
from .types import (
    ComplexityReport,
    DocumentationReport,
    LintingReport,
    PythonCode,
    QualityMetrics,
    TypingReport,
)


def to_source(code: PythonCode | str) -> str:
    """Extract source string from PythonCode or return string as-is."""
    if isinstance(code, str):
        return code
    return code.code


def check_syntax(code: PythonCode | str) -> Result[str, str]:
    """Check valid Python syntax and extract top-level function name.

    Returns:
        Success with function name, or Failure with error message.
    """
    source = to_source(code)
    try:
        tree = ast.parse(source)
        func_node = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        return Success(func_node.name) if func_node else Failure("No top-level function found")
    except SyntaxError as e:
        return Failure(f"Syntax error: {e}")


def check_code_quality(code: PythonCode | str) -> QualityMetrics:
    """Analyze code quality via AST metrics.

    Returns quality metrics with conservative defaults for linting/typing/docs
    and AST-based complexity scoring.
    """
    source = to_source(code)
    try:
        tree = ast.parse(source)
        node_count = sum(1 for _ in ast.walk(tree))
        complexity_score = max(0.0, min(1.0, 1.0 - node_count / 500.0))
    except SyntaxError:
        complexity_score = 0.0

    return QualityMetrics(
        linting=LintingReport(score=1.0, issues=[]),
        complexity=ComplexityReport(score=complexity_score, warnings=[]),
        typing=TypingReport(score=0.0),
        documentation=DocumentationReport(score=0.0),
    )


def check_functional_correctness(
    code: PythonCode | str, func_name: str, test_cases: list[models.TestCase]
) -> int:
    """Execute test cases against function and return passed count.

    Warning: Uses exec on trusted code only.
    """
    if not test_cases:
        return 0

    source = to_source(code)
    env: dict[str, object] = {}

    try:
        exec(compile(source, "<refactor>", "exec"), env)
    except Exception:
        return 0

    fn = env.get(func_name)
    if not callable(fn):
        return 0

    passed = 0
    for test in test_cases:
        try:
            result = fn(*test.args, **test.kwargs)
            if result == test.expected_output:
                passed += 1
        except Exception:
            pass

    return passed


__all__ = [
    "check_code_quality",
    "check_functional_correctness",
    "check_syntax",
    "to_source",
]
