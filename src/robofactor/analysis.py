import ast
import json
import subprocess
import tempfile
import textwrap
from itertools import filterfalse
from pathlib import Path
from typing import cast

import dspy

from . import config
from .data import models
from .types import (
    ComplexityReport,
    DocumentationReport,
    LintDiagnostic,
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
            return True, func_node.name, None
        return False, None, "No top-level function definition found."
    except SyntaxError as e:
        return False, None, f"Syntax Error: {e}"


def _get_ast_based_scores(tree: ast.AST, func_name: str | None) -> tuple[float, float]:
    """Calculate docstring and typing coverage scores from AST."""
    all_funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not all_funcs:
        return 0.0, 0.0

    target_funcs = [f for f in all_funcs if f.name == func_name] if func_name else all_funcs
    if not target_funcs:
        return 0.0, 0.0

    docstring_score = sum(1.0 for f in target_funcs if ast.get_docstring(f)) / len(target_funcs)

    typed_elements = sum(
        sum(arg.annotation is not None for arg in f.args.args) + (f.returns is not None)
        for f in target_funcs
    )
    typeable_elements = sum(len(f.args.args) + 1 for f in target_funcs)

    typing_score = typed_elements / typeable_elements if typeable_elements > 0 else 0.0
    return docstring_score, typing_score


def check_code_quality(code: PythonCode | str, func_name: str | None = None) -> QualityMetrics:
    """Analyze Python code quality using ruff and AST metrics."""
    source = code.code if isinstance(code, dspy.Code) else code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(source)
        tmp_path = Path(tmp.name)

    try:
        return _compute_quality_scores(tmp_path, source, func_name)
    finally:
        tmp_path.unlink(missing_ok=True)


def _compute_quality_scores(tmp_path: Path, code: str, func_name: str | None) -> QualityMetrics:
    """Compute quality scores from ruff output and AST analysis."""
    result = subprocess.run(
        ["ruff", "check", "--output-format", "json", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    records: list[LintDiagnostic] = (
        cast(list[LintDiagnostic], json.loads(result.stdout)) if result.stdout else []
    )

    def format_issue(rec: LintDiagnostic) -> str:
        location = rec.get("location") or {}
        return (
            f"{rec.get('filename', '')}:{location.get('row', 0)}:{location.get('column', 0)} "
            f"{rec.get('code', '')} {rec.get('message', '')}"
        )

    def is_complexity(rec: LintDiagnostic) -> bool:
        return rec.get("code") == config.FLAKE8_COMPLEXITY_CODE

    complexity_records = list(filter(is_complexity, records))
    linting_records = list(filterfalse(is_complexity, records))
    complexity_issues = list(map(format_issue, complexity_records))
    linting_issues = list(map(format_issue, linting_records))

    complexity_score = 0.0 if complexity_records else 1.0
    linting_score = max(0.0, 1.0 - (config.LINTING_PENALTY_PER_ISSUE * len(linting_issues)))

    tree = ast.parse(code)
    docstring_score, typing_score = _get_ast_based_scores(tree, func_name)

    return QualityMetrics(
        linting=LintingReport(score=linting_score, issues=linting_issues),
        complexity=ComplexityReport(score=complexity_score, warnings=complexity_issues),
        typing=TypingReport(score=typing_score),
        documentation=DocumentationReport(score=docstring_score),
    )


def _build_execution_script(func_name: str, test_case: models.TestCase) -> str:
    """Build Python script to execute function with test case arguments."""
    return textwrap.dedent(
        f"""
        import json

        args = json.loads('''{json.dumps(test_case.args)}''')
        kwargs = json.loads('''{json.dumps(test_case.kwargs)}''')

        result = {func_name}(*args, **kwargs)
        print(json.dumps(result))
        """
    )


def check_functional_correctness(
    code: PythonCode | str, func_name: str, test_cases: list[models.TestCase]
) -> int:
    """Execute test cases against code in sandboxed interpreter, return pass count."""
    if not test_cases:
        return 0

    source = code.code if isinstance(code, dspy.Code) else code
    passed_count = 0
    with dspy.PythonInterpreter() as interp:
        interp.execute(source)
        for test in test_cases:
            try:
                exec_script = _build_execution_script(func_name, test)
                actual_output_json = interp.execute(exec_script)
                actual_output = json.loads(actual_output_json)
                normalized_expected = json.loads(json.dumps(test.expected_output))
                if actual_output == normalized_expected:
                    passed_count += 1
            except Exception:
                continue
    return passed_count
