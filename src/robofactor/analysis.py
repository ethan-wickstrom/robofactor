import ast
import json
import re
import subprocess
import tempfile
import textwrap
from itertools import filterfalse
from pathlib import Path
from typing import cast

import dspy
from dspy.adapters.types.code import Code as DSPyCode

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
    create_python_code,
)


def _to_python_code(value: PythonCode | str) -> PythonCode:
    return value if isinstance(value, DSPyCode) else create_python_code(value)


def extract_python_code(text: PythonCode | str) -> PythonCode:
    """Extract Python code from a fenced markdown block.

    - Supports optional leading indentation before fences.
    - Returns original text if no Python fence is found.
    """
    if isinstance(text, DSPyCode):
        return text

    pattern = re.compile(r"^[ \t]*```python\s*\n(.*?)\n[ \t]*```", re.DOTALL | re.MULTILINE)
    match = pattern.search(text)
    extracted = match[1].strip() if match else text
    return _to_python_code(extracted)


def check_syntax(code: PythonCode) -> tuple[bool, str | None, str | None]:
    """
    Checks for valid Python syntax and a top-level function definition.

    Returns a tuple indicating validity, the function name, and an error message.
    This format is consumed by a wrapper that converts it into a `Result` monad.
    """
    source = code.code
    try:
        tree = ast.parse(source)
        if func_node := next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None):
            return True, func_node.name, None
        else:
            return False, None, "No top-level function definition found."
    except SyntaxError as e:
        return False, None, f"Syntax Error: {e}"


def _get_ast_based_scores(tree: ast.AST, func_name: str | None) -> tuple[float, float]:
    """Calculates docstring and typing scores from a parsed AST."""
    all_funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not all_funcs:
        return 0.0, 0.0

    target_funcs = [f for f in all_funcs if f.name == func_name] if func_name else all_funcs
    if not target_funcs:
        return 0.0, 0.0

    docstring_score = sum(1.0 for f in target_funcs if ast.get_docstring(f)) / len(target_funcs)

    typed_elements, typeable_elements = 0, 0
    for func_node in target_funcs:
        args = func_node.args
        typed_elements += sum(arg.annotation is not None for arg in args.args)
        typed_elements += 1 if func_node.returns is not None else 0
        typeable_elements += len(args.args) + 1

    typing_score = typed_elements / typeable_elements if typeable_elements > 0 else 0.0
    return docstring_score, typing_score


def check_code_quality(code: PythonCode, func_name: str | None = None) -> QualityMetrics:
    """
    Analyzes Python code for quality metrics using flake8 and AST.

    This function performs I/O by creating a temporary file and running a
    subprocess. It is designed to be wrapped by a decorator like `@safe`
    or `@impure_safe` to handle potential exceptions.
    """
    python_code = _to_python_code(code)
    source = python_code.code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(source)
        tmp_path = Path(tmp.name)

    try:
        return _compute_quality_scores(tmp_path, source, func_name)
    finally:
        # Ensure the temporary file is always cleaned up.
        tmp_path.unlink(missing_ok=True)


def _compute_quality_scores(tmp_path: Path, code: str, func_name: str | None) -> QualityMetrics:
    # Exceptions from subprocess.run will be caught by the @safe wrapper in the caller.
    # Use Ruff for linting and complexity (C901) analysis, outputting JSON for parsing.
    result = subprocess.run(
        [
            "ruff",
            "check",
            "--output-format",
            "json",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    records: list[LintDiagnostic] = (
        cast(list[LintDiagnostic], json.loads(result.stdout)) if result.stdout else []
    )

    def format_issue(rec: LintDiagnostic) -> str:
        filename = rec.get("filename", "")
        location = rec.get("location") or {}
        row = location.get("row", 0)
        col = location.get("column", 0)
        code = rec.get("code", "")
        message = rec.get("message", "")
        return f"{filename}:{row}:{col} {code} {message}"

    def is_complexity(rec: LintDiagnostic) -> bool:
        return rec.get("code") == config.FLAKE8_COMPLEXITY_CODE

    complexity_records = list(filter(is_complexity, records))
    linting_records = list(filterfalse(is_complexity, records))
    complexity_issues = list(map(format_issue, complexity_records))
    linting_issues = list(map(format_issue, linting_records))

    complexity_score = 0.0 if complexity_records else 1.0
    linting_score = max(0.0, 1.0 - (config.LINTING_PENALTY_PER_ISSUE * len(linting_issues)))

    # A SyntaxError here will be caught by the @safe wrapper in the caller.
    tree = ast.parse(code)
    docstring_score, typing_score = _get_ast_based_scores(tree, func_name)

    return QualityMetrics(
        linting=LintingReport(score=linting_score, issues=linting_issues),
        complexity=ComplexityReport(score=complexity_score, warnings=complexity_issues),
        typing=TypingReport(score=typing_score),
        documentation=DocumentationReport(score=docstring_score),
    )


def _build_execution_script(func_name: str, test_case: models.TestCase) -> str:
    """Constructs a Python script to execute a function with a given test case."""
    args_json = json.dumps(test_case.args)
    kwargs_json = json.dumps(test_case.kwargs)

    return textwrap.dedent(
        f"""
        import json
        import sys

        # This script assumes the function '{func_name}' has been defined in the
        # execution context by the dspy.PythonInterpreter.

        args = json.loads('''{args_json}''')
        kwargs = json.loads('''{kwargs_json}''')

        result = {func_name}(*args, **kwargs)
        print(json.dumps(result))
        """
    )


def check_functional_correctness(
    code: PythonCode, func_name: str, test_cases: list[models.TestCase]
) -> int:
    """
    Executes test cases against code in a sandboxed Python interpreter.

    This function can raise exceptions if the provided code is invalid or if
    the test execution fails unexpectedly. It is designed to be wrapped by a

    decorator like `@safe` to handle these failures gracefully.
    """
    if not test_cases:
        return 0

    source = code.code
    passed_count = 0
    # A failure in the interpreter setup will be caught by the @safe wrapper.
    with dspy.PythonInterpreter() as interp:
        interp.execute(source)  # Define the function in the interpreter's scope.
        for test in test_cases:
            # Handle failures for individual test cases gracefully to allow others to run.
            try:
                exec_script = _build_execution_script(func_name, test)
                actual_output_json = interp.execute(exec_script)
                actual_output = json.loads(actual_output_json)

                # Normalize expected output to ensure consistent comparison.
                normalized_expected_output = json.loads(json.dumps(test.expected_output))
                if actual_output == normalized_expected_output:
                    passed_count += 1
            except Exception:
                # If a single test case fails to execute or assert, continue to the next.
                continue
    return passed_count
