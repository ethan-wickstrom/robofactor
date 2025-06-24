"""
Provides functions for checking the quality and correctness of Python code.

This module contains checkers for syntax validity, code quality (linting,
complexity, docstrings, typing), and functional correctness against test cases.
These functions are designed to be composed into an evaluation pipeline.
"""

import ast
import json
import os
import subprocess
import tempfile
import textwrap
from collections.abc import Sequence
from pathlib import Path

import dspy

from ..app import config
from ..parsing.models import CodeQualityScores, TestCase


def check_syntax(code: str) -> tuple[bool, str | None, str | None]:
    """
    Checks for valid Python syntax and a top-level function definition.

    Args:
        code: The Python source code to check.

    Returns:
        A tuple containing:
        - A boolean indicating if the syntax is valid.
        - The name of the top-level function if found, otherwise None.
        - An error message if the syntax is invalid, otherwise None.
    """
    try:
        tree = ast.parse(code)
        func_node = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        if not func_node:
            return False, None, "No top-level function definition found."
        return True, func_node.name, None
    except SyntaxError as e:
        return False, None, f"Syntax Error: {e}"


def _get_ast_based_scores(tree: ast.AST, func_name: str | None) -> tuple[float, float]:
    """Calculates docstring and typing scores from a parsed AST."""
    all_funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not all_funcs:
        return 0.0, 0.0

    target_funcs = (
        [f for f in all_funcs if f.name == func_name] if func_name else all_funcs
    )
    if not target_funcs:
        return 0.0, 0.0

    docstring_score = sum(1.0 for f in target_funcs if ast.get_docstring(f)) / len(
        target_funcs
    )

    typed_elements, typeable_elements = 0, 0
    for func_node in target_funcs:
        args = func_node.args
        typed_elements += sum(1 for arg in args.args if arg.annotation is not None)
        typed_elements += 1 if func_node.returns is not None else 0
        typeable_elements += len(args.args) + 1

    typing_score = typed_elements / typeable_elements if typeable_elements > 0 else 0.0
    return docstring_score, typing_score


def check_code_quality(code: str, func_name: str | None = None) -> CodeQualityScores:
    """
    Analyzes Python code for quality metrics using flake8 and AST.

    This function performs I/O by creating a temporary file and running a
    subprocess. It is designed to be wrapped by a decorator like `@safe`
    to handle potential exceptions.

    Args:
        code: The Python code to analyze.
        func_name: The specific function name to target for AST-based checks.

    Returns:
        A CodeQualityScores object with the analysis results.

    Raises:
        subprocess.CalledProcessError: If the flake8 command fails.
        SyntaxError: If the code cannot be parsed into an AST.
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        _ = tmp.write(code)
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                "flake8",
                f"--max-complexity={config.FLAKE8_MAX_COMPLEXITY}",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        all_issues = result.stdout.strip().splitlines() if result.stdout else []

        complexity_warnings = [
            issue for issue in all_issues if config.FLAKE8_COMPLEXITY_CODE in issue
        ]
        linting_issues = [
            issue for issue in all_issues if config.FLAKE8_COMPLEXITY_CODE not in issue
        ]

        complexity_score = 1.0 if not complexity_warnings else 0.0
        linting_score = max(
            0.0, 1.0 - (config.LINTING_PENALTY_PER_ISSUE * len(linting_issues))
        )

        tree = ast.parse(code)
        docstring_score, typing_score = _get_ast_based_scores(tree, func_name)

        return CodeQualityScores(
            linting_score=linting_score,
            complexity_score=complexity_score,
            typing_score=typing_score,
            docstring_score=docstring_score,
            linting_issues=linting_issues,
        )
    finally:
        if tmp_path.exists():
            os.unlink(tmp_path)


def _build_execution_script(func_name: str, test_case: TestCase) -> str:
    """Constructs a Python script to execute a function with a given test case."""
    args_json = json.dumps(test_case.args)
    kwargs_json = json.dumps(test_case.kwargs)

    return textwrap.dedent(
        f"""
        import json
        import sys

        args = json.loads('''{args_json}''')
        kwargs = json.loads('''{kwargs_json}''')

        result = {func_name}(*args, **kwargs)
        print(json.dumps(result))
        """
    )


def check_functional_correctness(
    code: str, func_name: str, test_cases: Sequence[TestCase]
) -> int:
    """
    Executes test cases against code in a sandboxed Python interpreter.

    Args:
        code: The Python source code containing the function.
        func_name: The name of the function to test.
        test_cases: A sequence of TestCase objects to run.

    Returns:
        The number of test cases that passed.

    Raises:
        Exception: If the PythonInterpreter fails during setup or execution.
    """
    if not test_cases:
        return 0

    passed_count = 0
    with dspy.PythonInterpreter() as interp:
        interp.execute(code)
        for test in test_cases:
            try:
                exec_script = _build_execution_script(func_name, test)
                actual_output_json = interp.execute(exec_script)
                actual_output = json.loads(actual_output_json)
                normalized_expected_output = json.loads(
                    json.dumps(test.expected_output)
                )
                if actual_output == normalized_expected_output:
                    passed_count += 1
            except Exception:
                continue
    return passed_count
