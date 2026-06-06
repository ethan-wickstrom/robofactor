from __future__ import annotations

import ast
import io
import subprocess
import sys
import tempfile
import tokenize
from collections.abc import Sequence
from pathlib import Path

from robofactor.checks.model import QualityReport

RUFF_RULES = (
    "E",
    "W",
    "F",
    "I",
    "N",
    "B",
    "UP",
    "RUF",
    "C901",
    "C4",
    "SIM",
    "RET",
    "ARG",
    "PIE",
    "PLC",
    "PLE",
    "PLW",
)
SUPPRESSION_COMMENT_PREFIXES = (
    "# noqa",
    "# ruff: noqa",
    "# type: ignore",
    "# pyright: ignore",
    "# ty: ignore",
    "# mypy: ignore-errors",
    "# pylint: disable",
    "# pragma: no cover",
)


def python_quality(code: str, function_name: str) -> QualityReport:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as file:
        file.write(code)
        path = Path(file.name)
    try:
        ruff_issues = _ruff_issues(path)
        ty_issues = _ty_issues(path)
        tree = ast.parse(code)
        typing_score, docstring_score = _ast_scores(tree, function_name)
        complexity_issues = tuple(issue for issue in ruff_issues if "C901" in issue)
        return QualityReport(
            ruff_passed=not ruff_issues,
            ty_passed=not ty_issues,
            typing_score=typing_score,
            docstring_score=docstring_score,
            complexity_score=0.0 if complexity_issues else 1.0,
            ruff_issues=ruff_issues,
            ty_issues=ty_issues,
            complexity_issues=complexity_issues,
        )
    finally:
        path.unlink(missing_ok=True)


def suppression_warnings(code: str) -> tuple[str, ...]:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    except tokenize.TokenError as error:
        return (f"could not scan suppression comments: {error}",)

    return tuple(
        f"forbidden suppression comment: {token.string.strip()}"
        for token in tokens
        if token.type == tokenize.COMMENT and _is_suppression_comment(token.string)
    )


def _is_suppression_comment(comment: str) -> bool:
    normalized = " ".join(comment.lower().split())
    return any(normalized.startswith(prefix) for prefix in SUPPRESSION_COMMENT_PREFIXES)


def _ruff_issues(path: Path) -> tuple[str, ...]:
    return _python_tool_issues(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--output-format=concise",
            "--select",
            ",".join(RUFF_RULES),
            "--ignore",
            "E501",
            "--config",
            "lint.mccabe.max-complexity = 10",
            str(path),
        ]
    )


def _ty_issues(path: Path) -> tuple[str, ...]:
    return _python_tool_issues(
        [
            sys.executable,
            "-m",
            "ty",
            "check",
            "--output-format=concise",
            "--python-version=3.14",
            str(path),
        ]
    )


def _python_tool_issues(command: Sequence[str]) -> tuple[str, ...]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        return _diagnostic_lines(result.stderr)
    return _diagnostic_lines(result.stdout)


def _diagnostic_lines(output: str) -> tuple[str, ...]:
    ignored = {"All checks passed!"}
    return tuple(
        line
        for line in output.strip().splitlines()
        if line and line not in ignored and not line.startswith("Found ")
    )


def _ast_scores(tree: ast.Module, function_name: str) -> tuple[float, float]:
    for statement in tree.body:
        match statement:
            case ast.FunctionDef(name=name, args=args, returns=returns) if name == function_name:
                parameters = (*args.posonlyargs, *args.args, *args.kwonlyargs)
                typed_parameters = sum(parameter.annotation is not None for parameter in parameters)
                typed_return = 1 if returns is not None else 0
                typeable = len(parameters) + 1
                typing_score = (typed_parameters + typed_return) / typeable
                docstring_score = 1.0 if ast.get_docstring(statement) else 0.0
                return typing_score, docstring_score
            case _:
                continue

    raise RuntimeError(f"function not found after successful parse: {function_name}")
