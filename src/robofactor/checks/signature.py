from __future__ import annotations

import ast

from returns.result import Failure, Result, Success

from robofactor.checks.model import FunctionParameter, FunctionSignature


def parse_function_signature(code: str) -> Result[FunctionSignature, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return Failure(f"syntax error: {error}")

    if violation := _top_level_violation(tree):
        return Failure(violation)

    public_functions = tuple(
        statement
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef) and not statement.name.startswith("_")
    )
    match public_functions:
        case (function,):
            return Success(
                FunctionSignature(
                    name=function.name,
                    parameters=_function_parameters(function.args),
                )
            )
        case ():
            return Failure("no public top-level function found")
        case _:
            return Failure("expected exactly one public top-level function")


def _top_level_violation(tree: ast.Module) -> str | None:
    for statement in tree.body:
        match statement:
            case ast.Import() | ast.ImportFrom() | ast.FunctionDef() | ast.ClassDef():
                continue
            case ast.Expr(value=ast.Constant(value=str())):
                continue
            case _:
                return "top-level function checks do not allow executable module statements"
    return None


def _function_parameters(args: ast.arguments) -> tuple[FunctionParameter, ...]:
    positional = (*args.posonlyargs, *args.args)
    positional_defaults = _positional_defaults(positional, args.defaults)
    return (
        *(
            FunctionParameter(kind="positional_only", name=arg.arg, default=default)
            for arg, default in zip(
                args.posonlyargs,
                positional_defaults[: len(args.posonlyargs)],
                strict=True,
            )
        ),
        *(
            FunctionParameter(kind="positional_or_keyword", name=arg.arg, default=default)
            for arg, default in zip(
                args.args,
                positional_defaults[len(args.posonlyargs) :],
                strict=True,
            )
        ),
        *(
            (FunctionParameter(kind="var_positional", name=args.vararg.arg, default=None),)
            if args.vararg
            else ()
        ),
        *(
            FunctionParameter(kind="keyword_only", name=arg.arg, default=_default_text(default))
            for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
        ),
        *(
            (FunctionParameter(kind="var_keyword", name=args.kwarg.arg, default=None),)
            if args.kwarg
            else ()
        ),
    )


def _positional_defaults(
    positional: tuple[ast.arg, ...],
    defaults: list[ast.expr],
) -> tuple[str | None, ...]:
    required_count = len(positional) - len(defaults)
    return (
        *(None for _ in positional[:required_count]),
        *(_default_text(default) for default in defaults),
    )


def _default_text(default: ast.expr | None) -> str | None:
    return None if default is None else ast.unparse(default)
