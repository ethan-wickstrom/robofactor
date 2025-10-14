from __future__ import annotations

import ast


def docstring_and_typing_scores(tree: ast.AST, func_name: str | None) -> tuple[float, float]:
    """Compute (docstring_score, typing_score) from an AST, optionally scoped to a function.

    - docstring_score: fraction of target functions that have a docstring.
    - typing_score: fraction of typeable elements (args + return) that are annotated across target functions.
    """
    all_funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not all_funcs:
        return (0.0, 0.0)

    target_funcs = [f for f in all_funcs if f.name == func_name] if func_name else all_funcs
    if not target_funcs:
        return (0.0, 0.0)

    docstring_score = sum(1.0 for f in target_funcs if ast.get_docstring(f)) / len(target_funcs)

    typed_elements = sum(
        sum(arg.annotation is not None for arg in f.args.args) + (f.returns is not None)
        for f in target_funcs
    )
    typeable_elements = sum(len(f.args.args) + 1 for f in target_funcs)
    typing_score = typed_elements / typeable_elements if typeable_elements > 0 else 0.0

    return (docstring_score, typing_score)
