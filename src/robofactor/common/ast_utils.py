"""
Single source of truth for AST-related utilities.
"""

import ast


def ast_node_to_source(node: ast.AST) -> str:
    """
    Convert an AST node back to its source code representation.

    Args:
        node: The AST node to convert.

    Returns:
        The source code string for the node, or a repr for fallback.
    """
    try:
        return ast.unparse(node)
    except Exception:
        # Fallback for nodes that ast.unparse might not handle gracefully.
        # This ensures that even complex or unusual AST structures can be represented.
        return repr(node)
