"""
Common utilities and shared functionality for Robofactor.
This package provides single sources of truth for cross-cutting concerns.
"""

from .ast_utils import ast_node_to_source
from .code_extraction import extract_python_code
from .json_validation import is_json_list, is_json_object, is_training_item

__all__ = [
    "ast_node_to_source",
    "extract_python_code",
    "is_json_list",
    "is_json_object",
    "is_training_item",
]
