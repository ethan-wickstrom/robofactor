"""
Provides utility functions for parsing and extracting code from text.
"""

import re


def extract_python_code(text: str) -> str:
    """
    Extracts Python code from a markdown block.

    If a python markdown block (```python...```) is found, its content is
    returned. Otherwise, the original text is returned.

    Args:
        text: The string to search for a Python code block.

    Returns:
        The extracted Python code, or the original text if no block is found.
    """
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else text
