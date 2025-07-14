"""
Single source of truth for extracting Python code from text/markdown.
"""

import re
from typing import Final

_CODE_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


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
    match = _CODE_BLOCK_PATTERN.search(text)
    return match.group(1).strip() if match else text
