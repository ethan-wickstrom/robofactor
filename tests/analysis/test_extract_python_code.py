from robofactor.analysis import extract_python_code
from robofactor.types import create_python_code


def test_extracts_code_from_fence():
    text = """
    Some docs
    ```python
    def add(a, b):
        return a + b
    ```
    more text
    """.strip()

    extracted = extract_python_code(text)
    assert extracted.code.strip().startswith("def add(")
    assert "return a + b" in extracted.code


def test_returns_original_when_no_fence_present():
    code = "def mul(a, b):\n    return a * b"
    assert extract_python_code(code).code == code


def test_passthrough_for_python_code_instances():
    code = create_python_code("print('hi')\n")
    assert extract_python_code(code) is code
