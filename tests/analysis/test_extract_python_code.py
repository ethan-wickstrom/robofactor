from robofactor.analysis import extract_python_code


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
    assert extracted.strip().startswith("def add(")
    assert "return a + b" in extracted


def test_returns_original_when_no_fence_present():
    code = "def mul(a, b):\n    return a * b"
    assert extract_python_code(code) == code

