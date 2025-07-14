# Modern Python 3.13 Cheat-Sheet

## 1. Type Aliases (PEP 695)

```python
type UserId = int
type Result[T, E] = Success[T] | Failure[E]
```

Use them anywhere you repeat a complex type.

## 2. Safe Null Handling (Maybe)

Install: `pip install returns`

```python
from returns.maybe import Maybe, Some, Nothing

def safe_get(d, k) -> Maybe[V]:
    return Some(d[k]) if k in d else Nothing

match safe_get(data, "key"):
    case Some(v): use(v)
    case Nothing: handle_missing()
```

Chain safely with `.bind()` instead of `if x is not None`.

## 3. Type Guards (PEP 647)

```python
from typing import TypeGuard

def is_email(s: str) -> TypeGuard[Email]:
    return "@" in s and "." in s.split("@")[1]

if is_email(raw):
    # raw is now Email
    send_mail(raw)
```

No casts, no runtime surprises.
