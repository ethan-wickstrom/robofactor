# Using Modern Python 3.13 Type System Patterns

This document presents three fundamental patterns for leveraging Python 3.13's enhanced type system: type aliases using PEP 695 syntax, the Maybe monad pattern for nullable values, and TypeGuard functions for safe type narrowing. These patterns enable developers to write more expressive, type-safe code while avoiding unsafe practices such as type casting and runtime type assertions.

---

## 1. Type Aliases with PEP 695 Syntax

### 1.1 Functional Description

Type aliases in Python 3.13 provide a mechanism for creating semantic type synonyms that enhance code readability and maintainability. The new PEP 695 syntax introduces the `type` statement, which creates type aliases with improved scoping rules and cleaner syntax compared to traditional type variable assignments.

**Key Properties:**

- **Semantic clarity**: Type aliases communicate domain-specific meaning
- **Composability**: Complex types can be built from simpler components
- **Zero runtime overhead**: Aliases exist only at type-checking time
- **Generic support**: Type parameters can be directly specified

### 1.2 Instructions for Implementation

1. **Basic Type Alias Definition**

   ```python
   type UserId = int
   type Email = str
   type Timestamp = float
   ```

2. **Complex Type Aliases**

   ```python
   type JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None
   type HttpHeaders = dict[str, str]
   type QueryParams = dict[str, list[str]]
   ```

3. **Generic Type Aliases**

   ```python
   type Result[T, E] = Success[T] | Failure[E]
   type Predicate[T] = Callable[[T], bool]
   type Transform[A, B] = Callable[[A], B]
   ```

4. **Nested Type Aliases**

   ```python
   type Point2D = tuple[float, float]
   type Line = tuple[Point2D, Point2D]
   type Polygon = list[Point2D]
   ```

### 1.3 Practical Examples

#### Example 1: Domain Modeling with Type Aliases

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

# Define domain-specific type aliases
type CustomerId = int
type OrderId = int
type ProductCode = str
type Quantity = int
type Price = Decimal
type Currency = Literal["USD", "EUR", "GBP"]
type OrderStatus = Literal["pending", "confirmed", "shipped", "delivered", "cancelled"]

# Use aliases in data structures
@dataclass(frozen=True)
class OrderItem:
    product: ProductCode
    quantity: Quantity
    unit_price: Price

@dataclass(frozen=True)
class Order:
    id: OrderId
    customer: CustomerId
    items: tuple[OrderItem, ...]
    currency: Currency
    status: OrderStatus

    @property
    def total_price(self) -> Price:
        return sum(item.unit_price * item.quantity for item in self.items)
```

#### Example 2: Graph Algorithm Type Aliases

```python
from collections.abc import Mapping, Set

# Define graph-related type aliases
type NodeId = str
type Weight = float
type Edge = tuple[NodeId, NodeId, Weight]
type AdjacencyList = Mapping[NodeId, Set[NodeId]]
type WeightedAdjacencyList = Mapping[NodeId, Mapping[NodeId, Weight]]
type Path = list[NodeId]
type Distance = float | float('inf')

def dijkstra(
    graph: WeightedAdjacencyList,
    start: NodeId,
    end: NodeId
) -> tuple[Distance, Path]:
    """Find shortest path between nodes using Dijkstra's algorithm."""
    # Implementation details omitted for brevity
    pass
```

#### Example 3: Parser Combinator Type Aliases

```python
from typing import TypeVar, Callable
from collections.abc import Sequence

# Generic type aliases for parser combinators
type ParseResult[T] = tuple[T, str] | None
type Parser[T] = Callable[[str], ParseResult[T]]
type Combinator[A, B] = Callable[[Parser[A]], Parser[B]]

# Specific parser type aliases
type TokenParser = Parser[str]
type NumberParser = Parser[float]
type IdentifierParser = Parser[str]

def sequence[T](parsers: Sequence[Parser[T]]) -> Parser[list[T]]:
    """Combine multiple parsers in sequence."""
    def parse(input_str: str) -> ParseResult[list[T]]:
        results = []
        remaining = input_str
        for parser in parsers:
            result = parser(remaining)
            if result is None:
                return None
            value, remaining = result
            results.append(value)
        return results, remaining
    return parse
```

---

## 2. The Maybe Pattern for Nullable Values

### 2.1 Functional Description

The Maybe pattern, borrowed from functional programming languages like Haskell, provides a type-safe alternative to nullable references. It explicitly models the presence or absence of a value, forcing developers to handle both cases and eliminating null pointer exceptions at the type level.

**Mathematical Foundation:**
The Maybe type forms a monad with the following operations:

- `return`: Wraps a value in Some
- `bind`: Chains computations that may produce Nothing
- Identity laws and associativity hold

**Key Benefits:**

- **Explicit null handling**: Absence is a first-class concept
- **Composability**: Chain operations without null checks
- **Type safety**: Prevents null pointer exceptions
- **Functional purity**: No hidden nulls or exceptions

### 2.2 Instructions for Implementation

1. **Import Required Types**

   ```python
   from returns.maybe import Maybe, Some, Nothing
   from returns.pointfree import bind
   from returns.pipeline import flow
   ```

2. **Creating Maybe Values**

   ```python
   # From a value
   maybe_value = Some(42)

   # Representing absence
   no_value = Nothing

   # From optional
   maybe_from_optional = Maybe.from_optional(some_optional_value)
   ```

3. **Pattern Matching on Maybe**

   ```python
   match maybe_value:
       case Some(value):
           # Handle the present value
           process(value)
       case Nothing:
           # Handle absence
           handle_missing()
   ```

4. **Chaining Operations**

   ```python
   result = flow(
       initial_value,
       parse_input,
       bind(validate),
       bind(transform),
       bind(save_to_database)
   )
   ```

### 2.3 Practical Examples

#### Example 1: Safe Dictionary Access

```python
from returns.maybe import Maybe, Some, Nothing
from typing import TypeVar, Mapping

K = TypeVar('K')
V = TypeVar('V')

def safe_get[K, V](mapping: Mapping[K, V], key: K) -> Maybe[V]:
    """Safely retrieve a value from a mapping."""
    try:
        return Some(mapping[key])
    except KeyError:
        return Nothing

def get_nested_value(data: dict[str, dict[str, int]],
                    outer_key: str,
                    inner_key: str) -> Maybe[int]:
    """Safely navigate nested dictionaries."""
    return (
        safe_get(data, outer_key)
        .bind(lambda inner_dict: safe_get(inner_dict, inner_key))
    )

# Usage example
data = {
    "users": {"alice": 42, "bob": 17},
    "admins": {"charlie": 99}
}

# This returns Some(42)
alice_value = get_nested_value(data, "users", "alice")

# This returns Nothing (no "eve" in users)
eve_value = get_nested_value(data, "users", "eve")

# This returns Nothing (no "guests" key)
guest_value = get_nested_value(data, "guests", "anyone")
```

#### Example 2: Configuration Parsing

```python
from returns.maybe import Maybe, Some, Nothing
from returns.pointfree import bind
from pathlib import Path
import json

@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    username: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Maybe[DatabaseConfig]:
        """Parse database configuration from dictionary."""
        try:
            return Some(cls(
                host=data["host"],
                port=int(data["port"]),
                username=data["username"]
            ))
        except (KeyError, ValueError, TypeError):
            return Nothing

def load_config(path: Path) -> Maybe[DatabaseConfig]:
    """Load configuration from JSON file."""
    def read_file(p: Path) -> Maybe[str]:
        try:
            return Some(p.read_text())
        except (IOError, OSError):
            return Nothing

    def parse_json(content: str) -> Maybe[dict[str, Any]]:
        try:
            return Some(json.loads(content))
        except json.JSONDecodeError:
            return Nothing

    return (
        read_file(path)
        .bind(parse_json)
        .bind(DatabaseConfig.from_dict)
    )

# Usage
config_path = Path("config.json")
match load_config(config_path):
    case Some(config):
        print(f"Connecting to {config.host}:{config.port}")
    case Nothing:
        print("Failed to load configuration, using defaults")
```

#### Example 3: User Authentication Chain

```python
from returns.maybe import Maybe, Some, Nothing
from returns.pipeline import flow
from returns.pointfree import bind
import hashlib
from dataclasses import dataclass

@dataclass(frozen=True)
class User:
    id: int
    username: str
    password_hash: str
    is_active: bool

type UserId = int
type Username = str
type SessionToken = str

def find_user_by_username(username: Username) -> Maybe[User]:
    """Lookup user in database by username."""
    # Simulated database lookup
    users = {
        "alice": User(1, "alice", hashlib.sha256(b"secret123").hexdigest(), True),
        "bob": User(2, "bob", hashlib.sha256(b"password").hexdigest(), False),
    }
    return Maybe.from_optional(users.get(username))

def verify_password(password: str, user: User) -> Maybe[User]:
    """Verify password matches stored hash."""
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return Some(user) if password_hash == user.password_hash else Nothing

def check_active(user: User) -> Maybe[User]:
    """Ensure user account is active."""
    return Some(user) if user.is_active else Nothing

def generate_session(user: User) -> Maybe[SessionToken]:
    """Generate session token for authenticated user."""
    # Simplified token generation
    return Some(f"session_{user.id}_{user.username}")

def authenticate(username: Username, password: str) -> Maybe[SessionToken]:
    """Complete authentication pipeline."""
    return flow(
        find_user_by_username(username),
        bind(lambda user: verify_password(password, user)),
        bind(check_active),
        bind(generate_session)
    )

# Usage examples
match authenticate("alice", "secret123"):
    case Some(token):
        print(f"Authentication successful: {token}")
    case Nothing:
        print("Authentication failed")

match authenticate("bob", "password"):
    case Some(token):
        print(f"Authentication successful: {token}")
    case Nothing:
        print("Authentication failed")  # This will print (Bob is inactive)
```

---

## 3. TypeGuard Pattern for Safe Type Narrowing

### 3.1 Functional Description

TypeGuard functions provide a type-safe mechanism for narrowing types within conditional branches without resorting to unsafe type casting. Introduced in PEP 647, TypeGuards establish a contract between runtime checks and static type analysis, enabling type checkers to understand type refinements based on boolean predicates.

**Formal Properties:**

- **Soundness**: If a TypeGuard returns True, the type narrowing is guaranteed to be valid
- **Composability**: TypeGuards can be combined using logical operators
- **No runtime overhead**: TypeGuards are regular functions with special type annotations
- **Static verification**: Type checkers validate TypeGuard usage at compile time

### 3.2 Instructions for Implementation

1. **Basic TypeGuard Structure**

   ```python
   from typing import TypeGuard

   def is_type_name(value: broader_type) -> TypeGuard[narrower_type]:
       """Check if value is of narrower_type."""
       return isinstance(value, narrower_type)  # or other validation
   ```

2. **TypeGuard Requirements**

   - Must return a boolean value
   - The guarded type must be a subtype of the input type
   - The function body must actually perform the check
   - Should be pure (no side effects)

3. **Using TypeGuards**

   ```python
   if is_type_name(value):
       # value is now narrowed to narrower_type
       use_narrowed_value(value)
   else:
       # value remains as broader_type
       handle_other_case(value)
   ```

### 3.3 Practical Examples

#### Example 1: Literal Type Narrowing

```python
from typing import TypeGuard, Literal, get_args

# Define a complex literal type
type HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
type SafeMethod = Literal["GET", "HEAD", "OPTIONS"]
type DatabaseOperation = Literal["SELECT", "INSERT", "UPDATE", "DELETE"]

def is_http_method(value: str) -> TypeGuard[HttpMethod]:
    """Check if a string is a valid HTTP method."""
    valid_methods = get_args(HttpMethod)
    return value in valid_methods

def is_safe_method(method: HttpMethod) -> TypeGuard[SafeMethod]:
    """Check if an HTTP method is safe (no side effects)."""
    safe_methods = get_args(SafeMethod)
    return method in safe_methods

def is_database_operation(value: str) -> TypeGuard[DatabaseOperation]:
    """Check if a string is a valid database operation."""
    valid_ops = get_args(DatabaseOperation)
    return value in valid_ops

# Usage example
def process_request(method_str: str, path: str) -> str:
    """Process HTTP request with proper type narrowing."""
    if not is_http_method(method_str):
        return f"Invalid HTTP method: {method_str}"

    # method_str is now narrowed to HttpMethod
    if is_safe_method(method_str):
        # method_str is now narrowed to SafeMethod
        return f"Safe request: {method_str} {path}"
    else:
        # method_str is HttpMethod but not SafeMethod
        return f"Unsafe request: {method_str} {path} (requires authentication)"

# Examples
print(process_request("GET", "/users"))      # Safe request: GET /users
print(process_request("POST", "/users"))     # Unsafe request: POST /users (requires authentication)
print(process_request("INVALID", "/users"))  # Invalid HTTP method: INVALID
```

#### Example 2: Structural Type Validation

```python
from typing import TypeGuard, Protocol, Any
from dataclasses import dataclass

# Define protocols for structural typing
class Comparable(Protocol):
    def __lt__(self, other: Any) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...

class Sized(Protocol):
    def __len__(self) -> int: ...

class Container[T](Protocol):
    def __contains__(self, item: T) -> bool: ...
    def __len__(self) -> int: ...

# TypeGuard functions for protocol checking
def is_comparable(obj: object) -> TypeGuard[Comparable]:
    """Check if object implements Comparable protocol."""
    return (
        hasattr(obj, '__lt__') and
        callable(getattr(obj, '__lt__')) and
        hasattr(obj, '__eq__') and
        callable(getattr(obj, '__eq__'))
    )

def is_sized(obj: object) -> TypeGuard[Sized]:
    """Check if object implements Sized protocol."""
    return hasattr(obj, '__len__') and callable(getattr(obj, '__len__'))

def is_container(obj: object) -> TypeGuard[Container[Any]]:
    """Check if object implements Container protocol."""
    return (
        is_sized(obj) and
        hasattr(obj, '__contains__') and
        callable(getattr(obj, '__contains__'))
    )

# Usage with type narrowing
def find_min[T](items: object) -> T | None:
    """Find minimum value in a comparable container."""
    if not is_container(items):
        return None

    # items is now Container[Any]
    if len(items) == 0:
        return None

    result = None
    for item in items:  # Assuming Container is iterable
        if result is None:
            result = item
        elif is_comparable(item) and is_comparable(result):
            if item < result:
                result = item

    return result

# Examples
print(find_min([3, 1, 4, 1, 5]))  # 1
print(find_min("hello"))           # 'e'
print(find_min(42))                # None (not a container)
```

#### Example 3: Complex Data Validation

```python
from typing import TypeGuard, Any, TypedDict
from datetime import datetime

# Define typed dictionaries for API responses
class UserData(TypedDict):
    id: int
    username: str
    email: str
    created_at: str
    is_active: bool

class AdminData(UserData):
    permissions: list[str]
    admin_level: int

class PartialUserData(TypedDict, total=False):
    id: int
    username: str
    email: str

# TypeGuard functions for validation
def is_valid_email(email: str) -> bool:
    """Check if string is a valid email format."""
    return '@' in email and '.' in email.split('@')[1]

def is_user_data(data: dict[str, Any]) -> TypeGuard[UserData]:
    """Validate that dictionary conforms to UserData structure."""
    return (
        isinstance(data.get('id'), int) and
        isinstance(data.get('username'), str) and
        isinstance(data.get('email'), str) and
        is_valid_email(data.get('email', '')) and
        isinstance(data.get('created_at'), str) and
        isinstance(data.get('is_active'), bool)
    )

def is_admin_data(data: dict[str, Any]) -> TypeGuard[AdminData]:
    """Validate that dictionary conforms to AdminData structure."""
    return (
        is_user_data(data) and
        isinstance(data.get('permissions'), list) and
        all(isinstance(p, str) for p in data.get('permissions', [])) and
        isinstance(data.get('admin_level'), int) and
        data.get('admin_level', 0) > 0
    )

def is_partial_user_data(data: dict[str, Any]) -> TypeGuard[PartialUserData]:
    """Validate partial user data for updates."""
    allowed_keys = {'id', 'username', 'email'}
    return (
        all(key in allowed_keys for key in data.keys()) and
        all(
            isinstance(data.get('id'), int) if 'id' in data else True,
            isinstance(data.get('username'), str) if 'username' in data else True,
            isinstance(data.get('email'), str) and is_valid_email(data['email']) if 'email' in data else True,
        )
    )

# Usage in API endpoint
def process_user_update(user_id: int, update_data: dict[str, Any]) -> str:
    """Process user update with proper validation."""
    if not is_partial_user_data(update_data):
        return "Invalid update data format"

    # update_data is now PartialUserData
    if 'email' in update_data:
        print(f"Updating email to: {update_data['email']}")

    if 'username' in update_data:
        print(f"Updating username to: {update_data['username']}")

    return "Update successful"

# Example usage
api_response: dict[str, Any] = {
    "id": 123,
    "username": "alice",
    "email": "alice@example.com",
    "created_at": "2024-01-01T00:00:00Z",
    "is_active": True,
    "permissions": ["read", "write"],
    "admin_level": 2
}

if is_admin_data(api_response):
    # api_response is narrowed to AdminData
    print(f"Admin user {api_response['username']} has permissions: {api_response['permissions']}")
elif is_user_data(api_response):
    # api_response is narrowed to UserData
    print(f"Regular user {api_response['username']}")
else:
    print("Invalid user data")
```

---

## Conclusion

These three patterns—type aliases with PEP 695, the Maybe pattern, and TypeGuard functions—form a powerful trio for writing type-safe, expressive Python code. By adopting these patterns, developers can:

1. **Communicate intent** through semantic type aliases
2. **Eliminate null pointer exceptions** with explicit Maybe handling
3. **Avoid unsafe casting** through principled type narrowing

Together, they enable a more functional, type-driven development style that catches errors at compile time while maintaining Python's expressiveness and readability.

## References

1. PEP 695 – Type Parameter Syntax. Python Enhancement Proposals. <https://peps.python.org/pep-0695/>
2. PEP 647 – User-Defined Type Guards. Python Enhancement Proposals. <https://peps.python.org/pep-0647/>
3. Lipovača, M. (2011). Learn You a Haskell for Great Good! No Starch Press.
4. Petricek, T., & Skeet, J. (2009). Real World Functional Programming. Manning Publications.
5. Returns Documentation. (2024). <https://returns.readthedocs.io/>
