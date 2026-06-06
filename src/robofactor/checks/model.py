from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never

from robofactor.json_value import Json

type ParameterKind = Literal[
    "positional_only",
    "positional_or_keyword",
    "var_positional",
    "keyword_only",
    "var_keyword",
]
type CheckName = Literal[
    "explicit_behavior",
    "source_comparison",
    "generated_comparison",
    "suppression_comment",
]


@dataclass(frozen=True, kw_only=True)
class FunctionParameter:
    kind: ParameterKind
    name: str
    default: str | None


@dataclass(frozen=True, kw_only=True)
class FunctionSignature:
    name: str
    parameters: tuple[FunctionParameter, ...]

    def render(self) -> str:
        positional_only = tuple(
            parameter for parameter in self.parameters if parameter.kind == "positional_only"
        )
        positional_or_keyword = tuple(
            parameter for parameter in self.parameters if parameter.kind == "positional_or_keyword"
        )
        var_positional = tuple(
            parameter for parameter in self.parameters if parameter.kind == "var_positional"
        )
        keyword_only = tuple(
            parameter for parameter in self.parameters if parameter.kind == "keyword_only"
        )
        var_keyword = tuple(
            parameter for parameter in self.parameters if parameter.kind == "var_keyword"
        )
        parameters = (
            *tuple(_parameter_text(parameter) for parameter in positional_only),
            *(("/",) if positional_only else ()),
            *tuple(_parameter_text(parameter) for parameter in positional_or_keyword),
            *tuple(_parameter_text(parameter) for parameter in var_positional),
            *(("*",) if keyword_only and not var_positional else ()),
            *tuple(_parameter_text(parameter) for parameter in keyword_only),
            *tuple(_parameter_text(parameter) for parameter in var_keyword),
        )
        return f"def {self.name}({', '.join(parameters)})"


@dataclass(frozen=True, kw_only=True)
class BehaviorTest:
    case_id: str
    args: tuple[Json, ...]
    kwargs: dict[str, Json]
    expected_output: Json
    labels: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Returned:
    value: Json


@dataclass(frozen=True, kw_only=True)
class Raised:
    reason: str


type CallOutcome = Returned | Raised


@dataclass(frozen=True, kw_only=True)
class CheckFailure:
    check: CheckName
    case_id: str
    message: str
    source: CallOutcome | None = None
    refactored: CallOutcome | None = None


@dataclass(frozen=True, kw_only=True)
class CheckSummary:
    passed: int
    failed: int
    total: int

    @property
    def pass_rate(self) -> float:
        """Passed checks divided by total checks, with no checks treated as no failures."""
        return self.passed / self.total if self.total > 0 else 1.0


@dataclass(frozen=True, kw_only=True)
class QualityReport:
    ruff_passed: bool
    ty_passed: bool
    typing_score: float
    docstring_score: float
    complexity_score: float
    ruff_issues: tuple[str, ...]
    ty_issues: tuple[str, ...]
    complexity_issues: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class CheckReport:
    passed: bool
    function_name: str
    signature: FunctionSignature
    explicit_behavior: CheckSummary
    source_comparisons: CheckSummary
    generated_comparisons: CheckSummary
    quality: QualityReport
    failures: tuple[CheckFailure, ...]
    warnings: tuple[str, ...]


def _parameter_text(parameter: FunctionParameter) -> str:
    match parameter.kind:
        case "positional_only" | "positional_or_keyword" | "keyword_only":
            if parameter.default is None:
                return parameter.name
            return f"{parameter.name}={parameter.default}"
        case "var_positional":
            return f"*{parameter.name}"
        case "var_keyword":
            return f"**{parameter.name}"
        case _:
            assert_never(parameter.kind)
