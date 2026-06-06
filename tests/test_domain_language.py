from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_FILES = {
    Path("src/robofactor/CONTEXT.md"),
    Path("docs/agents/domain-language.md"),
    Path("tests/test_domain_language.py"),
}

FORBIDDEN_LANGUAGE = {
    "ApplyDecision": "Use Apply for the action, AppliedChange for success, and ApplyFailed for failure.",
    "ApplyResult": "Use AppliedChange or ApplyFailed.",
    "ReviewNote": "Use Review only for engineer-facing opportunity summaries.",
    "QualityReview": "Use QualityAssessment.",
    "QualityVerdict": "Use QualityAssessment.",
    "FinalEvaluation": "Use FinalAssessment.",
    "EvaluationRecommendation": "Use AssessmentRecommendation.",
    "final_evaluation": "Use final_assessment.",
    "TestCase": "Use BehaviorTest.",
    "test_cases": "Use behavior_tests.",
    "PublicFunctionContract": "Use FunctionSignature.",
    "adversarial probe": "Use GeneratedComparison.",
    "differential probe": "Use ComparisonCheck.",
    "quality review": "Use QualityAssessment.",
    "quality verdict": "Use QualityAssessment.",
    "verdict": "Use Assessment language.",
}

CHECKED_GLOBS = (
    "AGENTS.md",
    "README.md",
    "Makefile",
    "pyproject.toml",
    "tach.toml",
    ".github/**/*.yml",
    ".github/**/*.yaml",
    "docs/**/*.md",
    "scripts/**/*.py",
    "src/robofactor/**/*.py",
    "tests/**/*.py",
)


def test_working_model_uses_domain_language() -> None:
    occurrences = tuple(_forbidden_occurrences(_checked_files()))

    assert not occurrences, "\n".join(occurrences)


def _checked_files() -> tuple[Path, ...]:
    paths = {
        path
        for pattern in CHECKED_GLOBS
        for path in PROJECT_ROOT.glob(pattern)
        if path.is_file() and path.relative_to(PROJECT_ROOT) not in REFERENCE_FILES
    }
    return tuple(sorted(paths))


def _forbidden_occurrences(paths: Iterable[Path]) -> Iterable[str]:
    for path in paths:
        relative_path = path.relative_to(PROJECT_ROOT)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for term, replacement in FORBIDDEN_LANGUAGE.items():
                if _contains_forbidden_term(line, term):
                    yield f"{relative_path}:{line_number}: {term!r} is forbidden. {replacement}"


def _contains_forbidden_term(line: str, term: str) -> bool:
    if _accepts_compacted_match(term):
        return _compact(term) in _compact(line)

    return (
        re.search(
            rf"(?<![a-z0-9]){re.escape(term.casefold())}s?(?![a-z0-9])",
            line.casefold(),
        )
        is not None
    )


def _accepts_compacted_match(term: str) -> bool:
    return any(character in " _-" for character in term)


def _compact(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())
