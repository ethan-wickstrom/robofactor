from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from returns.result import Failure, Result, Success
from rope.base import libutils
from rope.base.project import Project

from robofactor.checks import BehaviorTest, CheckReport, check_refactor


@dataclass(frozen=True, kw_only=True)
class AppliedChange:
    changed_file: Path
    check: CheckReport


@dataclass(frozen=True, kw_only=True)
class ApplyFailed:
    changed_file: Path
    reason: str
    message: str
    check: CheckReport | None = None


@dataclass(frozen=True, kw_only=True)
class RopeFile:
    project_root: Path
    file_path: Path
    resource_path: str
    module_name: str


def apply_change(
    source_path: Path,
    proposed_code: str,
    tests: tuple[BehaviorTest, ...],
) -> AppliedChange | ApplyFailed:
    source_code = source_path.read_text(encoding="utf-8")
    check_result = check_refactor(source_code, proposed_code, tests)
    match check_result:
        case Failure(message):
            return ApplyFailed(
                changed_file=source_path,
                reason="check_error",
                message=message,
            )
        case Success(check) if not check.passed:
            return ApplyFailed(
                changed_file=source_path,
                reason="checks_failed",
                message="Refactored code did not preserve behavior.",
                check=check,
            )
        case Success(check):
            source_path.write_text(proposed_code, encoding="utf-8")
            return AppliedChange(changed_file=source_path, check=check)
    raise AssertionError("Unhandled refactor check result.")


def inspect_rope_file(project_root: Path, file_path: Path) -> Result[RopeFile, str]:
    root = project_root.resolve()
    path = file_path.resolve()
    project = Project(str(root), ropefolder=None)
    try:
        resource = libutils.path_to_resource(project, str(path))
        project.validate(resource)
        if not libutils.is_python_file(project, resource):
            return Failure(f"not a Python file: {path}")
        libutils.analyze_module(project, resource)
        return Success(
            RopeFile(
                project_root=root,
                file_path=path,
                resource_path=resource.path,
                module_name=libutils.modname(resource),
            )
        )
    except Exception as error:
        # RopeProject boundary: translate library/resource failures into inspect failures.
        return Failure(str(error))
    finally:
        project.close()
