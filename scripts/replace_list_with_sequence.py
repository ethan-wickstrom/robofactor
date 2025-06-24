"""
Functional code transformation script for replacing list/List with Sequence.

Implements functional programming principles with clear service boundaries,
error handling as values, and immutable data transformations.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Sequence as SeqType
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from returns.result import Result, Success, Failure, safe


# ============================================================================
# Domain Models
# ============================================================================

@dataclass(frozen=True)
class FileContent:
    """Immutable representation of file content."""
    path: Path
    content: str
    encoding: str = "utf-8"


@dataclass(frozen=True)
class TransformationRule:
    """Immutable rule for text transformation."""
    pattern: re.Pattern[str]
    replacement: str
    description: str


@dataclass(frozen=True)
class ImportStatement:
    """Immutable representation of an import statement."""
    module: str
    imports: tuple[str, ...]
    line: str


@dataclass(frozen=True)
class TransformationResult:
    """Result of applying transformations to file content."""
    original: FileContent
    transformed: FileContent
    rules_applied: tuple[TransformationRule, ...]
    import_added: ImportStatement | None = None


# ============================================================================
# Service Interfaces
# ============================================================================

class FileOperations(Protocol):
    """Interface for file I/O operations."""
    
    def read_file(self, path: Path) -> Result[FileContent, Exception]:
        """Read file content safely."""
        ...
    
    def write_file(self, content: FileContent) -> Result[Path, Exception]:
        """Write content to file safely."""
        ...
    
    def create_backup(self, path: Path) -> Result[Path, Exception]:
        """Create backup of file."""
        ...


class TextTransformer(Protocol):
    """Interface for text transformation operations."""
    
    def apply_transformations(
        self, 
        content: FileContent, 
        rules: SeqType[TransformationRule]
    ) -> Result[TransformationResult, str]:
        """Apply transformation rules to content."""
        ...
    
    def ensure_import(
        self, 
        content: FileContent, 
        import_statement: ImportStatement
    ) -> Result[FileContent, str]:
        """Ensure import statement exists in content."""
        ...


class DirectoryProcessor(Protocol):
    """Interface for directory traversal operations."""
    
    def find_python_files(self, directory: Path) -> Result[tuple[Path, ...], str]:
        """Find all Python files in directory recursively."""
        ...


# ============================================================================
# Implementation Services
# ============================================================================

class SafeFileOperations:
    """Safe file operations implementation using functional patterns."""
    
    @safe
    def read_file(self, path: Path) -> FileContent:
        """Read file content with automatic error handling."""
        content = path.read_text(encoding="utf-8")
        return FileContent(path=path, content=content)
    
    @safe
    def write_file(self, content: FileContent) -> Path:
        """Write content to file with automatic error handling."""
        _ = content.path.write_text(content.content, encoding=content.encoding)
        return content.path
    
    @safe
    def create_backup(self, path: Path) -> Path:
        """Create backup file with automatic error handling."""
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)
        return backup_path


class FunctionalTextTransformer:
    """Functional text transformation implementation."""
    
    def apply_transformations(
        self, 
        content: FileContent, 
        rules: SeqType[TransformationRule]
    ) -> Result[TransformationResult, str]:
        """Apply transformation rules functionally."""
        def _apply_rule(text: str, rule: TransformationRule) -> str:
            return rule.pattern.sub(rule.replacement, text)
        
        try:
            # Apply transformations immutably
            transformed_content = content.content
            applied_rules: list[TransformationRule] = []
            
            for rule in rules:
                original_content = transformed_content
                transformed_content = _apply_rule(transformed_content, rule)
                
                # Track which rules were actually applied
                if original_content != transformed_content:
                    applied_rules.append(rule)
            
            transformed_file = replace(content, content=transformed_content)
            
            return Success(TransformationResult(
                original=content,
                transformed=transformed_file,
                rules_applied=tuple(applied_rules)
            ))
            
        except Exception as e:
            return Failure(f"Transformation failed: {e}")
    
    def ensure_import(
        self, 
        content: FileContent, 
        import_statement: ImportStatement
    ) -> Result[FileContent, str]:
        """Ensure import statement exists, adding if necessary."""
        try:
            lines = content.content.splitlines()
            
            # Check if import already exists
            has_import = any(
                import_statement.module in line and 
                all(imp in line for imp in import_statement.imports)
                for line in lines
            )
            
            if has_import:
                return Success(content)
            
            # Add import at the top after any existing imports
            import_line = import_statement.line
            
            # Find insertion point (after last import or at beginning)
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    insert_index = i + 1
                elif line.strip() and not line.startswith('#'):
                    break
            
            new_lines = lines[:insert_index] + [import_line] + lines[insert_index:]
            new_content = '\n'.join(new_lines)
            
            return Success(replace(content, content=new_content))
            
        except Exception as e:
            return Failure(f"Import addition failed: {e}")


class RecursiveDirectoryProcessor:
    """Directory processing implementation."""
    
    def find_python_files(self, directory: Path) -> Result[tuple[Path, ...], str]:
        """Find Python files recursively with error handling."""
        try:
            if not directory.exists():
                return Failure(f"Directory does not exist: {directory}")
            
            if not directory.is_dir():
                return Failure(f"Path is not a directory: {directory}")
            
            python_files = tuple(
                path for path in directory.rglob("*.py") 
                if path.is_file()
            )
            
            return Success(python_files)
            
        except Exception as e:
            return Failure(f"Directory traversal failed: {e}")


# ============================================================================
# Configuration and Rules
# ============================================================================

# Transformation rules for list -> Sequence replacement
LIST_TO_SEQUENCE_RULES: tuple[TransformationRule, ...] = (
    TransformationRule(
        pattern=re.compile(r'\blist\b'),
        replacement="Sequence",
        description="Replace 'list' with 'Sequence'"
    ),
    TransformationRule(
        pattern=re.compile(r'\bList\b'),
        replacement="Sequence",
        description="Replace 'List' with 'Sequence'"
    ),
)

# Import statement to add
SEQUENCE_IMPORT = ImportStatement(
    module="collections.abc",
    imports=("Sequence",),
    line="from collections.abc import Sequence"
)


# ============================================================================
# Application Service
# ============================================================================

@dataclass(frozen=True)
class CodeTransformationService:
    """Main application service with dependency injection."""
    
    file_ops: FileOperations
    text_transformer: TextTransformer
    directory_processor: DirectoryProcessor
    
    def transform_file(self, file_path: Path) -> Result[TransformationResult, str]:
        """Transform a single file with full error handling."""
        def _process_content(content: FileContent) -> Result[TransformationResult, str]:
            # Apply transformations
            transform_result = self.text_transformer.apply_transformations(
                content, LIST_TO_SEQUENCE_RULES
            )
            
            match transform_result:
                case Success(result):
                    # Ensure import if transformations were applied
                    if result.rules_applied:
                        import_result = self.text_transformer.ensure_import(
                            result.transformed, SEQUENCE_IMPORT
                        )
                        match import_result:
                            case Success(updated_content):
                                # Create backup and write
                                backup_result = self.file_ops.create_backup(content.path)
                                match backup_result:
                                    case Success(_):
                                        write_result = self.file_ops.write_file(updated_content)
                                        match write_result:
                                            case Success(_):
                                                return Success(replace(result, transformed=updated_content))
                                            case Failure(error):
                                                return Failure(f"Write failed: {error}")
                                            case _:
                                                return Failure("Unknown write error")
                                    case Failure(error):
                                        return Failure(f"Backup failed: {error}")
                                    case _:
                                        return Failure("Unknown backup error")
                            case Failure(error):
                                return Failure(f"Import failed: {error}")
                            case _:
                                return Failure("Unknown import error")
                    else:
                        # No changes needed, return original
                        return Success(result)
                case Failure(error):
                    return Failure(error)
                case _:
                    return Failure("Unknown transformation error")
        
        # Read file and process
        read_result = self.file_ops.read_file(file_path)
        match read_result:
            case Success(content):
                return _process_content(content)
            case Failure(error):
                return Failure(f"Read failed: {error}")
            case _:
                return Failure("Unknown read error")
    
    def transform_directory(self, directory: Path) -> Result[tuple[TransformationResult, ...], str]:
        """Transform all Python files in directory."""
        def _transform_files(files: tuple[Path, ...]) -> Result[tuple[TransformationResult, ...], str]:
            results: list[TransformationResult] = []
            errors: list[str] = []
            
            for file_path in files:
                result = self.transform_file(file_path)
                match result:
                    case Success(transformation_result):
                        results.append(transformation_result)
                    case Failure(error):
                        errors.append(f"Failed to transform {file_path}: {error}")
                    case _:
                        errors.append(f"Unknown error for {file_path}")
            
            if errors:
                return Failure(f"Errors occurred: {'; '.join(errors)}")
            
            return Success(tuple(results))
        
        files_result = self.directory_processor.find_python_files(directory)
        match files_result:
            case Success(files):
                return _transform_files(files)
            case Failure(error):
                return Failure(error)
            case _:
                return Failure("Unknown directory processing error")


# ============================================================================
# Application Entry Point
# ============================================================================

def create_application() -> CodeTransformationService:
    """Factory function for creating the application with dependencies."""
    return CodeTransformationService(
        file_ops=SafeFileOperations(),
        text_transformer=FunctionalTextTransformer(),
        directory_processor=RecursiveDirectoryProcessor()
    )


def main() -> None:
    """Main application entry point."""
    app = create_application()
    target_directory = Path("src/robofactor")
    
    match app.transform_directory(target_directory):
        case Success(results):
            print(f"✅ Successfully transformed {len(results)} files in {target_directory}")
            for result in results:
                if result.rules_applied:
                    print(f"  📝 Transformed: {result.transformed.path}")
        case Failure(error):
            print(f"❌ Transformation failed: {error}")
            _ = exit(1)  # Explicitly ignore return value
        case _:
            print("❌ Unknown error occurred")
            _ = exit(1)


if __name__ == "__main__":
    main()
