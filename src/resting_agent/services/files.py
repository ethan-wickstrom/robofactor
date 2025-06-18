from pathlib import Path

from src.resting_agent.core.models import ExecutionResult


class FileSystemHandler:
    """Encapsulates all file system operations, ensuring they are safe and within the project directory."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path).resolve()

    def _get_full_path(self, relative_path: str) -> Path:
        """Constructs and validates the absolute path for a file."""
        full_path = (self.base_path / relative_path).resolve()
        if not str(full_path).startswith(str(self.base_path)):
            raise ValueError(
                f"Path '{relative_path}' attempts to access outside the project directory."
            )
        return full_path

    def read_file(self, path: str) -> ExecutionResult:
        """Reads content from a file, handling potential errors."""
        try:
            content = self._get_full_path(path).read_text(encoding="utf-8")
            return ExecutionResult(True, f"Successfully read content from {path}.", content)
        except FileNotFoundError:
            return ExecutionResult(False, f"File not found: {path}", "")
        except Exception as e:
            return ExecutionResult(False, f"Error reading file {path}: {e}", "")

    def write_file(self, path: str, content: str) -> ExecutionResult:
        """Writes content to a file, creating parent directories if necessary."""
        try:
            full_path = self._get_full_path(path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            return ExecutionResult(True, f"Successfully wrote to {path}.")
        except Exception as e:
            return ExecutionResult(False, f"Error writing to file {path}: {e}")
