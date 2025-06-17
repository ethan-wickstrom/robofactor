from datetime import datetime
from typing import Any

import dspy

from .core.models import Action, ExecutionResult
from .services.files import FileSystemHandler
from .services.shell import CommandExecutor


class ActionExecutor:
    """Base class for executing a specific type of action."""
    def __init__(self, file_handler: FileSystemHandler, command_executor: CommandExecutor):
        self.file_handler = file_handler
        self.command_executor = command_executor

    def execute(self, action: Action, context: dict[str, Any]) -> ExecutionResult:
        raise NotImplementedError

class CommandActionExecutor(ActionExecutor):
    """Executes shell command actions."""
    def execute(self, action: Action, context: dict[str, Any]) -> ExecutionResult:
        return self.command_executor.execute(action.path)

class FileActionExecutor(ActionExecutor):
    """Executes file creation and update actions by generating code."""
    def __init__(self, file_handler: FileSystemHandler, command_executor: CommandExecutor, code_generator: dspy.Module):
        super().__init__(file_handler, command_executor)
        self.code_generator = code_generator

    def _gather_context(self, context_files: list[str]) -> str:
        """Reads and formats content from specified context files."""
        if not context_files:
            return "No context files were provided."

        contexts = [
            f"--- {file} ---\n{result.data if result.success else '[File not found or could not be read]'}"
            for file in context_files
            if (result := self.file_handler.read_file(file))
        ]
        return "\n\n".join(contexts)

    def _resolve_dynamic_path(self, path: str) -> str:
        """Replaces placeholders in paths, like migration timestamps."""
        if 'YYYY_MM_DD_HHMMSS' in path:
            # A more robust implementation could check the last migration file to ensure order.
            # For this refactor, we maintain the original behavior of generating a new timestamp.
            timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
            return path.replace('YYYY_MM_DD_HHMMSS', timestamp)
        return path

    def execute(self, action: Action, context: dict[str, Any]) -> ExecutionResult:
        """Generates code and writes it to the specified file."""
        context_content = self._gather_context(action.context_files)
        resolved_path = self._resolve_dynamic_path(action.path)

        try:
            prediction = self.code_generator(
                intent=context['intent'],
                file_path=resolved_path,
                content_description=action.content_description,
                context=context_content
            )
            return self.file_handler.write_file(resolved_path, prediction.code_content)
        except Exception as e:
            return ExecutionResult(False, f"Code generation failed: {e}")

class TestActionExecutor(ActionExecutor):
    """Executes the test suite."""
    def execute(self, action: Action, context: dict[str, Any]) -> ExecutionResult:
        return self.command_executor.execute("php artisan test")
