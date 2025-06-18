import subprocess
from pathlib import Path

from src.resting_agent.core.models import ExecutionResult


class CommandExecutor:
    """Handles the execution of shell commands within the project's working directory."""

    def __init__(self, working_directory: str):
        self.cwd = Path(working_directory).resolve()

    def execute(self, command: str, timeout: int = 60) -> ExecutionResult:
        """Executes a shell command and captures its output."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.cwd,
                timeout=timeout,
                check=False,
            )
            if result.returncode == 0:
                return ExecutionResult(
                    True, "Command executed successfully.", result.stdout.strip()
                )
            return ExecutionResult(
                False, f"Command failed with exit code {result.returncode}.", result.stderr.strip()
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(False, f"Command timed out after {timeout} seconds.")
        except Exception as e:
            return ExecutionResult(
                False, f"An unexpected error occurred during command execution: {e}"
            )
