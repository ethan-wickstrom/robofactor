from enum import Enum


class ActionType(str, Enum):
    """Enumeration of available action types for the agent."""
    RUN_COMMAND = "run_command"
    CREATE_FILE = "create_file"
    UPDATE_FILE = "update_file"
    RUN_TESTS = "run_tests"
