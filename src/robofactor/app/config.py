"""
Configuration for the Robofactor tool.
"""

from pathlib import Path
from typing import Final

# --- File Paths ---
ROOT_DIR: Final[Path] = Path(__file__).parent.parent
OPTIMIZER_FILENAME: Final[Path] = ROOT_DIR / "optimized" / "program.pkl"
OPTIMIZER_METADATA: Final[Path] = ROOT_DIR / "optimized" / "metadata.json"
TRAINING_DATA_FILE: Final[Path] = ROOT_DIR / "training" / "training_data.json"

# --- DSPy Model Configuration ---
DEFAULT_TASK_LLM: Final[str] = "gemini/gemini-2.5-flash-lite-preview-06-17"
DEFAULT_PROMPT_LLM: Final[str] = "gemini/gemini-2.5-pro"
TASK_LLM_MAX_TOKENS: Final[int] = 64000
PROMPT_LLM_MAX_TOKENS: Final[int] = 64000

# --- Refinement Configuration ---
REFINEMENT_THRESHOLD: Final[float] = 0.9
REFINEMENT_COUNT: Final[int] = 3

# --- Analysis & Linting Configuration ---
FLAKE8_COMPLEXITY_CODE: Final[str] = "C901"
FLAKE8_MAX_COMPLEXITY: Final[int] = 10
LINTING_PENALTY_PER_ISSUE: Final[float] = 0.1

# --- UI Configuration ---
RICH_SYNTAX_THEME: Final[str] = "monokai"

# --- MLflow Configuration ---
DEFAULT_MLFLOW_TRACKING_URI: Final[str] = "http://127.0.0.1:5000"
DEFAULT_MLFLOW_EXPERIMENT_NAME: Final[str] = "robofactor"

# --- SSoT Enforcement ---
__all__ = [
    "DEFAULT_MLFLOW_EXPERIMENT_NAME",
    "DEFAULT_MLFLOW_TRACKING_URI",
    "DEFAULT_PROMPT_LLM",
    "DEFAULT_TASK_LLM",
    "FLAKE8_COMPLEXITY_CODE",
    "FLAKE8_MAX_COMPLEXITY",
    "LINTING_PENALTY_PER_ISSUE",
    "OPTIMIZER_FILENAME",
    "OPTIMIZER_METADATA",
    "PROMPT_LLM_MAX_TOKENS",
    "REFINEMENT_COUNT",
    "REFINEMENT_THRESHOLD",
    "RICH_SYNTAX_THEME",
    "ROOT_DIR",
    "TASK_LLM_MAX_TOKENS",
    "TRAINING_DATA_FILE",
]
