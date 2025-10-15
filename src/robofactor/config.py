from pathlib import Path

# --- File Paths ---
OPTIMIZER_PATH: Path = Path("optimized/")

# --- DSPy Model Configuration ---
DEFAULT_TASK_LLM: str = "gemini/gemini-flash-latest"
DEFAULT_PROMPT_LLM: str = "gemini/gemini-2.5-pro"
TASK_LLM_MAX_TOKENS: int = 64000
PROMPT_LLM_MAX_TOKENS: int = 64000

# --- Refinement Configuration ---
REFINEMENT_THRESHOLD: float = 0.9
REFINEMENT_COUNT: int = 3

# --- Analysis Configuration ---
FLAKE8_COMPLEXITY_CODE: str = "C901"
FLAKE8_MAX_COMPLEXITY: int = 10
LINTING_PENALTY_PER_ISSUE: float = 0.1

# --- UI Configuration ---
RICH_SYNTAX_THEME: str = "monokai"
UI_COLORS: dict[str, str] = {
    "section": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "accent": "magenta",
    "info": "blue",
}
UI_TRUNCATE_LIMIT: int = 20

# --- MLflow Configuration ---
DEFAULT_MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
DEFAULT_MLFLOW_EXPERIMENT_NAME: str = "robofactor"
