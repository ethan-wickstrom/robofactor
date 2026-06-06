from pathlib import Path

OPTIMIZER_PATH: Path = Path("optimized/")

DEFAULT_TASK_LLM: str = "gemini/gemini-flash-latest"
DEFAULT_PROMPT_LLM: str = "openai/gpt-5"
TASK_LLM_MAX_TOKENS: int = 64000
PROMPT_LLM_MAX_TOKENS: int = 64000

REFINEMENT_THRESHOLD: float = 0.9
REFINEMENT_COUNT: int = 3

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

DEFAULT_MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
DEFAULT_MLFLOW_EXPERIMENT_NAME: str = "robofactor"
