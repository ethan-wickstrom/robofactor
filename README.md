# Robofactor

> The robot who refactors: /[^_^]\

[![PyPI Version](https://badge.fury.io/py/robofactor.svg)](https://pypi.org/project/robofactor/)
[![Build Status](https://github.com/ethan-wickstrom/robofactor/actions/workflows/publish.yml/badge.svg)](https://github.com/ethan-wickstrom/robofactor/actions)
[![License](https://img.shields.io/github/license/ethan-wickstrom/robofactor)](https://github.com/ethan-wickstrom/robofactor/blob/main/LICENSE)

**Robofactor** is a DSPy-powered tool designed to analyze, plan, and refactor Python code. It leverages large language models to understand your code and suggest improvements, which are then programmatically verified for correctness and quality before being applied.

Robofactor is a command-line tool powered by the [DSPy](https://github.com/stanford-futuredata/dspy) framework, designed to automatically analyze, refactor, and evaluate Python code. By leveraging the structured prompting capabilities of DSPy, it can understand your code, propose improvements, and verify the results. The ultimate goal is to serve as an AI-powered assistant that helps improve the quality, readability, and maintainability of your Python projects.

## ✨ Features

- 🤖 **AI-Powered Refactoring**: Leverages `dspy-ai` to analyze, plan, and refactor your Python code, improving readability, style, and structure.
- 🐶 **Self-Refactoring Mode**: Use the `--dog-food` flag to turn Robofactor on itself, continuously improving its own codebase.
- 📝 **In-Place File Writing**: Automatically write the improved code back to the source file with the `--write` option.
- 🔧 **Configurable AI Models**: Easily switch between different LLMs for refactoring tasks (`--task-llm`) and prompt generation (`--prompt-llm`).
- 📊 **Experiment Tracing**: Integrates seamlessly with MLflow to trace refactoring runs, monitor performance, and compare results.
- 🧠 **DSPy Model Optimization**: Force a re-optimization of the underlying DSPy model with the `--optimize` flag to fine-tune the refactoring logic.

## 🚀 Installation

Follow these steps to get Robofactor set up on your local machine.

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager. If you don't have it, you can install it via pip:

  ```bash
  pip install uv
  ```

### Step-by-Step Guide

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ethan-wickstrom/robofactor.git
   cd robofactor
   ```

2. **Install dependencies:**

   This project uses `uv` to manage dependencies and virtual environments.

   - For a **standard installation** (to use the tool):

     ```bash
     uv sync --no-dev
     ```

   - For a **development installation** (includes testing and linting tools):

     ```bash
     uv sync --all-groups
     ```

## 🚀 Usage

To refactor a Python file, run Robofactor from your command line and provide the path to the file:

```bash
robofactor path/to/your/file.py
```

To have Robofactor refactor its own source code (a process often called "dogfooding"), use the `--dog-food` flag:

```bash
robofactor --dog-food
```

By default, Robofactor prints the refactored code to the console without modifying the original file. To write the changes back to the source file, include the `--write` flag:

```bash
robofactor path/to/your/file.py --write
```

For a complete list of all available commands and options, see the help text below.

<details>
<summary>Full CLI Options</summary>

```bash
 Usage: robofactor [OPTIONS] [PATH]

 A DSPy-powered tool to analyze, plan, and refactor Python code.

╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   path      [PATH]  Path to the Python file to refactor. [default: None]                                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --dog-food                                    Self-refactor the script you are running.                                                     │
│ --write                                       Write the refactored code back to the file.                                                   │
│ --optimize                                    Force re-optimization of the DSPy model.                                                      │
│ --task-llm                              TEXT  Model for the main refactoring task. [default: gemini/gemini-2.5-flash-lite-preview-06-17]    │
│ --prompt-llm                            TEXT  Model for generating prompts during optimization. [default: gemini/gemini-2.5-pro]            │
│ --tracing               --no-tracing          Enable MLflow tracing. [default: tracing]                                                     │
│ --mlflow-uri                            TEXT  MLflow tracking server URI. [default: http://127.0.0.1:5000]                                  │
│ --mlflow-experiment                     TEXT  MLflow experiment name. [default: robofactor]                                                 │
│ --install-completion                          Install completion for the current shell.                                                     │
│ --show-completion                             Show completion for the current shell, to copy it or customize the installation.              │
│ --help                                        Show this message and exit.                                                                   │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

</details>

## ⚙️ How It Works

Robofactor operates through a systematic, three-stage pipeline to ensure that code is not only refactored but also improved in a safe and verifiable way.

### 1. Code Parsing & Analysis

The process begins by deeply understanding the target Python file. Instead of treating the code as plain text, Robofactor uses Python's built-in `ast` (Abstract Syntax Tree) module to parse the source code into a structured tree.

- **Module:** `src/robofactor/function_extraction.py`
- **Process:** This module traverses the AST to identify individual functions, their signatures (parameters, decorators, return types), docstrings, and body content. This granular understanding allows the AI to focus its efforts on a specific, well-defined piece of code.

### 2. AI-Powered Refactoring with DSPy

Once a function is isolated, it's handed over to the AI core for refactoring. Robofactor leverages `dspy-ai`, a framework for programming with language models, to create a robust and optimizable refactoring program.

- **Module:** `src/robofactor/dspy_modules.py`
- **Process:** The parsed function code is fed into a compiled DSPy program. This program instructs a Large Language Model (LLM) to rewrite the code with specific goals: improving readability, adding missing type hints, generating comprehensive docstrings, and adhering to Python best practices.

### 3. Rigorous Evaluation

The AI's suggested refactoring is never trusted blindly. Before any changes are accepted, the new code is subjected to a strict, multi-faceted evaluation pipeline. This pipeline is built using a railway-oriented approach with the `returns` library, ensuring that if any single check fails, the entire process halts safely.

- **Module:** `src/robofactor/evaluation.py`
- **Process:** The evaluation consists of several automated checks:
  1. **Syntax Check**: The refactored code is parsed again to ensure it is valid Python syntax.
  2. **Code Quality Analysis**: The code is linted using `flake8` to check for style guide violations, logical errors, and code smells.
  3. **Functional Correctness**: The original function's test cases are executed against the refactored code in a sandboxed environment. This critical step verifies that the refactoring did not alter the function's behavior or introduce regressions.

Only if the refactored code passes all three checks is the process considered a success.

## 🔧 Configuration

You can configure Robofactor's behavior using command-line options, particularly for setting the language models and connecting to an MLflow instance for experiment tracing.

### Language Models (LLMs)

Robofactor uses two distinct language models: one for the primary refactoring task and another, typically more powerful, model for the one-time optimization process that generates effective prompts.

- `--task-llm`: Specifies the model used for the core refactoring task.
  - **Default**: `gemini/gemini-2.5-flash-lite-preview-06-17`
- `--prompt-llm`: Specifies the model used during the DSPy optimization step (`--optimize`) to generate high-quality prompts.
  - **Default**: `gemini/gemini-2.5-pro`

**Example:**

```bash
# Use OpenAI models for both tasks
robofactor --task-llm "openai/gpt-4o-mini" --prompt-llm "openai/gpt-4o" path/to/your/file.py
```

### MLflow Tracing

To monitor and debug the DSPy program's execution, Robofactor integrates with MLflow. Tracing is enabled by default.

- `--no-tracing`: Use this flag to disable MLflow integration entirely.
- `--mlflow-uri`: Sets the URI for your MLflow tracking server.
  - **Default**: `http://127.0.0.1:5000`
- `--mlflow-experiment`: Specifies the name of the MLflow experiment where runs will be logged.
  - **Default**: `robofactor`

**Example:**

```bash
# Run with a custom MLflow server and experiment name
robofactor --mlflow-uri "http://your-mlflow-server:5001" --mlflow-experiment "refactor-audits" path/to/your/file.py

# Run without any MLflow tracing
robofactor --no-tracing path/to/your/file.py
```

## 🛠️ Technology Stack

Robofactor is built on a modern stack of Python libraries, leveraging the power of LLMs, robust CLI frameworks, and functional programming principles.

- **[DSPy-AI](https://github.com/stanford-futuredata/dspy)**: The core AI programming model used to create, optimize, and execute the refactoring logic with language models.
- **[Typer](https://typer.tiangolo.com/)**: Powers the clean, user-friendly command-line interface.
- **[Rich](https://github.com/Textualize/rich)**: Provides beautiful and informative terminal output, including progress spinners, tables, and syntax-highlighted code.
- **[MLflow](https://mlflow.org/)**: Tracks and visualizes the refactoring process as experiments, enabling detailed analysis of the LLM's behavior.
- **[Flake8](https://flake8.pycqa.org/en/latest/) & `ast`**: Used for static analysis of Python code, checking for syntax errors, code quality issues, and extracting function metadata.
- **[Returns](https://returns.readthedocs.io/en/latest/)**: Implements a robust, railway-oriented programming pipeline for evaluating refactored code, ensuring each step is handled safely and declaratively.

## 🧑‍💻 Development

Contributions are welcome! To set up the development environment, first clone the repository. This project uses `uv` for dependency management. Install all dependencies, including development tools, with:

```bash
uv sync --all-groups
```

### Available Commands

The project includes several helper commands to streamline development, which can be executed with `uv run <command>`:

- **`lint`**: Run linting checks.
- **`format`**: Format code with black and isort.
- **`type-check`**: Run mypy type checking.
- **`test`**: Run all tests.
- **`test-unit`**: Run unit tests only.
- **`test-integration`**: Run integration tests only.
- **`test-coverage`**: Run tests and generate an HTML coverage report.
- **`check`**: Run all checks (lint, type-check, test).
- **`readme`**: Generate README.md using DSPy.

## 📜 License

This project is licensed under the Apache Version 2.0 License. See the `LICENSE` file for more details.
