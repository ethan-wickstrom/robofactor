# robofactor

> The robot who refactors: /[^_^]\

A command-line tool that uses an AI agent to analyze, refactor, and evaluate Python code.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Component Breakdown](#component-breakdown)

---

## Installation

This project uses `uv` for package management. It is recommended to work within a virtual environment.

#### Development Setup

For contributing to the project, install the package with all development dependencies.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ethan-wickstrom/robofactor.git
    cd robofactor
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    uv venv
    source .venv/bin/activate
    # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install in development mode using the Makefile:**
    This command will install all main and development dependencies.
    ```bash
    make install-dev
    ```

#### Production Setup

To install the package for regular use without the development tools:

1.  **Clone the repository and navigate into it** (if you haven't already).

2.  **Create and activate a virtual environment.**

3.  **Install using the Makefile:**
    This command will install only the required dependencies for running the tool.
    ```bash
    make install
    ```

---

## Usage

The primary command for this tool is `robofactor`. It analyzes and refactors a given Python file.

#### Basic Command

To run a refactoring analysis on a Python file, provide the path to the file as an argument. This will display the original code, the refactoring plan, and the proposed new code, but it will **not** save any changes.

```bash
robofactor path/to/your_script.py
```

#### Options

You can modify the tool's behavior with the following options:

| Option                       | Alias | Description                                                                       |
| :--------------------------- | :---- | :-------------------------------------------------------------------------------- |
| `--write`                    |       | Write the refactored code back to the original file. **Use with caution.**        |
| `--dog-food`                 |       | A special mode to run the refactorer on its own source code.                      |
| `--optimize`                 |       | Force re-optimization of the underlying DSPy model, even if a cached one exists.  |
| `--task-llm <MODEL>`         |       | Specify the language model for the main refactoring task (e.g., `ollama/llama3`). |
| `--prompt-llm <MODEL>`       |       | Specify the language model for prompt generation during optimization.             |
| `--no-tracing`               |       | Disable MLflow tracing (it is enabled by default).                                |
| `--mlflow-uri <URI>`         |       | Set a custom MLflow tracking server URI.                                          |
| `--mlflow-experiment <NAME>` |       | Set a custom MLflow experiment name.                                              |

#### Examples

**1. Analyze a file without saving changes:**

```bash
robofactor my_app/utils.py
```

**2. Analyze a file and save the refactored code:**

```bash
robofactor my_app/utils.py --write
```

**3. Self-refactor the tool itself and save the changes:**

```bash
robofactor --dog-food --write
```

**4. Force the model to be re-optimized before refactoring:**

```bash
robofactor my_app/utils.py --optimize --write
```

**5. Use a different language model for the refactoring task:**

```bash
robofactor my_app/utils.py --task-llm "ollama/codellama" --write
```

---

## Architecture

This project is a command-line tool, `robofactor`, designed to automatically refactor Python code using an AI agent. The architecture is a modular pipeline orchestrated by the main CLI entry point in `main.py`. The workflow begins when a user specifies a Python file. The `function_extraction` module parses this file to identify and isolate individual functions. These functions are then passed to the core AI agent, defined in `dspy_modules`, which uses a multi-step DSPy program to analyze the code, generate a refactoring plan, and produce the improved code. The newly generated code is then rigorously assessed by the `evaluation` module, which leverages `analysis_utils` to perform static analysis (syntax, linting, complexity) and dynamic analysis (executing functional tests). Finally, the `ui` module presents a comprehensive, color-coded comparison of the original and refactored code, along with the evaluation results, to the user in the console. The entire system is supported by a centralized `config` module for settings and a `functional_types` module for robust error handling.

---

## Component Breakdown

- **`src/robofactor/main.py`**: The main command-line interface (CLI) entry point built with `typer`. It orchestrates the entire refactoring workflow, from loading the AI model and parsing user input to running the evaluation and displaying results.
- **`src/robofactor/dspy_modules.py`**: The core AI engine of the application. It uses the DSPy framework to define a multi-step agent (`CodeRefactor`) that analyzes code, creates a refactoring plan, and implements the changes. It also includes an evaluation module to assess the output.
- **`src/robofactor/evaluation.py`**: Defines the high-level logic and data structures (e.g., `EvaluationResult`) for evaluating the quality of AI-generated code. It orchestrates the various checks performed by `analysis_utils`.
- **`src/robofactor/analysis_utils.py`**: A utility module that provides the concrete functions for code analysis. It performs static checks like syntax validation and linting (`flake8`), and dynamic checks by executing the code against test cases.
- **`src/robofactor/function_extraction.py`**: A static analysis tool that uses Python's `ast` module to parse source code. It extracts detailed information about every function, providing structured input for the refactoring agent.
- **`src/robofactor/ui.py`**: The presentation layer of the application. It uses the `rich` library to display the refactoring process, final code, and evaluation scores in a structured and visually appealing format in the console.
- **`src/robofactor/config.py`**: A centralized configuration module that stores all constants, model names, thresholds, and other settings, making the application easier to manage and modify.
- **`src/robofactor/functional_types.py`**: Provides a foundational, type-safe error-handling system (`Result`, `Ok`, `Err`) inspired by functional programming to improve code robustness and predictability across the project.
