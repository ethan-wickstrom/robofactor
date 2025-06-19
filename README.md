# robofactor

The robot who refactors: /[^_^]\

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
  - [Key Components](#key-components)

## Installation

You can install `robofactor` from PyPI or from source for development.

### From PyPI (Recommended)

The easiest way to install the tool is using `pip`:

```bash
uv add robofactor
```

### From Source (for Development)

For contributing to the project or using the latest unreleased version, you can install from the source repository. This project uses `uv` for dependency management.

1.  **Install `uv`:**

    ```bash
    pip install uv
    ```

2.  **Clone the repository:**

    ```bash
    git clone https://github.com/ethan-wickstrom/robofactor.git
    cd robofactor
    ```

3.  **Install dependencies:**
    The Makefile provides convenient targets for installation using `uv`. This will create a virtual environment and install the required packages.

    - To set up a full development environment, including testing and linting tools:
      ```bash
      make install-dev
      ```
    - To install only the runtime dependencies for a production-like setup:
      ```bash
      make install
      ```

## Usage

`robofactor` is a command-line tool that uses DSPy to analyze, plan, and refactor Python code.

### Basic Refactoring

To analyze a Python file and see the suggested refactoring, provide the path to the file. The refactored code will be printed to the console.

```bash
robofactor path/to/your_file.py
```

### Applying Changes

By default, `robofactor` only displays the suggested changes. To write the refactored code back to the original file, use the `--write` flag.

**Warning:** This will overwrite the contents of your file. It is strongly recommended to use version control.

```bash
robofactor --write path/to/your_file.py
```

### Configuring Language Models

You can specify which large language models (LLMs) to use for different parts of the process.

- **Change the main refactoring model:**

  ```bash
  robofactor --task-llm "openai/gpt-4o" path/to/your_file.py
  ```

- **Change the model for prompt optimization:**
  When using the `--optimize` flag, you can specify a different model for generating prompts.
  ```bash
  robofactor --optimize --prompt-llm "anthropic/claude-3-opus" path/to/your_file.py
  ```

### Tracing with MLflow

`robofactor` supports MLflow for tracing program executions, which is useful for debugging and monitoring the underlying AI system.

1.  **Start an MLflow tracking server:**

    ```bash
    mlflow ui
    ```

    By default, this starts the server at `http://127.0.0.1:5000`.

2.  **Run `robofactor` with tracing:**
    Tracing is enabled by default and will connect to the local MLflow server.

    ```bash
    robofactor path/to/your_file.py
    ```

    You can also specify a custom MLflow server URI and experiment name:

    ```bash
    robofactor --mlflow-uri "http://your-mlflow-server:5000" --mlflow-experiment "my-refactor-tests" path/to/your_file.py
    ```

    To disable tracing, use the `--no-tracing` flag:

    ```bash
    robofactor --no-tracing path/to/your_file.py
    ```

### Getting Help

To see all available options and their descriptions, use the `--help` flag.

```bash
robofactor --help
```

## Architecture

This project is a command-line tool for automatically refactoring Python code using a large language model. The architecture is centered around a main CLI entry point (`main.py`) that orchestrates the process. The core logic resides in a DSPy-based module (`dspy_modules.py`) which performs a multi-step refactoring: analyzing the code, creating a plan, and implementing the changes. This analysis is supported by detailed static analysis utilities that parse the code's Abstract Syntax Tree (`function_extraction.py`, `analysis_utils.py`). Once refactored, the new code is passed through a comprehensive evaluation pipeline (`evaluation.py`) that checks for syntax validity, code quality, and functional correctness. Finally, the results of the process and the evaluation are presented to the user in a formatted console output via a dedicated UI module (`ui.py`).

### Key Components

- `main.py`: The main command-line interface (CLI) entry point built with Typer. It orchestrates the entire refactoring workflow from file input to final output.
- `dspy_modules.py`: The core AI-driven refactoring engine. It uses the DSPy framework to define a multi-step process of code analysis, planning, and implementation.
- `evaluation.py`: A comprehensive evaluation pipeline that assesses the refactored code's quality through syntax validation, quality checks, and functional correctness testing.
- `analysis_utils.py`: A toolkit for static and dynamic code analysis, providing utilities for syntax validation, quality scoring (flake8, AST), and sandboxed test execution.
- `function_extraction.py`: A specialized module that uses Abstract Syntax Tree (AST) parsing to extract detailed metadata about functions from Python source code.
- `ui.py`: The user interface module responsible for presenting formatted, step-by-step process updates and final evaluation results to the console using the `rich` library.
- `config.py / functional_types.py`: Placeholder modules intended for centralized configuration settings and custom type definitions for the application.
