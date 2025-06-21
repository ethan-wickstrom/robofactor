# Robofactor

The robot who refactors: /[^_^]\

[![PyPI version](https://img.shields.io/pypi/v/robofactor)](https://pypi.org/project/robofactor)
[![Build Status](https://github.com/ethan-wickstrom/robofactor/actions/workflows/publish.yml/badge.svg)](https://github.com/ethan-wickstrom/robofactor/actions)
[![License](https://img.shields.io/pypi/l/robofactor)](https://github.com/ethan-wickstrom/robofactor)
[![Python versions](https://img.shields.io/pypi/pyversions/robofactor)](https://pypi.org/project/robofactor)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Development](#development)
- [Contributing](#contributing)

---

## Overview

Robofactor is a DSPy-powered tool to analyze, plan, and refactor Python code. It leverages a modern stack to programmatically assess and improve code quality through a structured, multi-step process.

The core technologies driving Robofactor include:

*   **DSPy (`dspy-ai`):** The project is built on the DSPy framework, which provides a structured way to program with language models. It is used to generate refactoring plans and implement code changes.
*   **Railway-Oriented Pipelines (`returns`):** The evaluation process is constructed as a robust pipeline using the `returns` library. This allows for a series of checks (syntax, quality, functional correctness) where any failure gracefully halts the process and returns a descriptive error.
*   **Code Quality Analysis (`flake8`):** Code quality is programmatically measured using `flake8`, providing objective metrics to evaluate the effectiveness of the refactoring.
*   **Rich CLI (`rich`):** All terminal output, from the refactoring process to the final evaluation results, is formatted for clarity and readability using the `rich` library.

## Key Features

*   **AI-Powered Refactoring**: Leverages a `CodeRefactor` module built with DSPy (`dspy_modules.py`) to intelligently analyze and generate refactoring suggestions for Python code snippets.
*   **Comprehensive Evaluation Pipeline**: Ensures the quality and correctness of refactored code through a multi-stage process (`evaluation.py`). This pipeline includes syntax validation (`check_syntax`), quality scoring using `flake8` and AST analysis (`check_code_quality`), and functional correctness checks against provided test cases (`check_functional_correctness`).
*   **Advanced Code Analysis**: Performs deep static analysis of Python code by parsing it into an Abstract Syntax Tree (AST). The `function_extraction.py` module is dedicated to extracting detailed information about functions, decorators, and parameters directly from the source code structure.
*   **DSPy Model Optimization**: Features the ability to compile and optimize the underlying DSPy program for improved performance and accuracy. This can be triggered using the `--optimize` flag in the CLI (`main.py`).
*   **Interactive CLI**: Provides a user-friendly command-line interface built with `typer`. It uses `rich` to deliver clear, well-formatted, and colorized output for refactoring plans and evaluation results (`main.py`, `ui.py`).
*   **MLflow Integration**: Comes with built-in support for experiment tracing using MLflow. Users can configure the MLflow tracking URI and experiment name via CLI arguments (`--mlflow-uri`, `--mlflow-experiment`) to log and monitor refactoring runs (`main.py`).

## Installation

Before you begin, ensure you have Python 3.10 or newer installed on your system. This project uses `uv` for fast and efficient dependency management.

### Standard Installation

To install Robofactor for regular use, clone the repository and run the following command from the project root:

```bash
make install
```

This command uses `uv` to install the package and its required dependencies.

### Development Installation

If you plan to contribute to the project, you will need to install the development dependencies, which include tools for testing, linting, and type-checking. Use the following command:

```bash
make install-dev
```

This will install all dependencies, including the development-specific ones listed in `pyproject.toml`.

## Usage

Robofactor is a command-line tool designed to analyze and refactor a single Python file.

To refactor a Python file, run the tool with the path to your script. By default, it performs a dry run, printing the proposed changes to the console without modifying the original file.

```bash
robofactor path/to/your/file.py
```

### Example Workflow

1.  **Analyze the Code (Dry Run)**

    Run Robofactor on a script to see the proposed refactoring. The tool will display the original code, the refactoring plan, the refactored code, and an evaluation of the changes.

    ```bash
    robofactor src/my_app/utils.py
    ```

2.  **Apply the Changes**

    If you are satisfied with the proposed changes, you can write them back to the original file using the `--write` flag.

    ```bash
    robofactor --write src/my_app/utils.py
    ```

### Command-Line Options

Here are some of the key arguments and options available. The descriptions are based on the output of `robofactor --help`.

| Argument / Option | Description |
| --- | --- |
| `PATH` | The path to the Python file you want to refactor. |
| `--write` | Write the refactored code back to the original file. |
| `--optimize` | Force re-optimization of the underlying DSPy model. |
| `--dog-food` | A special mode to make Robofactor refactor its own source code. |
| `--task-llm <MODEL>` | Specify the language model for the main refactoring task. |
| `--tracing / --no-tracing` | Enable or disable MLflow tracing for experiment tracking. |
| `--mlflow-uri <URI>` | Set the MLflow tracking server URI (default: `http://127.0.0.1:5000`). |
| `--mlflow-experiment <NAME>` | Set the MLflow experiment name (default: `robofactor`). |

For a complete list of all available options, run:

```bash
robofactor --help
```

## How It Works

Robofactor follows a structured, multi-stage process to analyze, refactor, and evaluate Python code. The architecture is designed to be robust and transparent, leveraging modern tools for each step.

1.  **Code Parsing & Extraction**
    The process begins by parsing the target Python file. Using Python's built-in `ast` (Abstract Syntax Tree) module, the tool traverses the code's structure. As detailed in `src/robofactor/function_extraction.py`, it identifies every function and extracts comprehensive metadata, including its name, parameters, decorators, and docstring. This creates a structured representation of the code to be refactored.

2.  **LLM-Powered Refactoring with DSPy**
    The extracted function code is then passed to a `dspy.Module`, specifically the `CodeRefactor` class found in `src/robofactor/dspy_modules.py`. This module contains a sophisticated prompt that instructs a Large Language Model (LLM) to analyze the provided code snippet, identify areas for improvement, and generate a refactored version. The LLM's goal is to enhance code quality, readability, and performance while preserving its original functionality.

3.  **Programmatic Evaluation Pipeline**
    Once the LLM returns the refactored code, it undergoes a rigorous, automated evaluation pipeline defined in `src/robofactor/evaluation.py`. This pipeline, built using the `returns` library for robust error handling (railway-oriented programming), consists of several checks:
    *   **Syntax Check**: Verifies that the generated code is valid Python.
    *   **Quality Check**: Uses `flake8` to score the code against PEP 8 standards and other common issues.
    *   **Functional Correctness**: Executes the refactored code against a set of predefined test cases to ensure it still produces the correct output.
    If any step fails, the pipeline short-circuits and reports the failure.

4.  **Rich Terminal Display**
    Finally, the results of the refactoring and evaluation are presented to the user in the terminal. The `src/robofactor/ui.py` module uses the `rich` library to create clear, well-formatted tables and panels that display the original code, the refactored code, the LLM's reasoning, and the detailed evaluation scores.

## Development

To contribute to Robofactor, you'll need to set up a local development environment. This project uses `uv` for fast dependency management and a `Makefile` to provide convenient shortcuts for common tasks.

First, clone the repository:

```bash
git clone https://github.com/ethan-wickstrom/robofactor.git
cd robofactor
```

### Setup

To install all dependencies, including development tools like `ruff`, `mypy`, and `pytest`, run the following command. This will create a virtual environment and install all required packages.

```bash
make install-dev
```

### Common Development Tasks

The `Makefile` includes several targets to streamline the development workflow:

*   **Run all checks:** To ensure code quality before committing, run all linters, type-checkers, and tests at once.
    ```bash
    make check
    ```
*   **Run tests:** Execute the test suite using pytest.
    ```bash
    make test
    ```
*   **Linting:** Check for code style issues and automatically apply fixes using Ruff.
    ```bash
    make lint
    ```
*   **Formatting:** Format the code using Ruff Formatter and isort.
    ```bash
    make format
    ```
*   **Type-checking:** Perform static type analysis with mypy.
    ```bash
    make type-check
    ```

## Contributing

Contributions are welcome! If you find a bug, have a feature request, or want to contribute to the code, please open an issue on our GitHub repository.

- **Issues:** [https://github.com/ethan-wickstrom/robofactor/issues](https://github.com/ethan-wickstrom/robofactor/issues)

Please check the existing issues to see if your suggestion has already been discussed.
