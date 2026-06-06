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

Robofactor reviews, checks, and optionally applies behavior-preserving Python refactors.

The project uses:

*   **DSPy (`dspy-ai`):** Generates the review, plan, refactored artifact, and final quality assessment.
*   **Behavior checks:** Compare explicit examples and source-versus-refactored calls before an apply.
*   **Code quality checks:** Run Ruff, ty, AST metrics, suppression-comment detection, and Rope-aware project inspection.
*   **Rich CLI (`rich`):** Renders the review, plan, diff, refactored code, and check results in the terminal.

## Key Features

*   **DSPy refactor pipeline**: `src/robofactor/modules/code_refactor.py` runs analysis, planning, implementation, and final assessment.
*   **Behavior-preservation checks**: `src/robofactor/checks/` validates the function signature, explicit behavior tests, source comparisons, generated comparisons, and forbidden suppression comments.
*   **Quality assessment**: Ruff, ty, AST signals, and Rope-backed project inspection provide deterministic feedback alongside model output.
*   **Optimization mode**: `--optimize` compiles the DSPy program against training examples.
*   **CLI output**: `src/robofactor/main.py` and `src/robofactor/ui.py` render the review, plan, diff, artifact, and check results.
*   **MLflow tracing**: `--tracing`, `--mlflow-uri`, and `--mlflow-experiment` configure experiment tracing.

## Installation

Before you begin, ensure you have Python 3.14 and Deno 2.x installed. This project uses `uv` for dependency management.

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

    Run Robofactor on a script to see the proposed refactoring. The tool will display the original code, the refactoring plan, the refactored code, and an assessment of the changes.

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

Robofactor keeps model output behind deterministic checks.

1.  **Read the target file**
    The CLI reads one Python file and sends the source to the DSPy refactoring module.

2.  **Generate a review, plan, and artifact**
    `CodeRefactor` asks DSPy signatures to produce a code review, ordered plan, refactored code artifact, and final assessment.

3.  **Check the artifact**
    `src/robofactor/refactor_check.py` validates syntax, preserves the public function signature, compares behavior against explicit tests, searches generated comparison cases, detects forbidden suppression comments, and calculates quality metrics.

4.  **Apply only after checks pass**
    Dry runs print the result. With `--write`, Robofactor writes the refactored code only after checks pass.

## Development

To contribute to Robofactor, you'll need to set up a local development environment. This project uses `uv` for fast dependency management and a `Makefile` to provide convenient shortcuts for common tasks.

First, clone the repository:

```bash
git clone https://github.com/ethan-wickstrom/robofactor.git
cd robofactor
```

### Setup

To install all dependencies, including development tools like `ruff`, `ty`, `tach`, `mutmut`, and `pytest`, run the following command. This will create a virtual environment and install all required packages.

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
*   **Formatting:** Format the code using Ruff Formatter and Ruff's import sorting.
    ```bash
    make format
    ```
*   **Type-checking:** Perform static type analysis with ty.
    ```bash
    make type-check
    ```

## Contributing

Contributions are welcome! If you find a bug, have a feature request, or want to contribute to the code, please open an issue on our GitHub repository.

- **Issues:** [https://github.com/ethan-wickstrom/robofactor/issues](https://github.com/ethan-wickstrom/robofactor/issues)

Please check the existing issues to see if your suggestion has already been discussed.
