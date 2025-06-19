# robofactor
The robot who refactors: /[^_^]\

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
  - [Overview](#overview)
  - [Component Breakdown](#component-breakdown)

## Installation

### Prerequisites

This project uses `uv` for package management. Before you begin, please install `uv` by following the official instructions: [https://github.com/astral-sh/uv#installation](https://github.com/astral-sh/uv#installation).

### Standard Installation

For regular use of the `robofactor` CLI tool. This will install the package and its required dependencies.

```bash
# Clone the repository
git clone https://github.com/ethan-wickstrom/robofactor.git
cd robofactor

# Install using the Makefile
make install
```

### Developer Installation

If you plan to contribute to the project, you'll need to install the development dependencies (e.g., for linting, formatting, and testing).

```bash
# Clone the repository
git clone https://github.com/ethan-wickstrom/robofactor.git
cd robofactor

# Install with all development dependencies
make install-dev
```

## Usage

The `robofactor` command-line tool analyzes and refactors Python code using a large language model.

### Basic Syntax

The tool is invoked with the `robofactor` command, followed by options and the path to the Python file you want to process.

```bash
robofactor [OPTIONS] path/to/your/file.py
```

### Examples

#### 1. Analyze a File (Dry Run)

To see the suggested refactoring without modifying the original file, simply provide the path to the file. The refactored code will be printed to the console.

```bash
robofactor path/to/your/code.py
```

#### 2. Refactor and Save Changes

To apply the refactoring and write the changes back to the original file, use the `--write` flag.

```bash
robofactor --write path/to/your/code.py
```

#### 3. Use a Different Language Model

You can specify a different model for the main refactoring task using the `--task-llm` option.

```bash
robofactor --task-llm "anthropic/claude-3.5-sonnet" --write path/to/your/code.py
```

#### 4. Self-Refactor ("Dogfooding")

The tool can even refactor its own source code. Use the `--dog-food` flag for this special mode.

```bash
# Analyze robofactor's own code
robofactor --dog-food

# Analyze and save changes to robofactor's code
robofactor --dog-food --write
```

### Advanced Usage: MLflow Tracing

If you have an MLflow server running, you can trace the execution of the refactoring process.

```bash
# Ensure your MLflow server is running, e.g., at http://127.0.0.1:5000

robofactor \
  --mlflow-uri "http://127.0.0.1:5000" \
  --mlflow-experiment "refactoring_experiments" \
  --write path/to/your/code.py
```

## Architecture

### Overview

Robofactor is a command-line tool designed to automatically analyze and refactor Python code using a large language model integrated via the DSPy framework. The architecture is modular, centered around a main controller (`main.py`) that orchestrates the entire process. The flow begins when the user specifies a target file. The `function_extraction.py` module first parses this file using an Abstract Syntax Tree (AST) to understand its structure. This structured data is then fed into the core `dspy_modules.py`, where an AI model generates a refactoring plan and produces new code. This new code is then rigorously assessed by the `evaluation.py` module, which leverages `analysis.py` to perform a series of static and dynamic checks, including syntax validation, quality metrics, and functional testing. Finally, the `ui.py` module presents a detailed report of the refactoring process and the evaluation results to the user in the terminal.

### Component Breakdown

- `src/robofactor/config.py`: Acts as the central configuration hub, storing global settings and parameters that control the application's behavior.
- `src/robofactor/analysis.py`: Provides a toolkit for static and dynamic code analysis, including syntax checks, style linting (flake8), and running code against test cases in a sandboxed environment.
- `src/robofactor/ui.py`: Manages the command-line user interface, responsible for formatting and displaying the refactoring process and final evaluation results to the user.
- `src/robofactor/evaluation.py`: Orchestrates the comprehensive evaluation of refactored code, sequencing checks for syntax, quality, and functional correctness in a fail-fast manner.
- `src/robofactor/functional_types.py`: Defines `Ok` and `Err` result types to enable robust, functional-style error handling, particularly within the evaluation pipeline.
- `src/robofactor/utils.py`: A utility module containing helper functions, such as suppressing non-critical warnings from third-party libraries like Pydantic.
- `src/robofactor/dspy_modules.py`: Contains the core AI logic, defining the DSPy modules (`CodeRefactor`, `RefactoringEvaluator`) that use a language model to analyze, plan, and implement code refactorings.
- `src/robofactor/function_extraction.py`: Performs static analysis on Python source code using AST to extract detailed metadata about functions, which serves as input for the refactoring model.
- `src/robofactor/main.py`: The main entry point of the application. It defines the CLI, parses user arguments, and orchestrates the end-to-end refactoring workflow from code analysis to final output.