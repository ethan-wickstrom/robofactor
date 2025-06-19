# Robofactor

> The robot who refactors: /[^_^]\

Robofactor is a command-line tool that leverages the DSPy framework to automatically analyze, plan, and refactor Python code. It takes a Python file as input, uses a large language model to generate a refactored version, and then programmatically evaluates the result for correctness and quality improvements. The goal is to provide an intelligent assistant that helps developers improve their codebase by enhancing readability, maintainability, and adherence to best practices.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Development](#development)
- [Contributing](#contributing)

## Features

- **AI-Powered Refactoring**: Leverages the [DSPy](https://github.com/stanfordnlp/dspy) framework to intelligently analyze, plan, and rewrite Python code for improved quality and maintainability.
- **Robust Multi-Stage Evaluation**: Ensures the quality of refactored code through a comprehensive pipeline that includes syntax validation, `flake8` quality analysis, and functional correctness testing.
- **Deep Code Comprehension**: Utilizes Python's Abstract Syntax Tree (AST) to parse and understand the structure, context, and metadata of functions within the source code.
- **Rich CLI Experience**: Provides a user-friendly command-line interface powered by `Typer` and `Rich`, offering clear, formatted output and progress indicators.
- **Experiment Tracing**: Integrates with [MLflow](https://mlflow.org/) for detailed tracing and logging of refactoring runs, enabling reproducibility and performance analysis.

## Installation

This project uses `uv` for fast and efficient dependency management.

1.  **Prerequisite: Install `uv`**

    If you do not have `uv` installed, you can install it with pip:
    ```bash
    pip install uv
    ```

2.  **Clone the Repository**

    Clone the project repository from GitHub:
    ```bash
    git clone https://github.com/ethan-wickstrom/robofactor.git
    cd robofactor
    ```

3.  **Install Dependencies**

    The `Makefile` provides convenient commands for installation. Choose the one that fits your needs.

    *   **Standard Installation**

        For regular use of the `robofactor` CLI tool, run the following command. This installs only the essential production dependencies.
        ```bash
        make install
        ```

    *   **Development Installation**

        If you plan to contribute to the project, run tests, or use the development tools (like linters and formatters), use this command. It installs all dependencies, including the development-specific ones.
        ```bash
        make install-dev
        ```

## Usage

Robofactor is a command-line tool designed to refactor a single Python file at a time.

To analyze and refactor a file, provide the path to the script:

```bash
robofactor path/to/your/file.py
```

By default, the tool will print the refactored code to the console without modifying the original file.

### Key Options

You can customize the behavior of Robofactor using the following command-line options:

*   `--write`: Writes the refactored code directly back to the source file.
    ```bash
    robofactor path/to/your/file.py --write
    ```
*   `--dog-food`: A special mode to run the refactorer on its own source code. This is useful for testing and demonstration.
    ```bash
    robofactor --dog-food
    ```
*   `--optimize`: Forces the recompilation of the underlying DSPy model. This can be slow but may improve results if the logic has changed.
*   `--task-llm` & `--prompt-llm`: Specify the large language models to use. The `task-llm` performs the main refactoring, while the `prompt-llm` is used during the optimization phase.
    ```bash
    # Example using different models
    robofactor path/to/file.py --task-llm gpt-4o --prompt-llm gpt-4-turbo
    ```
*   `--tracing` & `--mlflow-uri`: Configure integration with MLflow for experiment tracking. Tracing is enabled by default.
    ```bash
    # Disable tracing
    robofactor path/to/file.py --no-tracing

    # Point to a different MLflow instance
    robofactor path/to/file.py --mlflow-uri http://your-mlflow-server:5000
    ```

### Full Help Text

For a complete list of all available commands and options, you can run `robofactor --help`.

<details>
<summary>Click to view the full CLI help text</summary>

```text
                                                                                                    
 Usage: main [OPTIONS] [PATH]                                                                       
                                                                                                    
 A DSPy-powered tool to analyze, plan, and refactor Python code.                                    
                                                                                                    
                                                                                                    
 Arguments 
 ─────────────────────────────────────────────────────────────────────────────────────────────────── 
   path      [PATH]  Path to the Python file to refactor. [default: None]                         
                                                                                                    
 Options 
 ─────────────────────────────────────────────────────────────────────────────────────────────────── 
 --dog-food                                    Self-refactor the script you are running.          
 --write                                       Write the refactored code back to the file.        
 --optimize                                    Force re-optimization of the DSPy model.           
 --task-llm                              TEXT  Model for the main refactoring task.               
                                               [default:                                          
                                               gemini/gemini-2.5-flash-lite-preview-06-17]        
 --prompt-llm                            TEXT  Model for generating prompts during optimization.  
                                               [default: gemini/gemini-2.5-pro]                   
 --tracing               --no-tracing          Enable MLflow tracing. [default: tracing]          
 --mlflow-uri                            TEXT  MLflow tracking server URI.                        
                                               [default: http://127.0.0.1:5000]                   
 --mlflow-experiment                     TEXT  MLflow experiment name. [default: robofactor]      
 --install-completion                          Install completion for the current shell.          
 --show-completion                             Show completion for the current shell, to copy it  
                                               or customize the installation.                     
 --help                                        Show this message and exit.                        
                                                                                                    

```

</details>

## How It Works

Robofactor employs a multi-stage pipeline to intelligently refactor Python code, combining static analysis with the power of Large Language Models. The process is as follows:

1.  **Code Parsing & Analysis**
    The tool begins by parsing the target Python file into an Abstract Syntax Tree (AST). This detailed structural analysis, handled by the logic in `src/robofactor/function_extraction.py`, allows the tool to understand the code's components, such as functions and their signatures, before any refactoring begins.

2.  **AI-Powered Refactoring**
    The core refactoring logic is driven by a `dspy.Module` named `CodeRefactor` (found in `src/robofactor/dspy_modules.py`). This module orchestrates a process where a Large Language Model (LLM) first analyzes the code, creates a detailed plan for improvement, and then implements that plan by generating new, refactored code.

3.  **Rigorous Evaluation Pipeline**
    The newly generated code is not trusted blindly. It is immediately passed to a comprehensive evaluation pipeline, orchestrated by the `evaluate_refactored_code` function in `src/robofactor/evaluation.py`. This pipeline performs a series of critical checks:
    *   **Syntax Validation**: Ensures the code is syntactically correct Python.
    *   **Code Quality Analysis**: Measures metrics like docstring coverage, type hinting, and complexity using AST analysis and `flake8`.
    *   **Functional Correctness**: Executes the refactored code against a set of test cases in a sandboxed environment to verify that its behavior remains unchanged.

4.  **Rich Terminal Output**
    Finally, the entire process—from the AI's reasoning and refactoring plan to the detailed results of the evaluation—is presented to the user in a clear and readable format. This is handled by functions in `src/robofactor/ui.py`, which leverage the `rich` library to create styled tables and syntax-highlighted code for an intuitive user experience.

## Development

To get started with development, clone the repository and set up the environment using the provided Makefile. This project uses `uv` for package management and `ruff` for linting and formatting.

### Setup

Install all dependencies, including development tools, by running:

```sh
make install-dev
```

### Code Quality & Testing

The Makefile provides several targets to help maintain code quality.

*   **Run all checks**: To run linting, type-checking, and tests all at once, use:
    ```sh
    make check
    ```

*   **Linting**: To check for code style issues with `ruff`, run:
    ```sh
    make lint
    ```

*   **Formatting**: To automatically format the code with `ruff` and `isort`, run:
    ```sh
    make format
    ```

*   **Type Checking**: To perform static type analysis with `mypy`, use:
    ```sh
    make type-check
    ```

*   **Running Tests**: To execute the test suite with `pytest`, run:
    ```sh
    make test
    ```

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, or if you've found a bug, please open an issue on our GitHub repository. You can do so by visiting the [Issues page](https://github.com/ethan-wickstrom/robofactor/issues).

Thank you for helping to improve Robofactor!