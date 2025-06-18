import ast
import json
import os
import re
import warnings
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import subprocess
import tempfile
from pathlib import Path

import dspy
import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# Filter out Pydantic serialization warnings that occur due to LLM response format mismatches
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Pydantic serializer warnings.*PydanticSerializationUnexpectedValue.*"
)


class TestCase(BaseModel):
    """A single, executable test case for a function."""

    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    expected_output: Any


class CodeQualityScores(BaseModel):
    """Holds various code quality metrics."""

    linting_score: float
    complexity_score: float
    typing_score: float
    docstring_score: float
    linting_issues: List[str] = Field(default_factory=list)


class CodeAnalysis(dspy.Signature):
    """Analyze Python code for its purpose, complexity, and areas for improvement."""

    code_snippet: str = dspy.InputField(desc="The Python code to be analyzed.")
    analysis: str = dspy.OutputField(
        desc="A concise summary of the code's functionality and complexity."
    )
    refactoring_opportunities: List[str] = dspy.OutputField(
        desc="A bulleted list of specific, actionable refactoring opportunities."
    )


class RefactoringPlan(dspy.Signature):
    """Create a step-by-step plan to refactor Python code based on an analysis."""

    code_snippet: str = dspy.InputField(desc="The original Python code snippet.")
    analysis: str = dspy.InputField(desc="The analysis of the code snippet.")
    refactoring_summary: str = dspy.OutputField(
        desc="A high-level summary of the refactoring goal."
    )
    plan_steps: List[str] = dspy.OutputField(
        desc="A detailed, step-by-step list of actions to refactor the code."
    )


class RefactoredCode(dspy.Signature):
    """Generate refactored Python code based on a plan."""

    original_code: str = dspy.InputField(desc="The original, un-refactored Python code.")
    refactoring_summary: str = dspy.InputField(desc="The high-level goal of the refactoring.")
    plan_steps: List[str] = dspy.InputField(desc="The step-by-step plan to apply.")
    refactored_code: str = dspy.OutputField(
        prefix="```python\n",
        desc="The final, PEP8-compliant, refactored Python code block with type hints and docstrings.",
    )
    implementation_explanation: str = dspy.OutputField(
        desc="A brief explanation of how the plan was implemented."
    )


class EvaluationSignature(dspy.Signature):
    """Evaluate the refactored code based on quantitative scores and provide a final assessment."""

    code_snippet: str = dspy.InputField(desc="The refactored code being evaluated.")
    quality_scores: str = dspy.InputField(
        desc="A JSON object of quantitative scores (linting, complexity, typing, docstrings)."
    )
    functional_score: float = dspy.InputField(
        desc="A score from 0.0 to 1.0 indicating test pass rate."
    )
    final_score: float = dspy.OutputField(
        desc="A final, holistic score from 0.0 to 1.0, weighting functional correctness most heavily."
    )
    final_suggestion: str = dspy.OutputField(
        desc="A final suggestion for improvement or a confirmation of readiness."
    )


def _extract_python_code(text: str) -> str:
    """Extracts Python code from a markdown block."""
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    return match.group(1) if match else text


def check_syntax(code: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Checks for valid Python syntax and a top-level function."""
    try:
        tree = ast.parse(code)
        func_node = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        if not func_node:
            return False, None, "No top-level function definition found."
        return True, func_node.name, None
    except SyntaxError as e:
        return False, None, f"Syntax Error: {e}"


def check_code_quality(code: str, func_name: Optional[str] = None) -> CodeQualityScores:
    """
    Analyzes a string of Python code for quality metrics.

    This function assesses the code on several axes:
    1.  Linting: Uses flake8 to check for PEP 8 compliance and other common issues.
    2.  Complexity: Uses flake8's McCabe complexity check (C901).
    3.  Typing: Checks for the presence of type hints on arguments and return values.
    4.  Docstrings: Verifies the existence of a docstring for the specified function.

    If `func_name` is provided, analysis is focused on that function. Otherwise,
    it analyzes all functions found in the code.

    Args:
        code: A string containing the Python code to analyze.
        func_name: The optional name of a specific function to analyze.

    Returns:
        A CodeQualityScores object containing the calculated scores and a list
        of linting issues.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        result = subprocess.run(
            ["flake8", "--max-complexity=10", tmp_path],
            capture_output=True,
            text=True,
            check=False,
        )
        all_issues = result.stdout.strip().splitlines() if result.stdout else []

        complexity_warnings = [issue for issue in all_issues if "C901" in issue]
        linting_issues = [issue for issue in all_issues if "C901" not in issue]

        complexity_score = 1.0 if not complexity_warnings else 0.0
        linting_score = max(0.0, 1.0 - (0.1 * len(linting_issues)))

        docstring_score, typing_score = 0.0, 0.0
        try:
            tree = ast.parse(code)
            all_funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

            if not all_funcs:
                return CodeQualityScores(
                    linting_score=linting_score,
                    complexity_score=complexity_score,
                    typing_score=0.0,
                    docstring_score=0.0,
                    linting_issues=linting_issues,
                )

            target_funcs = (
                [f for f in all_funcs if f.name == func_name] if func_name else all_funcs
            )

            if not target_funcs:
                return CodeQualityScores(
                    linting_score=linting_score,
                    complexity_score=complexity_score,
                    typing_score=0.0,
                    docstring_score=0.0,
                    linting_issues=linting_issues,
                )

            docstring_score = sum(
                1.0 for f in target_funcs if ast.get_docstring(f)
            ) / len(target_funcs)

            typed_elements = 0
            typeable_elements = 0
            for func_node in target_funcs:
                args = func_node.args
                num_typed_args = sum(1 for arg in args.args if arg.annotation is not None)
                num_total_args = len(args.args)
                has_return_annotation = 1 if func_node.returns is not None else 0

                typed_elements += num_typed_args + has_return_annotation
                typeable_elements += num_total_args + 1

            typing_score = (
                typed_elements / typeable_elements if typeable_elements > 0 else 0.0
            )

        except SyntaxError:
            pass

        return CodeQualityScores(
            linting_score=linting_score,
            complexity_score=complexity_score,
            typing_score=typing_score,
            docstring_score=docstring_score,
            linting_issues=linting_issues,
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def check_functional_correctness(code: str, func_name: str, test_cases: List[TestCase]) -> int:
    """Executes test cases against the refactored code in a sandboxed environment."""
    if not test_cases:
        return 0
    passed_count = 0
    try:
        with dspy.PythonInterpreter() as interp:
            interp.execute(code)
            for test in test_cases:
                try:
                    # Use triple quotes for safer JSON embedding and normalize expected output
                    args_json = json.dumps(test.args)
                    kwargs_json = json.dumps(test.kwargs)
                    exec_cmd = f"""import json; print(json.dumps({func_name}(*json.loads('''{args_json}'''), **json.loads('''{kwargs_json}'''))))"""

                    actual_output_json = interp.execute(exec_cmd)
                    actual_output = json.loads(actual_output_json)

                    # Normalize expected output by serializing and deserializing
                    # to handle type differences (e.g., tuple vs. list).
                    normalized_expected_output = json.loads(json.dumps(test.expected_output))

                    if actual_output == normalized_expected_output:
                        passed_count += 1
                except Exception:
                    # Silently continue if a single test case fails to execute.
                    continue
    except Exception:
        # If the interpreter fails to initialize or execute the main code, return 0.
        return 0
    return passed_count


class CodeRefactor(dspy.Module):
    """A module that analyzes, plans, and refactors Python code."""

    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(CodeAnalysis)
        self.planner = dspy.Predict(RefactoringPlan)
        self.implementer = dspy.Predict(RefactoredCode)

    def forward(self, code_snippet: str) -> dspy.Prediction:
        analysis = self.analyzer(code_snippet=code_snippet)
        plan = self.planner(code_snippet=code_snippet, analysis=analysis.analysis)
        impl = self.implementer(
            original_code=code_snippet,
            refactoring_summary=plan.refactoring_summary,
            plan_steps=plan.plan_steps,
        )
        return dspy.Prediction(
            analysis=analysis.analysis,
            refactoring_opportunities=analysis.refactoring_opportunities,
            refactoring_summary=plan.refactoring_summary,
            plan_steps=plan.plan_steps,
            refactored_code=impl.refactored_code,
            implementation_explanation=impl.implementation_explanation,
        )


class RefactoringEvaluator(dspy.Module):
    """A module to evaluate refactored code using programmatic checks and LLM judgment."""

    def __init__(self):
        super().__init__()
        self.evaluator = dspy.Predict(EvaluationSignature)

    def forward(
        self, original_example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> float:
        code = _extract_python_code(prediction.refactored_code)
        raw_tests = original_example.get("test_cases")
        tests = [TestCase(**tc) for tc in raw_tests] if raw_tests else []

        is_valid, func_name, _ = check_syntax(code)
        if not is_valid:
            return 0.0

        # Handle module vs function refactoring
        if not tests:  # Module refactoring
            quality = check_code_quality(code)  # No func_name, analyze whole module
            functional_score = 1.0  # Assume functional if syntax is ok, since no tests
        else:  # Function refactoring
            if not func_name:
                return 0.0  # Cannot test if no function found in snippet
            quality = check_code_quality(code, func_name)
            passed_tests = check_functional_correctness(code, func_name, tests)
            functional_score = (passed_tests / len(tests)) if tests else 1.0

        eval_result = self.evaluator(
            code_snippet=code,
            quality_scores=quality.model_dump_json(),
            functional_score=functional_score,
        )
        try:
            return float(eval_result.final_score)
        except (ValueError, TypeError):
            return 0.0


def get_training_data() -> List[dspy.Example]:
    """Returns a list of examples for training the refactoring tool."""
    return [
        dspy.Example(
            code_snippet="""
def process_data(d):
    res = [x['price'] * x['qty'] for x in d if x['qty'] > 0]
    total = 0
    for r in res:
        total += r
    tax = 0.08
    final_total = total * (1 + tax)
    return final_total
""",
            test_cases=[
                TestCase(
                    args=[[{"price": 10, "qty": 2}, {"price": 5, "qty": -1}]],
                    expected_output=21.6,
                ).model_dump(),
                TestCase(
                    args=[[{"price": 100, "qty": 1}, {"price": 20, "qty": 5}]],
                    expected_output=216.0,
                ).model_dump(),
                TestCase(args=[[]], expected_output=0.0).model_dump(),
            ],
        ).with_inputs("code_snippet"),
        dspy.Example(
            code_snippet="""
            def proc_trans(t, d1, d2, disc_rules):
                r = {}
                for i in range(len(t)):
                    if t[i][2] >= d1 and t[i][2] <= d2:
                        u = t[i][0]
                        if u not in r:
                            r[u] = {'t': [], 'sum': 0, 'cnt': 0, 'max': 0, 'disc': 0}
                        r[u]['t'].append(t[i])
                        r[u]['sum'] = r[u]['sum'] + t[i][1]
                        r[u]['cnt'] = r[u]['cnt'] + 1
                        if t[i][1] > r[u]['max']:
                            r[u]['max'] = t[i][1]

                for k in r.keys():
                    total = r[k]['sum']
                    cnt = r[k]['cnt']

                    # Apply discounts based on complex rules
                    d = 0
                    for rule in disc_rules:
                        if rule[0] == 'total' and total > rule[1]:
                            d = d + rule[2]
                        elif rule[0] == 'count' and cnt > rule[1]:
                            d = d + rule[2]
                        elif rule[0] == 'max' and r[k]['max'] > rule[1]:
                            d = d + rule[2]

                    if d > 0.5:
                        d = 0.5  # Cap discount at 50%

                    r[k]['disc'] = d
                    r[k]['final'] = total * (1 - d)

                    # Calculate average
                    avg = 0
                    if cnt > 0:
                        avg = total / cnt
                    r[k]['avg'] = avg

                    # Format transactions
                    trans_str = ""
                    for j in range(len(r[k]['t'])):
                        if j > 0:
                            trans_str = trans_str + ";"
                        trans_str = trans_str + str(r[k]['t'][j][1])
                    r[k]['trans_str'] = trans_str

                # Convert to list format
                output = []
                for user in r:
                    entry = []
                    entry.append(user)
                    entry.append(r[user]['sum'])
                    entry.append(r[user]['avg'])
                    entry.append(r[user]['max'])
                    entry.append(r[user]['disc'])
                    entry.append(r[user]['final'])
                    entry.append(r[user]['trans_str'])
                    output.append(entry)

                return output
        """,
            test_cases=[
                TestCase(
                    args=[
                        [
                            ("user1", 100, "2024-01-01"),
                            ("user1", 200, "2024-01-02"),
                            ("user2", 150, "2024-01-01"),
                            ("user1", 50, "2024-01-03"),
                            ("user2", 300, "2024-01-04"),
                        ],
                        "2024-01-01",
                        "2024-01-03",
                        [("total", 250, 0.1), ("count", 2, 0.05), ("max", 150, 0.15)],
                    ],
                    expected_output=[
                        ["user1", 350, 116.66666666666667, 200, 0.3, 245.0, "100;200;50"],
                        ["user2", 150, 150.0, 150, 0.15, 127.5, "150"],
                    ],
                ).model_dump(),
                TestCase(
                    args=[[("user1", 100, "2024-01-01")], "2024-01-01", "2024-01-01", []],
                    expected_output=[["user1", 100, 100.0, 100, 0, 100.0, "100"]],
                ).model_dump(),
                TestCase(
                    args=[[], "2024-01-01", "2024-01-31", [("total", 100, 0.1)]], expected_output=[]
                ).model_dump(),
            ],
        ).with_inputs("code_snippet")
    ]


# --- Data Structures for Evaluation ---
class SyntaxCheckResult(NamedTuple):
    """Encapsulates the result of a syntax check."""

    is_valid: bool
    func_name: Optional[str]
    error_message: Optional[str]


class FunctionalCheckResult(NamedTuple):
    """Encapsulates the result of functional correctness tests."""

    passed_tests: int
    total_tests: int


class EvaluationResult(NamedTuple):
    """Holds all evaluation results for a piece of refactored code."""

    code: str
    syntax_check: SyntaxCheckResult
    quality_scores: Optional[CodeQualityScores]
    functional_check: Optional[FunctionalCheckResult]


# --- Core Logic (Pure Functions) ---
def evaluate_refactoring(
    prediction: dspy.Prediction, example: dspy.Example
) -> EvaluationResult:
    """
    Performs a full evaluation of the refactored code without any I/O.

    This function checks syntax, code quality, and functional correctness, returning
    a structured result. It handles both single-function and whole-module
    refactoring evaluation based on the presence of test cases in the example.

    Args:
        prediction: The dspy.Prediction object containing the refactored code.
        example: The dspy.Example object containing test cases.

    Returns:
        An EvaluationResult object with all analysis data.
    """
    code = _extract_python_code(prediction.refactored_code)
    is_valid, func_name, err = check_syntax(code)
    syntax_result = SyntaxCheckResult(is_valid, func_name, err)

    if not is_valid:
        return EvaluationResult(
            code=code,
            syntax_check=syntax_result,
            quality_scores=None,
            functional_check=None,
        )

    raw_tests = example.get("test_cases")
    tests = [TestCase(**tc) for tc in raw_tests] if raw_tests else []

    if not tests:
        # Module-level evaluation (no test cases)
        quality = check_code_quality(code)  # Analyze all functions
        functional_result = FunctionalCheckResult(passed_tests=0, total_tests=0)
    else:
        # Function-level evaluation
        if not func_name:
            # Tests provided, but no function found in the code snippet.
            err_msg = "Tests provided, but no function found in code snippet."
            return EvaluationResult(
                code=code,
                syntax_check=SyntaxCheckResult(is_valid, None, err_msg),
                quality_scores=None,
                functional_check=None,
            )

        quality = check_code_quality(code, func_name)
        passed_count = check_functional_correctness(code, func_name, tests)
        functional_result = FunctionalCheckResult(passed_count, len(tests))

    return EvaluationResult(
        code=code,
        syntax_check=syntax_result,
        quality_scores=quality,
        functional_check=functional_result,
    )


# --- Presentation Logic (Side Effects) ---
def display_refactoring_process(console: Console, prediction: dspy.Prediction) -> None:
    """Displays the LLM's refactoring process using rich components."""
    console.print(Panel(prediction.analysis, title="[bold cyan]Analysis[/bold cyan]", expand=False))

    plan_text = Text()
    plan_text.append("Summary: ", style="bold")
    plan_text.append(prediction.refactoring_summary)
    plan_text.append("\n\n")
    for i, step in enumerate(prediction.plan_steps, 1):
        plan_text.append(f"{i}. {step}\n")
    console.print(Panel(plan_text, title="[bold cyan]Refactoring Plan[/bold cyan]"))

    console.print(
        Panel(
            Syntax(
                _extract_python_code(prediction.refactored_code),
                "python",
                theme="monokai",
                line_numbers=True,
            ),
            title="[bold cyan]Final Refactored Code[/bold cyan]",
        )
    )
    console.print(
        Panel(
            prediction.implementation_explanation,
            title="[bold cyan]Implementation Explanation[/bold cyan]",
        )
    )


def display_evaluation_results(console: Console, result: EvaluationResult) -> None:
    """Displays the evaluation results using rich components."""
    console.print(Rule("[bold yellow]Final Output Evaluation[/bold yellow]"))

    if not result.syntax_check.is_valid:
        error_msg = result.syntax_check.error_message or "Unknown syntax error."
        console.print(
            Panel(
                f"[bold red]Syntax Error:[/bold red] {error_msg}",
                title="[bold red]Evaluation Failed[/bold red]",
                border_style="red",
            )
        )
        return

    # This check is needed for type checkers to understand the implications of the check above.
    if not result.quality_scores or not result.functional_check:
        console.print(
            "[bold red]Error:[/bold red] Missing quality or functional results despite valid syntax."
        )
        return

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column()
    table.add_column(style="bold magenta")

    func_check = result.functional_check
    if func_check.total_tests > 0:
        table.add_row(
            "Functional Equivalence:", f"{func_check.passed_tests} / {func_check.total_tests}"
        )
    else:
        table.add_row("Functional Equivalence:", "N/A (no tests)")

    quality = result.quality_scores
    table.add_row("Linting Score:", f"{quality.linting_score:.2f}")
    table.add_row("Typing Score:", f"{quality.typing_score:.2f}")
    table.add_row("Docstring Score:", f"{quality.docstring_score:.2f}")

    console.print(table)

    if quality.linting_issues:
        lint_issues_text = Text("\n- ".join(quality.linting_issues))
        console.print(
            Panel(lint_issues_text, title="[yellow]Linting Issues[/yellow]", border_style="yellow")
        )


def main(
    self_refactor: bool = typer.Option(
        False,
        "--self-refactor",
        help="Enable self-refactoring mode to refactor this script.",
    ),
    write: bool = typer.Option(
        False, "--write", help="Write the refactored code back to the file in self-refactor mode."
    ),
    optimize: bool = typer.Option(
        False, "--optimize", help="Run the DSPy optimizer to generate a new `optimized.json`."
    ),
):
    """A DSPy-powered tool to analyze, plan, and refactor Python code."""
    warnings.filterwarnings("ignore")
    console = Console()

    task_llm = dspy.LM("gemini/gemini-2.5-pro", max_tokens=64000)
    prompt_llm = dspy.LM("xai/grok-3-mini-fast", max_tokens=32000)
    dspy.configure(lm=task_llm)

    optimizer_path = Path("optimized.json")

    if optimize or not optimizer_path.exists():
        console.print(
            "[yellow]No optimized model found or --optimize flag set. Running optimization...[/yellow]"
        )
        teleprompter = dspy.MIPROv2(
            metric=RefactoringEvaluator(),
            prompt_model=prompt_llm,
            task_model=task_llm,
            auto="heavy",
            num_threads=8,
        )
        refactorer = teleprompter.compile(
            CodeRefactor(), trainset=get_training_data(), requires_permission_to_run=False
        )
        self_correcting_refactorer = dspy.Refine(
            module=refactorer, reward_fn=RefactoringEvaluator(), threshold=0.9, N=3
        )
        console.print(f"Optimization complete. Saving to {optimizer_path}...")
        self_correcting_refactorer.save(str(optimizer_path))
    else:
        console.print(f"Loading optimized model from {optimizer_path}...")
        refactorer = CodeRefactor()
        self_correcting_refactorer = dspy.Refine(
            module=refactorer, reward_fn=RefactoringEvaluator(), threshold=0.9, N=3
        )
        self_correcting_refactorer.load(str(optimizer_path))
        console.print("[green]Optimized model loaded successfully![/green]")

    if self_refactor:
        run_self_refactor(console, self_correcting_refactorer, write)


def run_self_refactor(console: Console, refactorer: dspy.Module, write: bool):
    """Handles the self-refactoring workflow."""
    console.print(Rule("[bold magenta]Self-Refactoring Mode[/bold magenta]"))
    script_path = Path(__file__).resolve()

    with open(script_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    console.print(
        Panel(
            Syntax(source_code, "python", theme="monokai", line_numbers=True),
            title=f"[bold]Original Code: {script_path.name}[/bold]",
            border_style="blue",
        )
    )

    self_refactor_example = dspy.Example(code_snippet=source_code, test_cases=[]).with_inputs(
        "code_snippet"
    )

    prediction = refactorer(**self_refactor_example.inputs())
    display_refactoring_process(console, prediction)

    evaluation = evaluate_refactoring(prediction, self_refactor_example)
    display_evaluation_results(console, evaluation)

    refactored_code = _extract_python_code(prediction.refactored_code)
    is_valid, _, err = check_syntax(refactored_code)

    if write:
        if is_valid:
            console.print(f"[yellow]Writing refactored code back to {script_path}...[/yellow]")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(refactored_code)
            console.print("[green]Self-refactoring complete.[/green]")
        else:
            console.print(
                f"[bold red]Skipping write-back due to syntax errors:[/bold red]\n{err}"
            )

if __name__ == "__main__":
    typer.run(main)
