from typing import Any

import dspy

from .core.config import AppConfig
from .core.constants import ActionType
from .core.models import ExecutionResult, Plan
from .executors import ActionExecutor, CommandActionExecutor, FileActionExecutor, TestActionExecutor
from .services.files import FileSystemHandler
from .services.shell import CommandExecutor
from .signatures import GenerateCode, GeneratePlan


class ApiAgent(dspy.Module):
    """An autonomous agent that builds Laravel RESTful APIs from a natural language intent."""

    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(GeneratePlan)
        self.code_generator = dspy.ChainOfThought(GenerateCode)
        self._executors: dict[ActionType, ActionExecutor] = {}

    def _initialize_executors(self, project_path: str):
        """Initializes action executors based on the project configuration."""
        file_handler = FileSystemHandler(project_path)
        command_executor = CommandExecutor(project_path)
        self._executors = {
            ActionType.RUN_COMMAND: CommandActionExecutor(file_handler, command_executor),
            ActionType.CREATE_FILE: FileActionExecutor(file_handler, command_executor, self.code_generator),
            ActionType.UPDATE_FILE: FileActionExecutor(file_handler, command_executor, self.code_generator),
            ActionType.RUN_TESTS: TestActionExecutor(file_handler, command_executor),
        }

    def _generate_plan(self, intent: str) -> ExecutionResult:
        """Generates a structured execution plan from the user's intent."""
        print("\n🧠 Generating execution plan...")
        try:
            prediction = self.planner(intent=intent)
            plan = prediction.plan
            if not plan.validate_dependencies():
                return ExecutionResult(False, "Plan validation failed: an action depends on a future action.")

            self._print_plan(plan)
            return ExecutionResult(True, "Plan generated successfully.", plan)
        except Exception as e:
            return ExecutionResult(False, f"Planning failed due to an unexpected error: {e}")

    def _execute_plan(self, plan: Plan, intent: str) -> list[dict[str, Any]]:
        """Executes each action in the plan sequentially."""
        print("\n🚀 Executing plan...")
        logs = []
        context = {'intent': intent}
        for i, action in enumerate(plan.actions):
            print(f"\n📋 Step {i+1}/{len(plan.actions)}: {action.action_type.value} -> {action.path or action.content_description}")

            executor = self._executors.get(action.action_type)
            if not executor:
                result = ExecutionResult(False, f"Execution failed: Unknown action type '{action.action_type}'.")
            else:
                result = executor.execute(action, context)

            if result.success:
                print(f"✅ Success: {result.message}")
            else:
                print(f"❌ Failure: {result.message}")
                if not result.success and result.data:
                    print(f"   Output: {result.data}")

            logs.append({
                'action': action.action_type.value,
                'path': action.path,
                'success': result.success,
                'message': result.message,
                'output': result.data
            })
        return logs

    def _run_final_tests(self, project_path: str) -> ExecutionResult:
        """Runs the final, complete test suite to verify the generated API."""
        print("\n🧪 Running final verification...")
        return CommandExecutor(project_path).execute("php artisan test")

    def _print_plan(self, plan: Plan):
        """Displays the generated plan in a readable format."""
        print("\n📋 Generated Plan:")
        if plan.description:
            print(f"📝 {plan.description}")
        print("-" * 50)
        for i, action in enumerate(plan.actions):
            deps = f" (depends on: {action.dependencies})" if action.dependencies else ""
            print(f"{i+1}. {action.action_type.value}: {action.path or action.content_description}{deps}")
        print("-" * 50)

    def forward(self, intent: str, project_path: str) -> dspy.Prediction:
            """Orchestrates the entire process of planning, executing, and verifying the API generation."""
            self._initialize_executors(project_path)

            plan_result = self._generate_plan(intent)
            if not plan_result.success:
                return dspy.Prediction(success=False, error=plan_result.message, final_plan=[], execution_logs=[])

            plan = plan_result.data
            if not isinstance(plan, Plan):
                return dspy.Prediction(success=False, error="Invalid plan data returned", final_plan=[], execution_logs=[])

            execution_logs = self._execute_plan(plan, intent)
            test_result = self._run_final_tests(project_path)

            return dspy.Prediction(
                success=test_result.success,
                final_plan=[action.model_dump() for action in plan.actions],
                execution_logs=execution_logs,
                final_test_results=test_result.data or test_result.message
            )
