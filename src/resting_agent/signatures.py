import dspy

from src.resting_agent.core.models import Plan


class GeneratePlan(dspy.Signature):
    """Generate a comprehensive, step-by-step plan for implementing a Laravel RESTful API.
    The plan must be structured as a list of actions in the correct order of execution.
    This includes creating models, migrations, controllers, request validation, routes,
    API resources, and feature tests.
    """
    intent: str = dspy.InputField(desc="A natural language description of the desired API features and requirements.")
    plan: Plan = dspy.OutputField(desc="A structured execution plan with an ordered list of actions.")

class GenerateCode(dspy.Signature):
    """Generate production-quality Laravel PHP code for a specific file.
    The generated code must be complete, syntactically correct, and adhere to modern
    Laravel (10+) and PHP standards, including PSR-12, strict type hints, and proper namespacing.
    The output should contain only the raw code, without any markdown formatting or explanations.
    """
    intent: str = dspy.InputField(desc="The original user intent to provide overall context.")
    file_path: str = dspy.InputField(desc="The target file path for the code, e.g., 'app/Models/Post.php'.")
    content_description: str = dspy.InputField(desc="Specific requirements and logic for this file.")
    context: str = dspy.InputField(desc="Content of related files to ensure consistency and correctness.")
    code_content: str = dspy.OutputField(desc="The complete, raw PHP code for the specified file.")
