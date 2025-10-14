import dspy

from robofactor import analysis
from robofactor.signatures.code_analysis import CodeAnalysis
from robofactor.signatures.final_evaluation import FinalEvaluation
from robofactor.signatures.refactored_code import RefactoredCode
from robofactor.signatures.refactoring_plan import RefactoringPlan
from robofactor.types import PythonCode


class CodeRefactor(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.analyzer = dspy.Predict(CodeAnalysis)
        self.planner = dspy.Predict(RefactoringPlan)
        self.implementer = dspy.Predict(RefactoredCode)
        self.evaluator = dspy.Predict(FinalEvaluation)

    def forward(self, code_snippet: PythonCode) -> dspy.Prediction:
        analysis_result = self.analyzer(code_snippet=code_snippet)
        plan_result = self.planner(
            code_snippet=code_snippet,
            analysis=analysis_result.report,
        )
        impl_result = self.implementer(
            original_code=code_snippet,
            plan=plan_result.plan,
        )
        quality_metrics = analysis.check_code_quality(impl_result.artifact.code)
        eval_result = self.evaluator(
            code_snippet=impl_result.artifact.code,
            quality_metrics=quality_metrics,
            functional_score=0.0,  # TODO: Implement functional scoring
        )
        return dspy.Prediction(
            analysis_report=analysis_result.report,
            plan=plan_result.plan,
            artifact=impl_result.artifact,
            quality_metrics=quality_metrics,
            final_score=eval_result.final_score,
            verdict=eval_result.verdict,
            recommendations=eval_result.recommendations,
        )
