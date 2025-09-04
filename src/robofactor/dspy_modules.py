import logging

import dspy

from robofactor.signatures.code_analysis import CodeAnalysis
from robofactor.signatures.final_evaluation import FinalEvaluation
from robofactor.signatures.refactored_code import RefactoredCode
from robofactor.signatures.refactoring_plan import RefactoringPlan

FAILURE_SCORE = 0.0
TRAINING_DATA_FILE = "training.json"
logger = logging.getLogger(__name__)


class CodeRefactor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(CodeAnalysis)
        self.planner = dspy.Predict(RefactoringPlan)
        self.implementer = dspy.Predict(RefactoredCode)
        self.evaluator = dspy.Predict(FinalEvaluation)

    def forward(self, code_snippet: str) -> dspy.Prediction:
        analysis_result = self.analyzer(code_snippet=code_snippet)
        plan_result = self.planner(
            code_snippet=code_snippet,
            analysis=analysis_result.analysis.analysis
        )
        impl_result = self.implementer(
            original_code=code_snippet,
            refactoring_summary=plan_result.plan.refactoring_summary,
            plan_steps=plan_result.plan.plan_steps,
        )
        eval_result = self.evaluator(
            code_snippet=impl_result.implementation.refactored_code,
            quality_scores=impl_result.implementation.quality_scores,
            functional_score=impl_result.implementation.functional_score,
        )
        return dspy.Prediction(
            analysis=analysis_result.analysis.analysis,
            refactoring_opportunities=analysis_result.analysis.refactoring_opportunities,
            refactoring_summary=plan_result.plan.refactoring_summary,
            plan_steps=plan_result.plan.plan_steps,
            refactored_code=impl_result.implementation.refactored_code,
            implementation_explanation=impl_result.implementation.implementation_explanation,
            final_score=eval_result.final_score,
            final_suggestion=eval_result.final_suggestion,
        )
