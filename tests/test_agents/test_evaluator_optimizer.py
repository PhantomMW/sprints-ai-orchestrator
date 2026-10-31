from unittest.mock import patch

from src.agents.evaluator_optimizer import (
    EvaluationResult,
    evaluate_and_optimize,
)


def test_maximum_optimization_iterations():
    """
    Verify that optimization never exceeds three iterations.
    """

    failed_evaluation = EvaluationResult(
        score=50,
        passed=False,
        feedback=["Needs improvement."],
    )

    module = "src.agents.evaluator_optimizer"

    with (
        patch(
            f"{module}.evaluate_response",
            side_effect=[failed_evaluation] * 4,
        ) as evaluator,
        patch(
            f"{module}.optimize_response",
            return_value="Improved response.",
        ) as optimizer,
    ):

        result = evaluate_and_optimize(
            response="Initial response.",
            task_context="Verified context.",
            max_iterations=10,
        )

    assert result["status"] == "success"
    assert result["iteration_count"] == 3
    assert result["max_iterations"] == 3

    assert evaluator.call_count == 4
    assert optimizer.call_count == 3


def test_stops_when_evaluation_passes():
    """
    Verify that optimization stops immediately when evaluation passes.
    """

    passed_evaluation = EvaluationResult(
        score=95,
        passed=True,
        feedback=["Response meets quality requirements."],
    )

    module = "src.agents.evaluator_optimizer"

    with (
        patch(
            f"{module}.evaluate_response",
            return_value=passed_evaluation,
        ) as evaluator,
        patch(
            f"{module}.optimize_response",
        ) as optimizer,
    ):

        result = evaluate_and_optimize(
            response="Excellent response.",
            task_context="Verified context.",
        )

    assert result["status"] == "success"
    assert result["iteration_count"] == 0
    assert result["passed"] is True

    evaluator.assert_called_once()
    optimizer.assert_not_called()