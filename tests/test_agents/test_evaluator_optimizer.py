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


def test_extract_critical_invariants():
    """Verify regex invariant extraction accurately pulls IDs, scores, salaries, and tracks."""
    from src.agents.evaluator_optimizer import extract_critical_invariants
    text = "Student SPR-1001 enrolled in Advanced Agentic AI achieved 95% with target 30000 EGP."
    inv = extract_critical_invariants(text)
    assert "SPR-1001" in inv["ids"]
    assert "95%" in inv["percentages"]
    assert "30000" in inv["salaries"]
    assert "Advanced Agentic AI" in inv["tracks"]


def test_verify_fact_preservation_valid():
    """Verify that consistent output passes fact preservation checks."""
    from src.agents.evaluator_optimizer import verify_fact_preservation
    context = "Learner SPR-2002 in Cybersecurity scored 100% with target salary 25000 EGP."
    response = "Recommendation for SPR-2002: In Cybersecurity, with your 100% score and 25000 EGP salary goal, apply to Tier-1 firms."
    valid, violations = verify_fact_preservation(context, response)
    assert valid is True
    assert violations == []


def test_verify_fact_preservation_detects_mutations():
    """Verify that hallucinated or altered facts are programmatically caught."""
    from src.agents.evaluator_optimizer import verify_fact_preservation
    context = "Learner SPR-3003 in Full Stack Web Development achieved 88% and wants 20000 EGP."
    
    # 1. Dropped ID
    bad_resp_1 = "Learner achieved 88% in Full Stack Web Development and wants 20000 EGP."
    valid, violations = verify_fact_preservation(context, bad_resp_1)
    assert valid is False
    assert any("Learner ID" in v for v in violations)

    # 2. Altered Track
    bad_resp_2 = "SPR-3003 in Cloud Computing & DevOps achieved 88% and wants 20000 EGP."
    valid, violations = verify_fact_preservation(context, bad_resp_2)
    assert valid is False
    assert any("Bootcamp track" in v for v in violations)

    # 3. Altered percentage
    bad_resp_3 = "SPR-3003 in Full Stack Web Development achieved 99% and wants 20000 EGP."
    valid, violations = verify_fact_preservation(context, bad_resp_3)
    assert valid is False
    assert any("Numerical percentage" in v for v in violations)


def test_optimization_loop_rejects_invariant_violating_candidate():
    """Verify that an optimizer hallucinating a new student ID is rejected by the invariant guard."""
    failed_evaluation = EvaluationResult(
        score=60,
        passed=False,
        feedback=["Add career roadmap."],
    )
    passed_evaluation = EvaluationResult(
        score=95,
        passed=True,
        feedback=["Approved."],
    )

    module = "src.agents.evaluator_optimizer"

    # Optimizer hallucinates wrong student ID (SPR-9999 instead of SPR-5005)
    corrupted_response = "SPR-9999 is recommended for Data Science."

    with (
        patch(f"{module}.evaluate_response", side_effect=[failed_evaluation, failed_evaluation, failed_evaluation]),
        patch(f"{module}.optimize_response", return_value=corrupted_response) as optimizer,
    ):
        result = evaluate_and_optimize(
            response="SPR-5005 is a Data Science student.",
            task_context="Learner SPR-5005 in Data Science completed modules.",
            max_iterations=2,
        )

    # Invariants were violated so corrupt candidate was rejected
    assert result["invariants_preserved"] is True
    # The output retained SPR-5005 rather than adopting SPR-9999
    assert "SPR-5005" in result["response"]
    assert "SPR-9999" not in result["response"]
