from unittest.mock import patch

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agents.orchestrator import build_orchestrator, orchestrator


def _successful_evaluator_result(response: str) -> dict:
    """Build a deterministic successful evaluator response for tests."""
    return {
        "status": "success",
        "response": response,
        "iteration_count": 1,
        "max_iterations": 3,
        "passed": True,
        "evaluation": {
            "score": 95,
            "passed": True,
            "feedback": ["Response is accurate."],
        },
    }


def _eligible_advisor_result() -> dict:
    """Return a verified learner profile eligible for tier-1 review."""
    return {
        "status": "success",
        "student_id": "SPR-1001",
        "learner_profile": {
            "learner_id": "SPR-1001",
            "name": "Ahmed Youssef",
            "bootcamp_track": "Advanced Agentic AI",
            "current_track": "Advanced Agentic AI",
            "preferred_employers": "Fawry, Swvl",
            "readiness_status": "Ready for Placement",
        },
        "advice": {
            "learner_name": "Ahmed Youssef",
            "current_track": "Advanced Agentic AI",
            "recommended_roles": ["Junior AI Engineer"],
            "overall_recommendation": "Prepare a GitHub portfolio.",
        },
        "sources": [],
        "retrieval": {"retrieval_sufficient": True},
        "evidence": {
            "learner_profile_verified": True,
            "knowledge_retrieved": True,
            "retrieval_quality_evaluated": True,
            "active_vacancy_verified": False,
        },
    }


def test_advisor_flows_through_evaluator():
    """Verify that successful advisor output reaches the evaluator."""
    advisor_result = {
        "status": "success",
        "advice": {
            "learner_name": "Ahmed Youssef",
            "current_track": "Advanced Agentic AI",
            "recommended_roles": ["Junior AI Engineer"],
            "overall_recommendation": "Prepare a GitHub portfolio.",
        },
    }

    config = {"configurable": {"thread_id": "test-advisor-evaluator"}}

    with (
        patch(
            "src.agents.orchestrator.generate_career_advice",
            return_value=advisor_result,
        ),
        patch(
            "src.agents.orchestrator.evaluate_and_optimize",
            return_value=_successful_evaluator_result(
                "Optimized career recommendation."
            ),
        ) as evaluator,
    ):
        result = orchestrator.invoke(
            {
                "student_id": "SPR-1001",
                "request_type": "advisor",
                "messages": [HumanMessage(content="Give me career advice.")],
            },
            config,
        )

    assert evaluator.called
    assert result["final_response"] == "Optimized career recommendation."
    assert result["iteration_count"] == 1
    assert result["evaluation_passed"] is True


def test_assessor_flows_through_evaluator():
    """Verify that successful Python assessment reaches the evaluator."""
    assessor_result = {
        "status": "success",
        "execution": {
            "status": "success",
            "stdout": "20.0\n",
            "stderr": "",
            "exit_code": 0,
            "error": None,
        },
        "grading": {
            "score": 100,
            "feedback_points": ["Correct average calculation."],
            "pass_status": True,
        },
    }

    config = {"configurable": {"thread_id": "test-assessor-evaluator"}}

    with (
        patch(
            "src.agents.orchestrator.assess_python_code",
            return_value=assessor_result,
        ),
        patch(
            "src.agents.orchestrator.evaluate_and_optimize",
            return_value=_successful_evaluator_result(
                "Optimized Python assessment feedback."
            ),
        ) as evaluator,
    ):
        result = orchestrator.invoke(
            {
                "request_type": "assessor",
                "submitted_code": "print(20)",
                "assignment": "Print the average.",
                "messages": [HumanMessage(content="Assess my Python code.")],
            },
            config,
        )

    assert evaluator.called
    assert result["final_response"] == (
        "Optimized Python assessment feedback."
    )
    assert result["grading_result"]["grading"]["score"] == 100
    assert result["grading_result"]["execution"]["stdout"] == "20.0\n"


def test_failed_assessment_bypasses_evaluator():
    """Verify that failed execution is not sent for optimization."""
    assessor_result = {
        "status": "error",
        "message": "Unable to verify: Python execution failed.",
    }

    config = {"configurable": {"thread_id": "test-assessor-failure"}}

    with (
        patch(
            "src.agents.orchestrator.assess_python_code",
            return_value=assessor_result,
        ),
        patch(
            "src.agents.orchestrator.evaluate_and_optimize",
        ) as evaluator,
    ):
        result = orchestrator.invoke(
            {
                "request_type": "assessor",
                "submitted_code": "print(1 / 0)",
                "assignment": "Print a number.",
                "messages": [HumanMessage(content="Assess my Python code.")],
            },
            config,
        )

    evaluator.assert_not_called()
    assert result["grading_result"]["status"] == "error"
    assert "Unable to verify" in result["final_response"]


def test_hitl_pauses_and_resumes_when_reviewer_approves():
    """Verify an eligible Fawry recommendation pauses for approval and resumes."""
    config = {"configurable": {"thread_id": "test-hitl-approved"}}

    with (
        patch(
            "src.agents.orchestrator.generate_career_advice",
            return_value=_eligible_advisor_result(),
        ),
        patch(
            "src.agents.orchestrator.evaluate_and_optimize",
            return_value=_successful_evaluator_result(
                "Optimized career recommendation."
            ),
        ),
    ):
        paused_result = orchestrator.invoke(
            {
                "student_id": "SPR-1001",
                "request_type": "advisor",
                "messages": [HumanMessage(content="Review my career options.")],
            },
            config,
        )

        assert paused_result.get("__interrupt__")
        interrupt_payload = paused_result["__interrupt__"][0].value
        assert interrupt_payload["type"] == (
            "tier_1_placement_recommendation_review"
        )
        assert interrupt_payload["student_id"] == "SPR-1001"
        assert interrupt_payload["active_vacancy_verified"] is False

        resumed_result = orchestrator.invoke(
            Command(
                resume={
                    "approved": True,
                    "reviewer": "Test Reviewer",
                    "comments": "Recommendation reviewed.",
                }
            ),
            config,
        )

    assert resumed_result["human_approval_required"] is True
    assert resumed_result["human_approval"] is True
    assert resumed_result["approval_details"]["approved"] is True
    assert resumed_result["approval_details"]["reviewer"] == "Test Reviewer"
    assert "APPROVED" in resumed_result["final_response"]
    assert "no active vacancy" in resumed_result["final_response"]


def test_hitl_pauses_and_resumes_when_reviewer_rejects():
    """Verify a human rejection is recorded and the graph completes."""
    config = {"configurable": {"thread_id": "test-hitl-rejected"}}

    with (
        patch(
            "src.agents.orchestrator.generate_career_advice",
            return_value=_eligible_advisor_result(),
        ),
        patch(
            "src.agents.orchestrator.evaluate_and_optimize",
            return_value=_successful_evaluator_result(
                "Optimized career recommendation."
            ),
        ),
    ):
        paused_result = orchestrator.invoke(
            {
                "student_id": "SPR-1001",
                "request_type": "advisor",
                "messages": [HumanMessage(content="Review my career options.")],
            },
            config,
        )

        assert paused_result.get("__interrupt__")

        resumed_result = orchestrator.invoke(
            Command(
                resume={
                    "approved": False,
                    "reviewer": "Test Reviewer",
                    "comments": "Needs more evidence before referral.",
                }
            ),
            config,
        )

    assert resumed_result["human_approval_required"] is True
    assert resumed_result["human_approval"] is False
    assert resumed_result["approval_details"]["approved"] is False
    assert resumed_result["approval_details"]["comments"] == (
        "Needs more evidence before referral."
    )
    assert "REJECTED" in resumed_result["final_response"]


def test_hitl_rejects_unauthorized_reviewer():
    """Verify that approval from an unauthorized reviewer is rejected by governance."""
    orchestrator = build_orchestrator()
    config = {"configurable": {"thread_id": "test-unauth-reviewer"}}

    with (
        patch(
            "src.agents.orchestrator.generate_career_advice",
            return_value=_eligible_advisor_result(),
        ),
        patch(
            "src.agents.orchestrator.evaluate_and_optimize",
            return_value=_successful_evaluator_result("Career recommendation."),
        ),
    ):
        paused_result = orchestrator.invoke(
            {
                "student_id": "SPR-1001",
                "request_type": "advisor",
                "messages": [HumanMessage(content="Check placement.")],
            },
            config,
        )

        assert paused_result.get("__interrupt__")

        # Attempt approval by unauthorized user
        resumed_result = orchestrator.invoke(
            Command(
                resume={
                    "approved": True,
                    "reviewer": "unauthorized_external_user",
                    "comments": "Attempting unauthorized approval.",
                }
            ),
            config,
        )

    assert resumed_result["human_approval_required"] is True
    # Governance must have forced approved=False
    assert resumed_result["human_approval"] is False
    assert resumed_result["approval_details"]["approved"] is False
    assert resumed_result["approval_details"]["reviewer_authenticated"] is False
    assert "REJECTED (UNAUTHORIZED REVIEWER)" in resumed_result["final_response"]
