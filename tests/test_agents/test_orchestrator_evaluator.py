from unittest.mock import patch

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agents.orchestrator import build_orchestrator, orchestrator
from src.utils.auth import generate_reviewer_token


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

        token = generate_reviewer_token("sarah_lead", "placement_approver")
        resumed_result = orchestrator.invoke(
            Command(
                resume={
                    "approved": True,
                    "reviewer": "sarah_lead",
                    "auth_token": token,
                    "comments": "Recommendation reviewed.",
                }
            ),
            config,
        )

    assert resumed_result["human_approval_required"] is True
    assert resumed_result["human_approval"] is True
    assert resumed_result["approval_details"]["approved"] is True
    assert resumed_result["approval_details"]["reviewer"] == "sarah_lead"
    assert resumed_result["approval_details"]["reviewer_authenticated"] is True
    assert resumed_result["approval_details"]["audit_record"]["signature_verified"] is True
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


def test_hitl_rejects_tampered_token_signature():
    """Verify that an approval with a forged or tampered HMAC signature is rejected."""
    orchestrator = build_orchestrator()
    config = {"configurable": {"thread_id": "test-tampered-token"}}

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
        orchestrator.invoke(
            {
                "student_id": "SPR-1001",
                "request_type": "advisor",
                "messages": [HumanMessage(content="Check placement.")],
            },
            config,
        )

        valid_token = generate_reviewer_token("sarah_lead", "placement_approver")
        b64_part, _ = valid_token.split(".")
        forged_token = f"{b64_part}.0123456789abcdef0123456789abcdef"

        resumed_result = orchestrator.invoke(
            Command(
                resume={
                    "approved": True,
                    "reviewer": "sarah_lead",
                    "auth_token": forged_token,
                    "comments": "Attempt with forged signature.",
                }
            ),
            config,
        )

    assert resumed_result["human_approval"] is False
    assert resumed_result["approval_details"]["approved"] is False
    assert resumed_result["approval_details"]["reviewer_authenticated"] is False
    assert "tampered" in resumed_result["approval_details"]["auth_reason"].lower()


def test_hitl_rejects_replay_attack():
    """Verify that replaying the exact same token nonce is detected and rejected."""
    orchestrator = build_orchestrator()
    token = generate_reviewer_token("sarah_lead", "placement_approver")

    # First use: valid
    config1 = {"configurable": {"thread_id": "test-replay-1"}}
    with (
        patch("src.agents.orchestrator.generate_career_advice", return_value=_eligible_advisor_result()),
        patch("src.agents.orchestrator.evaluate_and_optimize", return_value=_successful_evaluator_result("Rec.")),
    ):
        orchestrator.invoke({"student_id": "SPR-1001", "request_type": "advisor", "messages": [HumanMessage(content="A")]}, config1)
        res1 = orchestrator.invoke(Command(resume={"approved": True, "auth_token": token}), config1)
    assert res1["human_approval"] is True

    # Second use with identical token (replay attempt)
    config2 = {"configurable": {"thread_id": "test-replay-2"}}
    with (
        patch("src.agents.orchestrator.generate_career_advice", return_value=_eligible_advisor_result()),
        patch("src.agents.orchestrator.evaluate_and_optimize", return_value=_successful_evaluator_result("Rec.")),
    ):
        orchestrator.invoke({"student_id": "SPR-1001", "request_type": "advisor", "messages": [HumanMessage(content="B")]}, config2)
        res2 = orchestrator.invoke(Command(resume={"approved": True, "auth_token": token}), config2)
    assert res2["human_approval"] is False
    assert "replay" in res2["approval_details"]["auth_reason"].lower()

def test_advisor_node_passes_caller_token_in_production(tmp_path, monkeypatch):
    """Verify that orchestrator passes a verified caller token so learner DB lookup succeeds in production."""
    import sqlite3
    monkeypatch.setenv("SPRINTS_ENV", "production")
    monkeypatch.setenv("HITL_AUTH_SECRET", "prod-secret-for-advisor-token-test-123456789")
    monkeypatch.setenv("DB_CALLER_AUTH_SECRET", "prod-db-caller-secret-for-advisor-test-123456789")

    test_db = tmp_path / "prod_learners.sqlite3"
    with sqlite3.connect(test_db) as conn:
        conn.execute("""
            CREATE TABLE learners (
                learner_id TEXT PRIMARY KEY,
                name TEXT,
                bootcamp_track TEXT,
                completed_modules INTEGER,
                python_assignment_score INTEGER,
                target_salary_egp INTEGER,
                preferred_employers TEXT,
                readiness_status TEXT
            )
        """)
        conn.execute(
            "INSERT INTO learners VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("SPR-PROD-01", "Production Learner", "Advanced Agentic AI", 10, 95, 30000, "Fawry", "Ready for Placement")
        )
    monkeypatch.setattr("src.tools.db_queries.DB_PATH", test_db)

    from src.agents.orchestrator import advisor_node
    with patch("src.agents.advisor_agent.agentic_retrieve_career_knowledge", return_value={"status": "success", "documents": [{"source": "brief.md", "trust_level": "high", "content": "Req"}], "final_query": "q", "retrieval_attempts": 1, "retry_count": 0, "retrieval_sufficient": True, "assessment": {}, "retrieval_trace": []}):
        with patch("src.agents.advisor_agent.create_advisor_agent") as mock_create_adv:
            from pydantic import BaseModel
            class FakeAdvice(BaseModel):
                learner_name: str = "Production Learner"
                current_track: str = "Advanced Agentic AI"
                recommended_roles: list[str] = ["AI Engineer"]
                technical_skill_gaps: list[str] = []
                learning_roadmap: list[str] = ["Step 1"]
                target_salary_analysis: str = "Realistic"
                overall_recommendation: str = "Portfolio verified"
            mock_adv_inst = mock_create_adv.return_value
            mock_adv_inst.invoke.return_value = FakeAdvice()

            res = advisor_node({"student_id": "SPR-PROD-01"})
            assert res["career_advice"]["status"] == "success"
            assert res["career_advice"]["student_id"] == "SPR-PROD-01"
            assert res["career_advice"]["learner_profile"]["name"] == "Production Learner"


def test_advisor_agent_direct_call_without_token_rejected_in_production(monkeypatch):
    """Verify that calling advisor_agent directly without a caller token in production is strictly rejected."""
    monkeypatch.setenv("SPRINTS_ENV", "production")
    monkeypatch.setenv("DB_CALLER_AUTH_SECRET", "prod-db-caller-secret-for-advisor-test-123456789")

    from src.agents.advisor_agent import generate_career_advice, get_learner_profile
    res = generate_career_advice("SPR-PROD-01", caller_token=None)
    assert res["status"] == "error"
    assert "caller token must be provided by the trusted orchestration layer" in res["message"].lower()

    profile_res = get_learner_profile("SPR-PROD-01", caller_token=None)
    assert profile_res["status"] == "error"
    assert "caller token is mandatory" in profile_res["message"].lower()
