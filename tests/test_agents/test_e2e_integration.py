from unittest.mock import patch
import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agents.orchestrator import build_orchestrator
from src.tools.db_queries import get_learner_by_id
from src.tools.sqlite_checkpoint import SqliteSaver
from src.utils.auth import (
    generate_caller_token,
    login_and_issue_token,
    verify_reviewer_auth,
)


def _mock_advisor_response():
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
            "recommended_roles": ["Junior AI Agent Engineer"],
            "overall_recommendation": "Strong multi-agent architecture skills.",
        },
        "sources": [{"source": "project_brief.md", "trust_level": "high"}],
        "retrieval": {"retrieval_sufficient": True},
        "evidence": {
            "learner_profile_verified": True,
            "knowledge_retrieved": True,
            "retrieval_quality_evaluated": True,
            "active_vacancy_verified": False,
        },
    }


def _mock_evaluator_response():
    return {
        "status": "success",
        "response": "Recommendation verified and approved by optimizer.",
        "iteration_count": 1,
        "max_iterations": 3,
        "passed": True,
        "evaluation": {
            "score": 96,
            "passed": True,
            "feedback": ["High alignment with track competencies."],
        },
    }


def test_e2e_pipeline_full_lifecycle(tmp_path):
    """
    End-to-End Integration Test:
    1. Caller token authentication & Learner DB lookup.
    2. Career Advisor Agent with RAG evidence.
    3. Evaluator-Optimizer loop validation.
    4. HITL Placement Review interruption & review request structure.
    5. Reviewer login & HMAC token issuance.
    6. Workflow resumption and SqliteSaver checkpoint persistence.
    """
    db_file = tmp_path / "e2e_checkpoints.sqlite3"
    saver = SqliteSaver(db_file)
    orchestrator = build_orchestrator(checkpointer=saver)

    # 1. Database caller authentication
    caller_token = generate_caller_token("orchestrator_agent", "advisor")
    learner_data = get_learner_by_id("SPR-1001", auth_token=caller_token)
    assert learner_data["status"] == "success"
    assert learner_data["row_count"] == 1
    assert learner_data["data"][0]["name"] == "Ahmed Youssef"

    # 2. Execute orchestrator workflow
    thread_id = "e2e-thread-integration-101"
    config = {"configurable": {"thread_id": thread_id}}

    with (
        patch("src.agents.orchestrator.generate_career_advice", return_value=_mock_advisor_response()),
        patch("src.agents.orchestrator.evaluate_and_optimize", return_value=_mock_evaluator_response()),
    ):
        orchestrator.invoke(
            {
                "student_id": "SPR-1001",
                "request_type": "advisor",
                "messages": [HumanMessage(content="Evaluate Ahmed for Fawry placement.")],
            },
            config,
        )

        # 3. Verify graph paused at Human-in-the-Loop review with proper payload
        paused_state = orchestrator.get_state(config)
        assert paused_state is not None
        assert "placement_review" in paused_state.next
        assert len(paused_state.tasks) > 0
        interrupt_val = paused_state.tasks[0].interrupts[0].value
        assert "Fawry" in interrupt_val["preferred_employers"]
        assert interrupt_val["student_id"] == "SPR-1001"

        # 4. Reviewer authenticates with institutional credentials to obtain signed session token
        login_success, msg, reviewer_token = login_and_issue_token(
            username="sarah_lead",
            password="SprintsAdvisor2026!",
        )
        assert login_success is True
        assert reviewer_token is not None

        # 5. Resume workflow with authenticated token
        resumed_state = orchestrator.invoke(
            Command(
                resume={
                    "approved": True,
                    "reviewer": "sarah_lead",
                    "auth_token": reviewer_token,
                    "comments": "E2E Verified: Portfolio and competencies meet Fawry partner bar.",
                }
            ),
            config,
        )

        # 6. Verify successful approval and audit trail
        assert resumed_state["human_approval"] is True
        assert resumed_state["approval_details"]["approved"] is True
        assert resumed_state["approval_details"]["reviewer_authenticated"] is True
        assert resumed_state["approval_details"]["reviewer"] == "sarah_lead"
        assert resumed_state["approval_details"]["audit_record"]["signature_verified"] is True

    # 7. Verify persistent SQLite checkpoint can be restored by a brand new instance
    restored_saver = SqliteSaver(db_file)
    restored_orchestrator = build_orchestrator(checkpointer=restored_saver)
    saved_state = restored_orchestrator.get_state(config)
    assert saved_state is not None
    assert saved_state.values["human_approval"] is True
    assert saved_state.values["student_id"] == "SPR-1001"


def test_e2e_pipeline_rejects_unauthorized_reviewer_credentials():
    """Verify that attempting to resume with invalid credentials or forged token fails safely."""
    orchestrator = build_orchestrator()
    config = {"configurable": {"thread_id": "e2e-thread-unauth-test"}}

    with (
        patch("src.agents.orchestrator.generate_career_advice", return_value=_mock_advisor_response()),
        patch("src.agents.orchestrator.evaluate_and_optimize", return_value=_mock_evaluator_response()),
    ):
        orchestrator.invoke(
            {"student_id": "SPR-1001", "request_type": "advisor", "messages": [HumanMessage(content="Eval")]},
            config,
        )

        # Reviewer attempts login with incorrect password
        login_success, _, bad_token = login_and_issue_token("sarah_lead", "WrongPassword123!")
        assert login_success is False
        assert bad_token is None

        # Attacker attempts to resume with forged token
        forged_token = "eyJhbGciOiJIUzI1NiJ9.deadbeef0123456789"
        res = orchestrator.invoke(
            Command(
                resume={
                    "approved": True,
                    "reviewer": "sarah_lead",
                    "auth_token": forged_token,
                    "comments": "Exploit attempt",
                }
            ),
            config,
        )
        assert res["human_approval"] is False
        assert res["approval_details"]["reviewer_authenticated"] is False
