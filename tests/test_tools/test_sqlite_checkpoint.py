from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langgraph.types import Command

from src.agents.orchestrator import build_orchestrator
from src.tools.sqlite_checkpoint import SqliteSaver


def test_sqlite_saver_persists_state_across_instances(tmp_path):
    """Verify that state persists across separate SqliteSaver instances."""
    db_file = tmp_path / "checkpoints.sqlite3"

    # Instance 1: Run graph and write checkpoint
    saver1 = SqliteSaver(db_file)
    builder1 = StateGraph(dict)
    builder1.add_node("increment", lambda s: {"count": s.get("count", 0) + 1})
    builder1.set_entry_point("increment")
    builder1.set_finish_point("increment")
    graph1 = builder1.compile(checkpointer=saver1)

    config = {"configurable": {"thread_id": "session-101"}}
    res1 = graph1.invoke({"count": 10}, config=config)
    assert res1["count"] == 11

    # Instance 2: Brand new instance reading from the same SQLite file
    saver2 = SqliteSaver(db_file)
    builder2 = StateGraph(dict)
    builder2.add_node("increment", lambda s: {"count": s.get("count", 0) + 1})
    builder2.set_entry_point("increment")
    builder2.set_finish_point("increment")
    graph2 = builder2.compile(checkpointer=saver2)

    loaded_state = graph2.get_state(config)
    assert loaded_state is not None
    assert loaded_state.values["count"] == 11

    # Resume computation on thread in Instance 2
    res2 = graph2.invoke({"count": 20}, config=config)
    assert res2["count"] == 21


def test_sqlite_saver_handles_missing_thread(tmp_path):
    """Verify that querying a nonexistent thread returns empty/None state."""
    db_file = tmp_path / "checkpoints_empty.sqlite3"
    saver = SqliteSaver(db_file)
    builder = StateGraph(dict)
    builder.add_node("dummy", lambda s: s)
    builder.set_entry_point("dummy")
    builder.set_finish_point("dummy")
    graph = builder.compile(checkpointer=saver)

    missing_config = {"configurable": {"thread_id": "nonexistent-thread-999"}}
    state = graph.get_state(missing_config)
    assert state.values == {}


def test_sqlite_saver_concurrent_writes(tmp_path):
    """Verify that concurrent thread writes to the same SQLite database succeed without locking."""
    db_file = tmp_path / "checkpoints_concurrent.sqlite3"
    saver = SqliteSaver(db_file)

    builder = StateGraph(dict)
    builder.add_node("compute", lambda s: {"val": s.get("val", 0) * 2})
    builder.set_entry_point("compute")
    builder.set_finish_point("compute")
    graph = builder.compile(checkpointer=saver)

    def write_worker(thread_num):
        thread_id = f"worker-thread-{thread_num}"
        cfg = {"configurable": {"thread_id": thread_id}}
        return graph.invoke({"val": thread_num}, config=cfg)

    # 10 concurrent threads executing on the checkpointer
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(write_worker, range(1, 11)))

    for idx, r in enumerate(results, 1):
        assert r["val"] == idx * 2

    # Verify each thread's state was independently persisted
    saver2 = SqliteSaver(db_file)
    graph2 = builder.compile(checkpointer=saver2)
    for idx in range(1, 11):
        thread_id = f"worker-thread-{idx}"
        cfg = {"configurable": {"thread_id": thread_id}}
        saved = graph2.get_state(cfg)
        assert saved.values["val"] == idx * 2


def test_sqlite_saver_hitl_resumption_across_process_restart(tmp_path):
    """
    Verify durable persistence of Human-in-the-Loop review:
    1. Graph 1 runs until HITL interrupt and pauses.
    2. Simulated process termination (Graph 1 destroyed).
    3. Graph 2 initializes from disk, retrieves the paused state, and resumes to completion.
    """
    db_file = tmp_path / "checkpoints_hitl.sqlite3"
    thread_id = "durable-hitl-session"
    config = {"configurable": {"thread_id": thread_id}}

    eligible_advisor_result = {
        "status": "success",
        "student_id": "SYN-DEMO-01",
        "learner_profile": {
            "learner_id": "SYN-DEMO-01",
            "name": "Synthetic Demo",
            "bootcamp_track": "Advanced Agentic AI",
            "current_track": "Advanced Agentic AI",
            "preferred_employers": "Fawry",
            "readiness_status": "Ready for Placement",
        },
        "advice": {
            "learner_name": "Synthetic Demo",
            "current_track": "Advanced Agentic AI",
            "recommended_roles": ["AI Engineer"],
            "overall_recommendation": "Great portfolio.",
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

    evaluator_result = {
        "status": "success",
        "response": "High quality career advice.",
        "iteration_count": 1,
        "max_iterations": 3,
        "passed": True,
        "evaluation": {"score": 95, "passed": True, "feedback": ["Verified."]},
    }

    # Step 1: Run in Process 1
    saver1 = SqliteSaver(db_file)
    orchestrator1 = build_orchestrator(checkpointer=saver1)

    with patch("src.agents.orchestrator.generate_career_advice", return_value=eligible_advisor_result), \
         patch("src.agents.orchestrator.evaluate_and_optimize", return_value=evaluator_result):

        paused_result = orchestrator1.invoke(
            {
                "student_id": "SYN-DEMO-01",
                "request_type": "advisor",
                "messages": [HumanMessage(content="Recommend placement pathways.")],
            },
            config=config,
        )

        assert paused_result.get("__interrupt__")
        interrupt_val = paused_result["__interrupt__"][0].value
        assert interrupt_val["type"] == "tier_1_placement_recommendation_review"
        assert interrupt_val["student_id"] == "SYN-DEMO-01"

    # Step 2: Simulate process crash/exit
    del orchestrator1
    del saver1

    # Step 3: Brand new process instance reading from SQLite
    saver2 = SqliteSaver(db_file)
    orchestrator2 = build_orchestrator(checkpointer=saver2)

    # Verify state snapshot was durable across process boundary
    state_after_restart = orchestrator2.get_state(config)
    assert state_after_restart is not None
    assert "placement_review" in state_after_restart.next

    # Step 4: Resume workflow in the new process
    resume_command = Command(
        resume={
            "approved": True,
            "reviewer": "Director of Placement",
            "comments": "Approved after system restart verification.",
        }
    )
    resumed_result = orchestrator2.invoke(resume_command, config=config)

    assert resumed_result["human_approval"] is True
    assert resumed_result["approval_details"]["reviewer"] == "Director of Placement"
    assert "Approved after system restart verification." in resumed_result["approval_details"]["comments"]
    assert "Human placement recommendation review: APPROVED" in resumed_result["final_response"]
