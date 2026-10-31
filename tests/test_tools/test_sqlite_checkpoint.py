from pathlib import Path
import pytest
from langgraph.graph import StateGraph

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
