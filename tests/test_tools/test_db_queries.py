import pytest

from src.tools.db_queries import (
    ALLOWED_COLUMNS,
    MAX_QUERY_ROWS,
    get_learner_by_id,
    learner_sql_tool,
    query_learner_database,
)


def test_query_learner_database_enforces_row_limit():
    """Verify that broad queries cannot return all database rows and set truncated=True."""
    result = query_learner_database("SELECT * FROM learners")

    assert result["status"] == "success"
    assert result["row_count"] == MAX_QUERY_ROWS
    assert result["truncated"] is True
    assert len(result["data"]) == MAX_QUERY_ROWS


def test_query_learner_database_filters_unauthorized_columns():
    """Verify that only explicitly permitted columns are returned."""
    result = query_learner_database("SELECT learner_id, name, bootcamp_track FROM learners")

    assert result["status"] == "success"
    assert result["row_count"] > 0
    for row in result["data"]:
        for key in row.keys():
            assert key in ALLOWED_COLUMNS


def test_query_learner_database_rejects_non_select():
    """Verify that non-SELECT statements are rejected."""
    bad_queries = [
        "INSERT INTO learners (learner_id) VALUES ('TEST')",
        "UPDATE learners SET name = 'Hacked'",
        "DELETE FROM learners",
        "DROP TABLE learners",
    ]
    for q in bad_queries:
        result = query_learner_database(q)
        assert result["status"] == "error"
        assert "Only SELECT queries are allowed" in result["message"]


def test_query_learner_database_rejects_multiple_statements():
    """Verify that multi-statement queries containing semicolons are blocked."""
    result = query_learner_database("SELECT * FROM learners; DROP TABLE learners;")
    assert result["status"] == "error"
    assert "Multiple SQL statements are not permitted" in result["message"]


def test_query_learner_database_rejects_sql_comments():
    """Verify that comment evasion syntax is blocked."""
    result1 = query_learner_database("SELECT * FROM learners -- bypass")
    assert result1["status"] == "error"
    assert "comments are not permitted" in result1["message"]

    result2 = query_learner_database("SELECT * FROM learners /* bypass */")
    assert result2["status"] == "error"
    assert "comments are not permitted" in result2["message"]


def test_query_learner_database_blocks_system_tables():
    """Verify that internal SQLite system tables cannot be inspected."""
    result = query_learner_database("SELECT * FROM sqlite_master")
    assert result["status"] == "error"
    assert "restricted" in result["message"].lower() or "not permitted" in result["message"].lower()


def test_query_learner_database_blocks_unauthorized_tables():
    """Verify that tables other than 'learners' are blocked."""
    result = query_learner_database("SELECT * FROM users")
    assert result["status"] == "error"
    assert "not permitted" in result["message"]


def test_get_learner_by_id_returns_targeted_learner_only():
    """Verify that get_learner_by_id returns exactly one learner without cross-learner exposure."""
    # First get an existing ID
    preview = query_learner_database("SELECT learner_id FROM learners")
    target_id = preview["data"][0]["learner_id"]

    result = get_learner_by_id(target_id)
    assert result["status"] == "success"
    assert result["row_count"] == 1
    assert len(result["data"]) == 1
    learner = result["data"][0]
    assert learner["learner_id"] == target_id
    assert set(learner.keys()).issubset(ALLOWED_COLUMNS)


def test_get_learner_by_id_with_nonexistent_id():
    """Verify that non-existent IDs return zero rows safely."""
    result = get_learner_by_id("NON_EXISTENT_ID_999999")
    assert result["status"] == "success"
    assert result["row_count"] == 0
    assert result["data"] == []


def test_get_learner_by_id_with_invalid_id():
    """Verify that empty or invalid IDs are rejected."""
    result1 = get_learner_by_id("")
    assert result1["status"] == "error"

    result2 = get_learner_by_id("   ")
    assert result2["status"] == "error"


def test_learner_sql_tool_invokes_query_learner_database():
    """Verify that the LangChain tool wrapper enforces row limits."""
    res = learner_sql_tool.invoke({"sql_query": "SELECT * FROM learners"})
    assert res["status"] == "success"
    assert res["row_count"] == MAX_QUERY_ROWS
    assert res["truncated"] is True
