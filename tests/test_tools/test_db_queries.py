import sqlite3
import pytest

from src.tools.db_queries import (
    ALLOWED_COLUMNS,
    MAX_QUERY_ROWS,
    MIN_COHORT_SIZE,
    get_learner_by_id,
    get_learner_by_id_tool,
    get_track_aggregate_stats,
    get_track_aggregate_stats_tool,
    query_learner_database,
    search_learners_by_track,
    search_learners_by_track_tool,
)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """
    Provide an isolated synthetic SQLite database for tests.
    Ensures zero reliance on uncommitted production data and 100% CI reproducibility.
    """
    test_db = tmp_path / "test_sprints_learners.sqlite3"
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
        for i in range(1, 15):
            conn.execute(
                """
                INSERT INTO learners VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"SPR-{1000 + i}",
                    f"Synthetic Learner {i}",
                    "Advanced Agentic AI" if i % 2 == 0 else "Python Data Science",
                    i,
                    70 + (i * 2),
                    20000 + (i * 1000),
                    "Fawry, Swvl" if i % 2 == 0 else "Vodafone",
                    "Ready for Placement" if i > 5 else "In Progress",
                ),
            )
    monkeypatch.setattr("src.tools.db_queries.DB_PATH", test_db)
    yield test_db


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
    result = get_learner_by_id("SPR-1002")
    assert result["status"] == "success"
    assert result["row_count"] == 1
    assert len(result["data"]) == 1
    learner = result["data"][0]
    assert learner["learner_id"] == "SPR-1002"
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


def test_get_track_aggregate_stats_enforces_minimum_cohort_size(setup_test_db):
    """Verify that small cohorts (< MIN_COHORT_SIZE) are masked for differential privacy."""
    # Insert exactly 1 learner in a rare track (below MIN_COHORT_SIZE = 3)
    with sqlite3.connect(setup_test_db) as conn:
        conn.execute(
            "INSERT INTO learners VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("SPR-9999", "Solo Student", "Rare Quantum Track", 1, 99, 50000, "None", "In Progress"),
        )

    res_small = get_track_aggregate_stats("Rare Quantum Track")
    assert res_small["status"] == "error"
    assert res_small["data"]["privacy_restricted"] is True
    assert "minimum privacy threshold" in res_small["message"]

    # Cohort with 7 learners satisfies privacy threshold
    res_large = get_track_aggregate_stats("Advanced Agentic AI")
    assert res_large["status"] == "success"
    assert res_large["data"]["privacy_restricted"] is False
    assert res_large["data"]["learner_count"] >= MIN_COHORT_SIZE


def test_search_learners_by_track():
    """Verify parameterized domain lookup by track."""
    res = search_learners_by_track("Advanced Agentic AI", limit=3)
    assert res["status"] == "success"
    assert res["row_count"] == 3
    for row in res["data"]:
        assert row["bootcamp_track"] == "Advanced Agentic AI"


def test_get_track_aggregate_stats():
    """Verify safe aggregate metrics computation."""
    res = get_track_aggregate_stats()
    assert res["status"] == "success"
    assert res["data"]["learner_count"] == 14
    assert res["data"]["avg_python_score"] > 0


def test_get_learner_by_id_tool():
    """Verify the parameterized tool wrapper for learner lookup."""
    res = get_learner_by_id_tool.invoke({"student_id": "SPR-1001"})
    assert res["status"] == "success"
    assert res["row_count"] == 1
    assert res["data"][0]["name"] == "Synthetic Learner 1"
