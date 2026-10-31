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
    assert result["status"] == "not_found" or (result["status"] == "success" and result["row_count"] == 0)
    assert result["data"] == []


def test_get_learner_by_id_with_invalid_id():
    """Verify that empty or whitespace IDs are rejected."""
    result1 = get_learner_by_id("")
    assert result1["status"] == "error"

    result2 = get_learner_by_id("   ")
    assert result2["status"] == "error"


def test_get_learner_by_id_sql_injection_union_blocked():
    """Verify parameterized query prevents SQL injection UNION payload."""
    payload = "SPR-1001' UNION SELECT 1,2,3,4,5,6,7,8 --"
    result = get_learner_by_id(payload)
    # The payload is treated literally as a learner_id and finds nothing
    assert result["data"] == []


def test_get_learner_by_id_sql_injection_or_always_true_blocked():
    """Verify parameterized query prevents ' OR '1'='1 bulk extraction."""
    payload = "' OR '1'='1"
    result = get_learner_by_id(payload)
    assert result["data"] == []


def test_get_learner_by_id_stacked_drop_blocked(setup_test_db):
    """Verify stacked SQL statement injection cannot execute destructive commands."""
    payload = "SPR-1001'; DROP TABLE learners; --"
    result = get_learner_by_id(payload)
    assert result["data"] == []

    # Verify table remains intact
    with sqlite3.connect(setup_test_db) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM learners")
        assert cur.fetchone()[0] == 14


def test_get_learner_by_id_tool():
    """Verify the parameterized tool wrapper for learner lookup."""
    res = get_learner_by_id_tool.invoke({"student_id": "SPR-1001"})
    assert res["status"] == "success"
    assert res["row_count"] == 1
    assert res["data"][0]["name"] == "Synthetic Learner 1"


def test_search_learners_by_track():
    """Verify parameterized domain lookup by track."""
    res = search_learners_by_track("Advanced Agentic AI", limit=3)
    assert res["status"] == "success"
    assert res["row_count"] == 3
    for row in res["data"]:
        assert row["bootcamp_track"] == "Advanced Agentic AI"


def test_search_learners_by_track_sql_injection_blocked():
    """Verify SQL injection payloads in track search return no rows safely."""
    payload = "' OR 1=1 --"
    res = search_learners_by_track(payload)
    assert res["status"] == "success"
    assert res["row_count"] == 0
    assert res["data"] == []


def test_search_learners_by_track_enforces_max_rows_limit():
    """Verify that requests for large row counts are strictly clamped to MAX_QUERY_ROWS."""
    res = search_learners_by_track("Advanced Agentic AI", limit=100)
    assert res["status"] == "success"
    assert res["row_count"] <= MAX_QUERY_ROWS
    assert len(res["data"]) <= MAX_QUERY_ROWS


def test_search_learners_by_track_column_allowlisting():
    """Verify that only explicitly permitted columns are returned from track search."""
    res = search_learners_by_track("Advanced Agentic AI", limit=2)
    assert res["status"] == "success"
    for row in res["data"]:
        assert set(row.keys()).issubset(ALLOWED_COLUMNS)


def test_search_learners_by_track_tool():
    """Verify the LangChain tool wrapper for track search."""
    res = search_learners_by_track_tool.invoke({"bootcamp_track": "Python Data Science"})
    assert res["status"] == "success"
    assert res["row_count"] > 0
    for row in res["data"]:
        assert row["bootcamp_track"] == "Python Data Science"


def test_get_track_aggregate_stats():
    """Verify safe aggregate metrics computation."""
    res = get_track_aggregate_stats()
    assert res["status"] == "success"
    assert res["data"]["learner_count"] == 14
    assert res["data"]["avg_python_score"] > 0


def test_get_track_aggregate_stats_enforces_minimum_cohort_size(setup_test_db):
    """Verify that small cohorts (< MIN_COHORT_SIZE) are masked for k-anonymity privacy."""
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


def test_get_track_aggregate_stats_tool():
    """Verify the LangChain tool wrapper for aggregate statistics."""
    res = get_track_aggregate_stats_tool.invoke({})
    assert res["status"] == "success"
    assert res["data"]["learner_count"] == 14
