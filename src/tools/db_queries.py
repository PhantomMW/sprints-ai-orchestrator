import sqlite3
from pathlib import Path
from typing import Any

from langchain.tools import tool


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "sprints_learners.sqlite3"

MAX_QUERY_ROWS = 5

MIN_COHORT_SIZE = 3

ALLOWED_TABLES = {"learners"}

ALLOWED_COLUMNS = {
    "learner_id",
    "name",
    "bootcamp_track",
    "completed_modules",
    "python_assignment_score",
    "target_salary_egp",
    "preferred_employers",
    "readiness_status",
}


def _ensure_db_exists() -> bool:
    """Check if the configured SQLite database file exists on disk."""
    return DB_PATH.exists()


# ==============================================================================
# Safe Parameterized Domain Operations (Strictly No Arbitrary SQL Execution)
# ==============================================================================

def get_learner_by_id(student_id: str) -> dict[str, Any]:
    """
    Retrieve an individual learner profile using a safe, parameterized SQL query.

    Privacy and Security Controls:
        - Uses strictly parameterized queries (? placeholder).
        - Enforces targeted, single-learner isolation: returns exactly one record.
        - Never executes arbitrary SQL strings or broad SELECT * scans.
        - Database connection opened in read-only mode (mode=ro).

    Args:
        student_id: Learner identifier (e.g. 'SYN-DEMO-01').

    Returns:
        Dictionary with status 'success' and learner data in a bounded list.
    """
    if not isinstance(student_id, str) or not student_id.strip():
        return {
            "status": "error",
            "message": "Invalid or empty learner identifier.",
            "row_count": 0,
            "data": [],
        }

    if not _ensure_db_exists():
        return {
            "status": "error",
            "message": "Learner database file is not available.",
            "row_count": 0,
            "data": [],
        }

    clean_id = student_id.strip()

    query = """
        SELECT
            learner_id,
            name,
            bootcamp_track,
            completed_modules,
            python_assignment_score,
            target_salary_egp,
            preferred_employers,
            readiness_status
        FROM learners
        WHERE learner_id = ?
        LIMIT 1
    """

    try:
        with sqlite3.connect(
            f"file:{DB_PATH.as_posix()}?mode=ro",
            uri=True
        ) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (clean_id,))
            rows = [dict(row) for row in cursor.fetchall()]

            sanitized_rows = []
            for row in rows:
                sanitized_rows.append(
                    {k: v for k, v in row.items() if k in ALLOWED_COLUMNS}
                )

            return {
                "status": "success",
                "row_count": len(sanitized_rows),
                "data": sanitized_rows,
            }

    except sqlite3.Error as error:
        return {
            "status": "error",
            "message": f"Unable to verify: {error}",
            "row_count": 0,
            "data": [],
        }


def search_learners_by_track(bootcamp_track: str, limit: int = MAX_QUERY_ROWS) -> dict[str, Any]:
    """
    Retrieve learners enrolled in a specific bootcamp track using a parameterized query.

    Privacy and Security Controls:
        - Uses strictly parameterized queries (? placeholder).
        - Capped at MAX_QUERY_ROWS (5) to prevent bulk extraction.
        - Column projection restricted to verified ALLOWED_COLUMNS.
        - Database opened in read-only mode (mode=ro).

    Args:
        bootcamp_track: Track name (e.g. 'Advanced Agentic AI').
        limit: Maximum rows to return (capped at MAX_QUERY_ROWS).

    Returns:
        Bounded list of learners in the specified track.
    """
    if not isinstance(bootcamp_track, str) or not bootcamp_track.strip():
        return {
            "status": "error",
            "message": "Invalid or empty bootcamp track.",
            "row_count": 0,
            "data": [],
        }

    if not _ensure_db_exists():
        return {
            "status": "error",
            "message": "Learner database file is not available.",
            "row_count": 0,
            "data": [],
        }

    effective_limit = min(max(1, limit), MAX_QUERY_ROWS)
    clean_track = bootcamp_track.strip()

    query = """
        SELECT
            learner_id,
            name,
            bootcamp_track,
            completed_modules,
            python_assignment_score,
            target_salary_egp,
            preferred_employers,
            readiness_status
        FROM learners
        WHERE LOWER(bootcamp_track) = LOWER(?)
        LIMIT ?
    """

    try:
        with sqlite3.connect(
            f"file:{DB_PATH.as_posix()}?mode=ro",
            uri=True
        ) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (clean_track, effective_limit))
            rows = [dict(row) for row in cursor.fetchall()]

            sanitized_rows = []
            for row in rows:
                sanitized_rows.append(
                    {k: v for k, v in row.items() if k in ALLOWED_COLUMNS}
                )

            return {
                "status": "success",
                "row_count": len(sanitized_rows),
                "data": sanitized_rows,
            }

    except sqlite3.Error as error:
        return {
            "status": "error",
            "message": f"Unable to verify: {error}",
            "row_count": 0,
            "data": [],
        }


def get_track_aggregate_stats(bootcamp_track: str | None = None) -> dict[str, Any]:
    """
    Compute aggregate performance metrics across a track or all tracks.

    Privacy Controls (k-Anonymity Cohort Threshold):
        - Enforces MIN_COHORT_SIZE = 3: If fewer than 3 learners exist in a cohort,
          individual statistics are withheld (privacy_restricted=True) to prevent
          re-identification of individuals in sparse groups.
        - Only returns aggregated averages and counts; no row-level records are exposed.
        - Database opened in read-only mode (mode=ro).

    Args:
        bootcamp_track: Optional track filter.

    Returns:
        Dictionary containing cohort count and average scores, or privacy restriction.
    """
    if not _ensure_db_exists():
        return {
            "status": "error",
            "message": "Learner database file is not available.",
            "data": {},
        }

    try:
        with sqlite3.connect(
            f"file:{DB_PATH.as_posix()}?mode=ro",
            uri=True
        ) as conn:
            conn.row_factory = sqlite3.Row

            if bootcamp_track:
                count_sql = "SELECT COUNT(*) as count FROM learners WHERE LOWER(bootcamp_track) = LOWER(?)"
                cursor = conn.execute(count_sql, (bootcamp_track.strip(),))
                cohort_count = cursor.fetchone()["count"]

                if cohort_count < MIN_COHORT_SIZE:
                    return {
                        "status": "error",
                        "message": (
                            f"Cohort size ({cohort_count}) is below minimum privacy threshold "
                            f"({MIN_COHORT_SIZE}). Aggregation restricted to prevent identification."
                        ),
                        "data": {
                            "privacy_restricted": True,
                            "min_cohort_size": MIN_COHORT_SIZE,
                            "cohort_count": cohort_count,
                        },
                    }

                query = """
                    SELECT
                        bootcamp_track,
                        COUNT(*) as learner_count,
                        ROUND(AVG(python_assignment_score), 2) as avg_python_score,
                        ROUND(AVG(completed_modules), 2) as avg_completed_modules
                    FROM learners
                    WHERE LOWER(bootcamp_track) = LOWER(?)
                    GROUP BY bootcamp_track
                """
                cursor = conn.execute(query, (bootcamp_track.strip(),))
            else:
                count_sql = "SELECT COUNT(*) as count FROM learners"
                cursor = conn.execute(count_sql)
                cohort_count = cursor.fetchone()["count"]

                if cohort_count < MIN_COHORT_SIZE:
                    return {
                        "status": "error",
                        "message": (
                            f"Total cohort size ({cohort_count}) is below minimum privacy threshold "
                            f"({MIN_COHORT_SIZE}). Aggregation restricted."
                        ),
                        "data": {
                            "privacy_restricted": True,
                            "min_cohort_size": MIN_COHORT_SIZE,
                            "cohort_count": cohort_count,
                        },
                    }

                query = """
                    SELECT
                        COUNT(*) as learner_count,
                        ROUND(AVG(python_assignment_score), 2) as avg_python_score,
                        ROUND(AVG(completed_modules), 2) as avg_completed_modules
                    FROM learners
                """
                cursor = conn.execute(query)

            row = cursor.fetchone()
            data_dict = dict(row) if row else {}
            data_dict["privacy_restricted"] = False
            data_dict["min_cohort_size"] = MIN_COHORT_SIZE
            return {
                "status": "success",
                "data": data_dict,
            }
    except sqlite3.Error as error:
        return {
            "status": "error",
            "message": f"Unable to verify: {error}",
            "data": {},
        }


# ==============================================================================
# LangChain Tool Bindings for Agent Integration
# ==============================================================================

@tool
def get_learner_by_id_tool(student_id: str) -> dict[str, Any]:
    """
    Safely retrieve a single verified learner profile by unique ID.

    Recommended parameterized tool for learner profile lookups.
    """
    return get_learner_by_id(student_id)


@tool
def search_learners_by_track_tool(bootcamp_track: str) -> dict[str, Any]:
    """
    Safely retrieve learners enrolled in a specific bootcamp track.

    Parameterized tool ensuring safe query execution without raw SQL strings.
    """
    return search_learners_by_track(bootcamp_track)


@tool
def get_track_aggregate_stats_tool() -> dict[str, Any]:
    """
    Retrieve aggregate learner metrics (average scores, counts) without exposing individual rows.
    """
    return get_track_aggregate_stats()
