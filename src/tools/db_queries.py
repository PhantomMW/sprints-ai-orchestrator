import os
import sqlite3
from pathlib import Path
from typing import Any

from langchain.tools import tool
from src.utils.auth import verify_caller_token


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


def _authorize_caller(requester_role: str, auth_token: str | None = None) -> tuple[bool, str]:
    """Verify caller authorization using cryptographic tokens in production or role allowlists."""
    app_env = os.getenv("SPRINTS_ENV", os.getenv("ENV", "development")).strip().lower()

    if auth_token:
        valid, reason, payload = verify_caller_token(auth_token)
        if not valid:
            return False, f"Unauthorized access: Invalid caller token ({reason})"
        role = str(payload.get("role", "")).strip().casefold()
        if role not in AUTHORIZED_QUERY_ROLES:
            return False, f"Unauthorized access: Token role '{role}' lacks database query privilege."
        return True, f"Authorized via token for caller '{payload.get('caller_id')}' (role: {role})"

    if app_env in ("production", "prod", "staging"):
        return False, "Unauthorized access: Cryptographic auth_token is required to query learner PII in production."

    clean_role = str(requester_role).strip().casefold()
    if clean_role not in AUTHORIZED_QUERY_ROLES:
        return False, f"Unauthorized access: Caller role '{requester_role}' is not permitted to query learner PII."
    return True, f"Authorized for role '{clean_role}'"

# Role-based access control for accessing learner PII
AUTHORIZED_QUERY_ROLES = {
    "advisor",
    "career_advisor",
    "instructor",
    "admin",
    "system",
}


def _ensure_db_exists() -> bool:
    """Check if the configured SQLite database file exists on disk."""
    return DB_PATH.exists()


# ==============================================================================
# Safe Parameterized Domain Operations (Strictly No Arbitrary SQL Execution)
# ==============================================================================

def get_learner_by_id(student_id: str, requester_role: str = "advisor", auth_token: str | None = None) -> dict[str, Any]:
    """
    Retrieve an individual learner profile using a safe, parameterized SQL query.

    Privacy and Security Controls:
        - Role-based authorization: Rejects unauthorized callers (e.g. public or unauthenticated students).
        - Uses strictly parameterized queries (? placeholder).
        - Enforces targeted, single-learner isolation: returns exactly one record.
        - Never executes arbitrary SQL strings or broad SELECT * scans.
        - Database connection opened in read-only mode (mode=ro).

    Args:
        student_id: Learner identifier (e.g. 'SYN-DEMO-01').
        requester_role: Calling agent/user role (must be in AUTHORIZED_QUERY_ROLES).

    Returns:
        Dictionary with status 'success' and learner data in a bounded list.
    """
    authorized, auth_msg = _authorize_caller(requester_role=requester_role, auth_token=auth_token)
    if not authorized:
        return {
            "status": "error",
            "message": auth_msg,
            "row_count": 0,
            "data": [],
        }

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


def search_learners_by_track(
    bootcamp_track: str,
    limit: int = MAX_QUERY_ROWS,
    requester_role: str = "advisor",
    auth_token: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve learners enrolled in a specific bootcamp track using a parameterized query.

    Privacy and Security Controls:
        - Role-based authorization: Rejects unauthorized callers.
        - Uses strictly parameterized queries (? placeholder).
        - Capped at MAX_QUERY_ROWS (5) to prevent bulk extraction.
        - Column projection restricted to verified ALLOWED_COLUMNS.
        - Database opened in read-only mode (mode=ro).

    Args:
        bootcamp_track: Track name (e.g. 'Advanced Agentic AI').
        limit: Maximum rows to return (capped at MAX_QUERY_ROWS).
        requester_role: Calling agent/user role (must be in AUTHORIZED_QUERY_ROLES).

    Returns:
        Bounded list of learners in the specified track.
    """
    authorized, auth_msg = _authorize_caller(requester_role=requester_role, auth_token=auth_token)
    if not authorized:
        return {
            "status": "error",
            "message": auth_msg,
            "row_count": 0,
            "data": [],
        }

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


def get_track_aggregate_stats(
    bootcamp_track: str | None = None,
    requester_role: str = "advisor",
    auth_token: str | None = None,
) -> dict[str, Any]:
    """
    Compute aggregate performance metrics across a track or all tracks.

    Privacy & Security Controls (k-Anonymity Cohort Threshold & Caller Authorization):
        - Role-based authorization & cryptographic token verification: Unauthorized callers
          or unsigned requests in production are strictly rejected to prevent inference attacks.
        - Enforces MIN_COHORT_SIZE = 3: If fewer than 3 learners exist in a cohort,
          individual statistics are withheld (privacy_restricted=True) to prevent
          re-identification of individuals in sparse groups (k-Anonymity Cohort Suppression).
        - Only returns aggregated averages and counts; no row-level records are exposed.
        - Database opened in read-only mode (mode=ro).

    Args:
        bootcamp_track: Optional track filter.
        requester_role: Calling agent/user role (must be in AUTHORIZED_QUERY_ROLES).
        auth_token: Cryptographic HMAC caller token. Mandatory in production.

    Returns:
        Dictionary containing cohort count and average scores, or privacy restriction.
    """
    authorized, auth_msg = _authorize_caller(requester_role=requester_role, auth_token=auth_token)
    if not authorized:
        return {
            "status": "error",
            "message": auth_msg,
            "data": {},
        }

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
                            f"Aggregation suppressed: Cohort size is below the privacy threshold "
                            f"(k={MIN_COHORT_SIZE}). Individual re-identification protection active."
                        ),
                        "data": {
                            "privacy_restricted": True,
                            "min_cohort_size": MIN_COHORT_SIZE,
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
                            f"Aggregation suppressed: Cohort size is below the privacy threshold "
                            f"(k={MIN_COHORT_SIZE}). Individual re-identification protection active."
                        ),
                        "data": {
                            "privacy_restricted": True,
                            "min_cohort_size": MIN_COHORT_SIZE,
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
def get_learner_by_id_tool(student_id: str, auth_token: str | None = None) -> dict[str, Any]:
    """
    Safely retrieve a single verified learner profile by unique ID.
    Enforces cryptographic caller authorization and single-record isolation.
    """
    return get_learner_by_id(student_id, auth_token=auth_token)


@tool
def search_learners_by_track_tool(
    bootcamp_track: str,
    limit: int = MAX_QUERY_ROWS,
    auth_token: str | None = None,
) -> dict[str, Any]:
    """
    Safely retrieve learners enrolled in a specific bootcamp track.
    Enforces cryptographic caller authorization and bounded row limits.
    """
    return search_learners_by_track(bootcamp_track, limit=limit, auth_token=auth_token)


@tool
def get_track_aggregate_stats_tool(
    bootcamp_track: str | None = None,
    auth_token: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve aggregate learner metrics (averages, counts) with minimum cohort suppression (k=3).
    Enforces cryptographic caller authorization to prevent inference attacks.
    """
    return get_track_aggregate_stats(bootcamp_track=bootcamp_track, auth_token=auth_token)
