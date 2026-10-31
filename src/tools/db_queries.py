import re
import sqlite3
from pathlib import Path
from typing import Any

from langchain.tools import tool


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "sprints_learners.sqlite3"

MAX_QUERY_ROWS = 5

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

FORBIDDEN_SQL_PATTERNS = [
    (r";", "Multiple SQL statements are not permitted."),
    (r"--", "SQL single-line comments are not permitted."),
    (r"/\*", "SQL multi-line comments are not permitted."),
    (r"\bsqlite_master\b", "Access to SQLite system metadata is restricted."),
    (r"\bsqlite_schema\b", "Access to SQLite system metadata is restricted."),
    (r"\bsqlite_temp_master\b", "Access to SQLite system metadata is restricted."),
    (r"\bpragma\b", "PRAGMA statements are not permitted."),
    (r"\battach\b", "ATTACH statements are not permitted."),
]


def query_learner_database(sql_query: str) -> dict[str, Any]:
    """
    Execute a bounded, read-only SQL query against the Sprints learner database.

    Privacy and Security Controls:
        - Only SELECT statements are permitted.
        - Multiple statements and comment sequences are forbidden.
        - System metadata tables (sqlite_master, pragma) are blocked.
        - Queries are restricted to the allowed 'learners' table.
        - Maximum returned rows capped at MAX_QUERY_ROWS (5) to prevent bulk export.
        - Disallowed or unauthorized columns are strictly filtered from output.
        - For single learner lookups, get_learner_by_id() must be used instead.

    Args:
        sql_query: SQL SELECT query.

    Returns:
        Dictionary containing bounded query results or an error message.
    """
    if not isinstance(sql_query, str) or not sql_query.strip():
        return {
            "status": "error",
            "message": "Empty or invalid SQL query.",
            "row_count": 0,
            "data": [],
        }

    normalized_query = sql_query.strip()
    lowered_query = normalized_query.lower()

    if not lowered_query.startswith("select"):
        return {
            "status": "error",
            "message": "Only SELECT queries are allowed.",
            "row_count": 0,
            "data": [],
        }

    for pattern, reason in FORBIDDEN_SQL_PATTERNS:
        if re.search(pattern, lowered_query):
            return {
                "status": "error",
                "message": f"Query rejected: {reason}",
                "row_count": 0,
                "data": [],
            }

    from_match = re.search(r"\bfrom\s+([a-zA-Z0-9_]+)", lowered_query)
    if from_match:
        table_name = from_match.group(1)
        if table_name not in ALLOWED_TABLES:
            return {
                "status": "error",
                "message": f"Access to table '{table_name}' is not permitted. Allowed tables: {list(ALLOWED_TABLES)}.",
                "row_count": 0,
                "data": [],
            }

    try:
        with sqlite3.connect(
            f"file:{DB_PATH.as_posix()}?mode=ro",
            uri=True
        ) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(normalized_query)

            raw_rows = cursor.fetchmany(MAX_QUERY_ROWS + 1)
            truncated = len(raw_rows) > MAX_QUERY_ROWS
            selected_rows = raw_rows[:MAX_QUERY_ROWS]

            sanitized_rows = []
            for row in selected_rows:
                row_dict = dict(row)
                filtered = {k: v for k, v in row_dict.items() if k in ALLOWED_COLUMNS}
                sanitized_rows.append(filtered)

            return {
                "status": "success",
                "row_count": len(sanitized_rows),
                "max_rows": MAX_QUERY_ROWS,
                "truncated": truncated,
                "data": sanitized_rows,
            }

    except sqlite3.Error as error:
        return {
            "status": "error",
            "message": f"Unable to verify: {error}",
            "row_count": 0,
            "data": [],
        }


@tool
def learner_sql_tool(sql_query: str) -> dict[str, Any]:
    """
    Retrieve learner information from the Sprints SQLite database.

    Privacy and Security Notice:
        - Results are strictly capped at 5 rows to prevent bulk data exfiltration.
        - Only SELECT queries targeting the 'learners' table are permitted.
        - Disallowed or unauthorized columns are filtered.
        - For individual learner recommendations, use get_learner_by_id()
          to guarantee learner data privacy and prevent cross-learner exposure.
    """
    return query_learner_database(sql_query)


def get_learner_by_id(student_id: str) -> dict[str, Any]:
    """
    Retrieve an individual learner profile using a safe, parameterized SQL query.

    This is the primary and recommended method for retrieving learner data.
    It guarantees strict learner isolation, parameterized safety, and
    prevents unintended access or cross-learner data leakage.

    Args:
        student_id: Unique learner identifier.

    Returns:
        Verified learner profile dictionary or an error response.
    """
    if not isinstance(student_id, str) or not student_id.strip():
        return {
            "status": "error",
            "message": "Invalid or empty learner identifier.",
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
