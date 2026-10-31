import sqlite3
from pathlib import Path
from typing import Any

from langchain.tools import tool


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "sprints_learners.sqlite3"


def query_learner_database(sql_query: str) -> dict[str, Any]:
    """
    Execute a read-only SQL query against the Sprints learner database.

    Retrieves learner information such as completed modules,
    Python assignment scores, target salaries, preferred employers,
    and readiness status.

    Only SELECT statements are allowed.

    Args:
        sql_query: SQL SELECT query.

    Returns:
        Dictionary containing query results or an error message.
    """

    normalized_query = sql_query.strip().lower()

    if not normalized_query.startswith("select"):
        return {
            "status": "error",
            "message": "Only SELECT queries are allowed."
        }

    try:
        with sqlite3.connect(
            f"file:{DB_PATH.as_posix()}?mode=ro",
            uri=True
        ) as conn:

            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql_query)

            rows = [dict(row) for row in cursor.fetchall()]

            return {
                "status": "success",
                "row_count": len(rows),
                "data": rows
            }

    except sqlite3.Error as error:
        return {
            "status": "error",
            "message": f"Unable to verify: {error}"
        }


@tool
def learner_sql_tool(sql_query: str) -> dict[str, Any]:
    """
    Retrieve learner information from the Sprints SQLite database.

    Use this tool to answer questions about learner progress,
    completed modules, Python scores, target salaries,
    preferred employers, and readiness status.

    The tool accepts read-only SQL SELECT queries.
    """
    return query_learner_database(sql_query)

def get_learner_by_id(student_id: str) -> dict[str, Any]:
    """
    Retrieve a learner profile using a parameterized SQL query.

    Args:
        student_id: Unique learner identifier.

    Returns:
        Verified learner profile or an error response.
    """

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
    """

    try:
        with sqlite3.connect(
            f"file:{DB_PATH.as_posix()}?mode=ro",
            uri=True
        ) as conn:

            conn.row_factory = sqlite3.Row

            cursor = conn.execute(query, (student_id,))
            rows = [dict(row) for row in cursor.fetchall()]

            return {
                "status": "success",
                "row_count": len(rows),
                "data": rows
            }

    except sqlite3.Error as error:
        return {
            "status": "error",
            "message": f"Unable to verify: {error}"
        }