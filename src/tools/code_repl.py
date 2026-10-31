from typing import Any

from langchain.tools import tool

from src.tools.docker_executor import execute_in_docker


def execute_python_code(code: str) -> dict[str, Any]:
    """
    Execute learner Python code inside an isolated Docker container.

    The Docker executor applies resource and security restrictions.
    """

    return execute_in_docker(code)


@tool
def python_repl_tool(code: str) -> dict[str, Any]:
    """
    Execute Python code submitted by a learner using Docker isolation.

    Returns execution status, stdout, stderr, and error details.
    """

    return execute_python_code(code)