import subprocess
import uuid
from typing import Any


SANDBOX_IMAGE = "sprints-python-sandbox:latest"

EXECUTION_TIMEOUT = 5

MEMORY_LIMIT = "128m"
CPU_LIMIT = "0.5"
PID_LIMIT = "64"


def execute_in_docker(code: str) -> dict[str, Any]:
    """
    Execute untrusted Python code inside an isolated Docker container.

    Security controls:
        - No network access.
        - Read-only root filesystem.
        - Non-root user.
        - Memory and CPU limits.
        - Process limit.
        - No Linux capabilities.
        - No privilege escalation.
        - Temporary writable storage only.
        - Automatic container removal.
        - Execution timeout.
    """

    if not isinstance(code, str) or not code.strip():
        return {
            "status": "error",
            "stdout": "",
            "stderr": "",
            "error": "Empty or invalid Python submission.",
            "exit_code": None,
        }

    container_name = f"sprints-assessment-{uuid.uuid4().hex[:12]}"

    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        "--read-only",
        "--memory",
        MEMORY_LIMIT,
        "--cpus",
        CPU_LIMIT,
        "--pids-limit",
        str(PID_LIMIT),
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        "--user",
        "10001:10001",
        "--interactive",
        SANDBOX_IMAGE,
        "python",
        "-I",
        "-B",
        "-c",
        code,
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT,
            check=False,
            shell=False,
        )

        response = {
            "status": (
                "success" if result.returncode == 0 else "error"
            ),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "error": None,
        }

        if result.returncode != 0:
            response["error"] = (
                result.stderr.strip()
                or "Python execution failed."
            )

        return response

    except subprocess.TimeoutExpired:
        # Stop the named container if it is still running.
        subprocess.run(
            ["docker", "stop", "--time", "0", container_name],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
            shell=False,
        )

        return {
            "status": "error",
            "stdout": "",
            "stderr": "",
            "error": (
                f"Execution exceeded the "
                f"{EXECUTION_TIMEOUT}-second time limit."
            ),
            "exit_code": None,
        }

    except Exception as error:
        return {
            "status": "error",
            "stdout": "",
            "stderr": "",
            "error": f"Unable to verify execution: {error}",
            "exit_code": None,
        }