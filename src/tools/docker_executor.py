import subprocess
import threading
import uuid
from typing import Any


import os

# Base image reference for reproducible sandbox builds (supports sha256 digest pinning via SPRINTS_BASE_IMAGE)
DEFAULT_SANDBOX_IMAGE = "sprints-python-sandbox:latest"
SANDBOX_IMAGE = os.getenv("SPRINTS_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE)
PINNED_BASE_IMAGE = os.getenv("SPRINTS_BASE_IMAGE", "python:3.10-slim")

EXECUTION_TIMEOUT = 5

MEMORY_LIMIT = "128m"
CPU_LIMIT = "0.5"
PID_LIMIT = "64"

# Maximum captured output per stream. Reader threads continue draining the
# pipes after the cap is reached so the child process cannot block on a full
# pipe. Only the first MAX_OUTPUT_BYTES from each stream are retained.
MAX_OUTPUT_BYTES = 64 * 1024
OUTPUT_READ_CHUNK = 8 * 1024


def _read_limited(stream: Any, limit: int) -> tuple[bytes, bool]:
    captured = bytearray()
    truncated = False

    while True:
        chunk = stream.read(OUTPUT_READ_CHUNK)
        if not chunk:
            break

        remaining = limit - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True

    return bytes(captured), truncated


def execute_in_docker(code: str) -> dict[str, Any]:
    """
    Execute submitted Python code inside a restricted Docker container prototype.

    PROTOTYPE SECURITY NOTICE:
        This execution environment is an experimental prototype designed for
        assignment evaluation with strict resource constraints. It is NOT a
        certified production-grade sandbox. Untrusted arbitrary code execution
        in production requires dedicated microVM isolation (e.g., gVisor, Kata,
        Firecracker), strict egress network firewalls, and independent security auditing.

    Resource and Isolation Controls:
        - Network disabled (--network none).
        - Read-only root filesystem (--read-only).
        - Non-root user (UID 10001).
        - Memory limit: 128 MB (--memory 128m).
        - CPU limit: 0.5 CPU (--cpus 0.5).
        - Process limit: 64 PIDs (--pids-limit 64).
        - All Linux capabilities dropped (--cap-drop ALL).
        - No privilege escalation (--security-opt no-new-privileges).
        - Bounded tmpfs writable storage (/tmp: 16 MB, noexec, nosuid).
        - Wall-clock timeout: 5 seconds with explicit container stopping and removal.
        - Bounded output capture: 64 KiB each for stdout and stderr, with continuous
          chunk draining to prevent pipe deadlocks.
        - Code passed via stdin pipe to avoid command-line escaping vulnerabilities.
    """

    if not isinstance(code, str) or not code.strip():
        return {
            "status": "error",
            "stdout": "",
            "stderr": "",
            "error": "Empty or invalid Python submission.",
            "exit_code": None,
            "output_truncated": False,
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
    ]

    process = None
    stdout_result: list[tuple[bytes, bool]] = []
    stderr_result: list[tuple[bytes, bool]] = []

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )

        try:
            process.stdin.write(code.encode("utf-8"))
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        stdout_thread = threading.Thread(
            target=lambda: stdout_result.append(
                _read_limited(process.stdout, MAX_OUTPUT_BYTES)
            ),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=lambda: stderr_result.append(
                _read_limited(process.stderr, MAX_OUTPUT_BYTES)
            ),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            exit_code = process.wait(timeout=EXECUTION_TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                subprocess.run(
                    ["docker", "stop", "--time", "0", container_name],
                    capture_output=True,
                    timeout=5,
                    check=False,
                    shell=False,
                )
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                    timeout=5,
                    check=False,
                    shell=False,
                )
            except Exception:
                pass
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=3)
            except Exception:
                pass
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)

            return {
                "status": "error",
                "stdout": "",
                "stderr": "",
                "error": (
                    f"Execution exceeded the {EXECUTION_TIMEOUT}-second "
                    "time limit."
                ),
                "exit_code": None,
                "output_truncated": False,
            }

        stdout_thread.join()
        stderr_thread.join()

        stdout_bytes, stdout_truncated = (
            stdout_result[0] if stdout_result else (b"", False)
        )
        stderr_bytes, stderr_truncated = (
            stderr_result[0] if stderr_result else (b"", False)
        )
        output_truncated = stdout_truncated or stderr_truncated

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        if stdout_truncated:
            stdout_text += "\n[stdout truncated at 65536 bytes]"
        if stderr_truncated:
            stderr_text += "\n[stderr truncated at 65536 bytes]"

        response = {
            "status": "success" if exit_code == 0 else "error",
            "stdout": stdout_text,
            "stderr": stderr_text,
            "exit_code": exit_code,
            "error": None,
            "output_truncated": output_truncated,
        }

        if exit_code != 0:
            response["error"] = (
                stderr_text.strip() or "Python execution failed."
            )

        return response

    except Exception as error:
        if process is not None and process.poll() is None:
            try:
                process.kill()
                process.wait(timeout=3)
            except Exception:
                pass
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=5,
                check=False,
                shell=False,
            )
        except Exception:
            pass
        return {
            "status": "error",
            "stdout": "",
            "stderr": "",
            "error": f"Unable to verify execution: {error}",
            "exit_code": None,
            "output_truncated": False,
        }
