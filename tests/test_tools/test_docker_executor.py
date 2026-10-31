from io import BytesIO
import shutil
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.tools.docker_executor import (
    EXECUTION_TIMEOUT,
    MAX_OUTPUT_BYTES,
    _read_limited,
    execute_in_docker,
    get_sandbox_containment_mode,
)


def _docker_is_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        res = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=3,
        )
        return res.returncode == 0
    except Exception:
        return False


DOCKER_AVAILABLE = _docker_is_available()
requires_docker = pytest.mark.skipif(
    not DOCKER_AVAILABLE,
    reason="Docker daemon is not available in this environment",
)


# ==========================================
# Unit Tests (Pure Python / Stream Buffers)
# ==========================================

def test_read_limited_keeps_only_configured_bytes():
    payload = b"x" * (MAX_OUTPUT_BYTES + 100)

    captured, truncated = _read_limited(BytesIO(payload), MAX_OUTPUT_BYTES)

    assert len(captured) == MAX_OUTPUT_BYTES
    assert truncated is True


def test_read_limited_reports_untruncated_stream():
    payload = b"assessment output"

    captured, truncated = _read_limited(BytesIO(payload), MAX_OUTPUT_BYTES)

    assert captured == payload
    assert truncated is False


def test_empty_submission_is_rejected_without_docker():
    result = execute_in_docker("  ")

    assert result["status"] == "error"
    assert result["stdout"] == ""
    assert result["stderr"] == ""
    assert result["output_truncated"] is False


def test_non_string_submission_is_rejected_without_docker():
    result = execute_in_docker(None)  # type: ignore

    assert result["status"] == "error"
    assert "Empty or invalid Python submission" in result["error"]
    assert result["exit_code"] is None


# ==========================================
# Lifecycle & Failure Handling Tests
# ==========================================

def test_timeout_triggers_docker_stop_and_cleanup():
    """Verify that TimeoutExpired invokes docker stop and docker rm to clean up container."""
    with patch("src.tools.docker_executor.subprocess.Popen") as mock_popen, \
         patch("src.tools.docker_executor.subprocess.run") as mock_run:

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = BytesIO(b"")
        mock_process.stderr = BytesIO(b"")
        # First call to wait() raises TimeoutExpired; subsequent calls in cleanup return 0
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="docker run", timeout=EXECUTION_TIMEOUT),
            0,
        ]
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        result = execute_in_docker("import time; time.sleep(10)")

        assert result["status"] == "error"
        assert f"exceeded the {EXECUTION_TIMEOUT}-second time limit" in result["error"]
        assert result["exit_code"] is None

        # Verify docker stop and docker rm were called
        called_cmds = [call.args[0] for call in mock_run.call_args_list if call.args]
        has_stop = any("stop" in cmd for cmd in called_cmds)
        has_rm = any("rm" in cmd for cmd in called_cmds)
        assert has_stop, "docker stop should be called on timeout"
        assert has_rm, "docker rm should be called on timeout"


def test_popen_failure_is_handled_gracefully():
    """Verify that unexpected subprocess errors return a structured error response."""
    with patch("src.tools.docker_executor.subprocess.Popen", side_effect=OSError("Docker binary missing")):
        result = execute_in_docker("print('hello')")

        assert result["status"] == "error"
        assert "Unable to verify execution" in result["error"]
        assert result["exit_code"] is None


# ==========================================
# Full Docker Runtime & Security Tests
# ==========================================

@requires_docker
def test_docker_successful_execution():
    """Verify that valid code executes inside Docker and returns standard output."""
    result = execute_in_docker("print('Docker Sandbox Active')")

    assert result["status"] == "success"
    assert "Docker Sandbox Active" in result["stdout"]
    assert result["stderr"] == ""
    assert result["exit_code"] == 0
    assert result["output_truncated"] is False
    assert result["error"] is None


@requires_docker
def test_docker_execution_failure_nonzero_exit():
    """Verify that unhandled exceptions produce non-zero exit codes and capture stderr."""
    result = execute_in_docker("raise ValueError('Specific assessment error')")

    assert result["status"] == "error"
    assert result["exit_code"] != 0
    assert "ValueError" in result["stderr"]
    assert "Specific assessment error" in result["error"]


@requires_docker
def test_docker_syntax_error_failure():
    """Verify that Python syntax errors are safely captured."""
    result = execute_in_docker("def bad_syntax(")

    assert result["status"] == "error"
    assert result["exit_code"] != 0
    assert "SyntaxError" in result["stderr"]


@requires_docker
def test_docker_runtime_timeout_enforcement():
    """Verify that a running container exceeding EXECUTION_TIMEOUT is terminated."""
    result = execute_in_docker("import time; time.sleep(8)")

    assert result["status"] == "error"
    assert f"exceeded the {EXECUTION_TIMEOUT}-second time limit" in result["error"]
    assert result["exit_code"] is None
    assert result["output_truncated"] is False


@requires_docker
def test_docker_massive_stdout_truncation():
    """Verify that output exceeding 64 KiB is safely truncated and flagged."""
    code = "import sys; sys.stdout.write('A' * 100000)"
    result = execute_in_docker(code)

    assert result["status"] == "success"
    assert result["output_truncated"] is True
    assert "[stdout truncated at 65536 bytes]" in result["stdout"]
    # Captured text should be bounded around MAX_OUTPUT_BYTES + marker
    assert len(result["stdout"].encode("utf-8")) <= MAX_OUTPUT_BYTES + 200


@requires_docker
def test_docker_massive_stderr_truncation():
    """Verify that error output exceeding 64 KiB is safely truncated and flagged."""
    code = "import sys; sys.stderr.write('E' * 100000); sys.exit(1)"
    result = execute_in_docker(code)

    assert result["status"] == "error"
    assert result["output_truncated"] is True
    assert "[stderr truncated at 65536 bytes]" in result["stderr"]
    assert len(result["stderr"].encode("utf-8")) <= MAX_OUTPUT_BYTES + 200


@requires_docker
def test_docker_storage_read_only_restriction():
    """Verify that root filesystem is read-only and unauthorized writes fail."""
    code = "open('/root_test_file.txt', 'w').write('data')"
    result = execute_in_docker(code)

    assert result["status"] == "error"
    assert any(err in result["stderr"] for err in ["Read-only file system", "Permission denied", "Errno 30", "Errno 13"])


@requires_docker
def test_docker_network_isolation():
    """Verify that container network access is disabled (--network none)."""
    code = "import socket; socket.create_connection(('8.8.8.8', 53), timeout=1)"
    result = execute_in_docker(code)

    assert result["status"] == "error"
    assert any(err in result["stderr"] for err in ["Network is unreachable", "Errno 101", "TimeoutError", "OSError"])

def test_sandbox_containment_mode_defaults_and_overrides(monkeypatch):
    """Verify inspection function reports correct containment mode for runc, gVisor, and Kata."""
    from src.tools.docker_executor import get_sandbox_containment_mode

    # Default runc mode
    monkeypatch.delenv("SPRINTS_DOCKER_RUNTIME", raising=False)
    info_default = get_sandbox_containment_mode()
    assert info_default["containment_mode"] == "container_runc"
    assert info_default["is_microvm"] is False

    # gVisor (runsc) MicroVM mode
    monkeypatch.setenv("SPRINTS_DOCKER_RUNTIME", "runsc")
    info_gvisor = get_sandbox_containment_mode()
    assert info_gvisor["containment_mode"] == "microvm_gvisor"
    assert info_gvisor["is_microvm"] is True
    assert "gVisor" in info_gvisor["description"]

    # Kata MicroVM mode
    monkeypatch.setenv("SPRINTS_DOCKER_RUNTIME", "kata-qemu")
    info_kata = get_sandbox_containment_mode()
    assert info_kata["containment_mode"] == "microvm_kata"
    assert info_kata["is_microvm"] is True


def test_docker_command_includes_runtime_flag_when_configured(monkeypatch):
    """Verify execute_in_docker passes --runtime when SPRINTS_DOCKER_RUNTIME is set."""
    monkeypatch.setenv("SPRINTS_DOCKER_RUNTIME", "runsc")
    with patch("src.tools.docker_executor.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = BytesIO(b"")
        mock_proc.stderr = BytesIO(b"")
        mock_proc.wait.return_value = 0
        mock_proc.poll.return_value = 0
        mock_popen.return_value = mock_proc

        execute_in_docker("print('test')")
        cmd_called = mock_popen.call_args[0][0]
        assert "--runtime" in cmd_called
        idx = cmd_called.index("--runtime")
        assert cmd_called[idx + 1] == "runsc"


def test_docker_runtime_blocks_option_injection_and_invalid_names(monkeypatch):
    """Verify adversarial runtime injection attempts (--privileged, semicolons, shell metacharacters) are rejected."""
    # Attempt flag injection via SPRINTS_DOCKER_RUNTIME
    monkeypatch.setenv("SPRINTS_DOCKER_RUNTIME", "--privileged")
    res1 = execute_in_docker("print('adversarial')")
    assert res1["status"] == "error"
    assert "Security violation" in res1["error"]
    assert "illegal container runtime parameter" in res1["error"]

    info1 = get_sandbox_containment_mode()
    assert info1["runtime"] == "invalid_sanitized"

    # Attempt shell command chaining injection
    monkeypatch.setenv("SPRINTS_DOCKER_RUNTIME", "runsc; rm -rf /")
    res2 = execute_in_docker("print('adversarial')")
    assert res2["status"] == "error"
    assert "Security violation" in res2["error"]
