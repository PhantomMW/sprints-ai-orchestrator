import json
import re
import uuid
from typing import Any

from src.tools.docker_executor import execute_in_docker


def parse_test_suite_into_cases(test_suite: str) -> list[tuple[str, str]]:
    """
    Parse a multiline test suite string into individual named test assertions.

    Each non-empty, non-comment line is treated as an isolated assertion.
    """
    cases = []
    lines = [
        line.strip()
        for line in test_suite.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for idx, line in enumerate(lines, 1):
        name = f"Test Case {idx}: {line[:40]}"
        cases.append((name, line))
    return cases


def run_automated_tests(
    code: str,
    test_suite: str = "",
    expected_output: str = "",
) -> dict[str, Any]:
    """
    Execute deterministic unit tests against a learner's Python submission with
    multi-layer OS process isolation and anti-tampering protection.

    Isolation & Verification Architecture:
        - Executes inside an isolated, non-root Docker container prototype.
        - Direct Primitive Binding: Critical OS and runtime functions (fork, waitpid,
          pipe, close, write, _exit) are bound to private function references before learner
          code is executed, preventing module-level monkey-patching from breaking the harness.
        - Single Code Execution: The learner's code is executed once to capture base stdout,
          stderr, and export its globals, avoiding redundant re-runs.
        - True OS Process Isolation: When running in Linux (Docker container), each test
          assertion executes within a dedicated child process (`os.fork()`), providing complete
          copy-on-write memory and process-level isolation between tests.
        - Abrupt Termination Trapping: If student code or an assertion calls `sys.exit()` or
          `os._exit()`, or causes a segfault, only the child process terminates; the runner harness
          survives and records the test failure.
        - Concurrently evaluates both unit assertions and expected stdout matching.
        - Unverified state: If neither assertions nor expected output are provided,
          returns an explicit 'unverified' status instead of fabricating false passes.

    Args:
        code: Python source code submitted by the learner.
        test_suite: Multiline string containing test assertions.
        expected_output: Optional expected standard output string.

    Returns:
        Structured test metrics dictionary including status, total, passed,
        failed, percentage, captured stdout, and individual test case outcomes.
    """
    if not isinstance(code, str) or not code.strip():
        return {
            "status": "error",
            "message": "Empty or invalid code submission.",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "percentage": 0.0,
            "results": [],
            "execution": {"status": "error", "error": "Empty submission"},
        }

    test_cases: list[tuple[str, str]] = []
    if test_suite.strip():
        test_cases.extend(parse_test_suite_into_cases(test_suite))

    if expected_output.strip():
        test_cases.append(
            (
                "Standard Output Match",
                f"assert _captured_stdout.strip() == {repr(expected_output.strip())}",
            )
        )

    # When no assertions or expected output are supplied, do not fabricate passes
    if not test_cases:
        smoke_res = execute_in_docker(code)
        is_clean = smoke_res.get("status") == "success" and smoke_res.get("exit_code") == 0
        return {
            "status": "unverified",
            "message": "No test assertions or expected output provided; correctness unverified.",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "percentage": 0.0,
            "results": [],
            "smoke_test_passed": is_clean,
            "execution": smoke_res,
        }

    session_token = uuid.uuid4().hex
    start_marker = f"__TEST_METRICS_{session_token}_START__"
    end_marker = f"__TEST_METRICS_{session_token}_END__"

    # Python harness script executed inside the Docker container
    harness = f"""
import os
import sys
import io
import json

# Bind safe primitive functions directly before untrusted code can monkey-patch them
_safe_fork = getattr(os, "fork", None)
_safe_waitpid = getattr(os, "waitpid", None)
_safe_pipe = getattr(os, "pipe", None)
_safe_close = getattr(os, "close", None)
_safe_write = getattr(os, "write", None)
_safe_exit = getattr(os, "_exit", None)
_safe_WIFEXITED = getattr(os, "WIFEXITED", None)
_safe_WEXITSTATUS = getattr(os, "WEXITSTATUS", None)
_safe_fdopen = getattr(os, "fdopen", None)

_safe_StringIO = io.StringIO
_safe_BaseException = BaseException
_safe_orig_stdout = sys.stdout
_safe_json_dumps = json.dumps
_safe_has_fork = callable(_safe_fork) and callable(_safe_waitpid) and callable(_safe_pipe) and callable(_safe_exit)

_code_str = {repr(code)}
_start_marker = {repr(start_marker)}
_end_marker = {repr(end_marker)}
_test_cases = {repr(test_cases)}

_compilation_error = None
_compiled_code = None

try:
    _compiled_code = compile(_code_str, "<learner_code>", "exec")
except _safe_BaseException as _ce:
    _compilation_error = f"{{type(_ce).__name__}}: {{_ce}}"

_captured_stdout = ""
_base_globals = {{"__builtins__": __builtins__}}

if not _compilation_error:
    _io_buf = _safe_StringIO()
    sys.stdout = _io_buf
    try:
        # Execute student code ONCE to capture stdout and populate namespace
        exec(_compiled_code, _base_globals)
    except _safe_BaseException as _re:
        _compilation_error = f"{{type(_re).__name__}}: {{_re}}"
    finally:
        sys.stdout = _safe_orig_stdout
        _captured_stdout = _io_buf.getvalue()

_base_globals["_captured_stdout"] = _captured_stdout

_results = []
if _compilation_error:
    for _name, _ in _test_cases:
        _results.append({{"name": _name, "status": "FAILED", "error": _compilation_error}})
else:
    for _name, _case_src in _test_cases:
        if _safe_has_fork:
            # True OS Process Isolation via fork (Linux Docker sandbox)
            _r_fd, _w_fd = _safe_pipe()
            _pid = _safe_fork()
            if _pid == 0:
                # Inside isolated child process
                _safe_close(_r_fd)
                try:
                    exec(_case_src, _base_globals)
                    _safe_write(_w_fd, b"OK")
                    _safe_exit(0)
                except AssertionError as _ae:
                    _msg = (str(_ae) or "Assertion failed").encode("utf-8", "replace")
                    _safe_write(_w_fd, b"FAIL:" + _msg)
                    _safe_exit(1)
                except _safe_BaseException as _be:
                    _msg = f"{{type(_be).__name__}}: {{_be}}".encode("utf-8", "replace")
                    _safe_write(_w_fd, b"ERR:" + _msg)
                    _safe_exit(2)
            else:
                # Inside parent harness process
                _safe_close(_w_fd)
                _, _status = _safe_waitpid(_pid, 0)
                with _safe_fdopen(_r_fd, "rb") as _rf:
                    _pipe_out = _rf.read().decode("utf-8", "replace")
                
                _exit_code = _safe_WEXITSTATUS(_status) if (_safe_WIFEXITED and _safe_WIFEXITED(_status)) else -1
                if _exit_code == 0 and _pipe_out == "OK":
                    _results.append({{"name": _name, "status": "PASSED", "error": ""}})
                elif _pipe_out.startswith("FAIL:"):
                    _results.append({{"name": _name, "status": "FAILED", "error": _pipe_out[5:]}})
                elif _pipe_out.startswith("ERR:"):
                    _results.append({{"name": _name, "status": "FAILED", "error": _pipe_out[4:]}})
                else:
                    _results.append({{
                        "name": _name,
                        "status": "FAILED",
                        "error": f"Abnormal process exit (exit code {{_exit_code}})",
                    }})
        else:
            # Fallback per-test namespace isolation for non-fork environments (Windows tests)
            _test_ns = dict(_base_globals)
            try:
                exec(_case_src, _test_ns)
                _results.append({{"name": _name, "status": "PASSED", "error": ""}})
            except AssertionError as _ae:
                _results.append({{"name": _name, "status": "FAILED", "error": str(_ae) or "Assertion failed"}})
            except _safe_BaseException as _be:
                _results.append({{"name": _name, "status": "FAILED", "error": f"{{type(_be).__name__}}: {{_be}}"}})

_passed = sum(1 for _r in _results if _r["status"] == "PASSED")
_total = len(_results)
_pct = round((_passed / _total * 100), 1) if _total > 0 else 0.0

# Emit output using safe primitive stdout writer
_output_json = _safe_json_dumps({{
    "total": _total,
    "passed": _passed,
    "failed": _total - _passed,
    "percentage": _pct,
    "results": _results,
    "captured_stdout": _captured_stdout,
    "compilation_error": _compilation_error,
}})

_safe_orig_stdout.write(f"\\n{{_start_marker}}\\n{{_output_json}}\\n{{_end_marker}}\\n")
"""

    docker_res = execute_in_docker(harness)
    if docker_res["status"] != "success":
        return {
            "status": "error",
            "message": docker_res.get("error", "Execution failed"),
            "total": len(test_cases),
            "passed": 0,
            "failed": len(test_cases),
            "percentage": 0.0,
            "results": [
                {
                    "name": "Docker Sandbox",
                    "status": "FAILED",
                    "error": docker_res.get("error", ""),
                }
            ],
            "execution": docker_res,
        }

    stdout = docker_res.get("stdout", "")
    pattern = re.escape(start_marker) + r"\s*(.*?)\s*" + re.escape(end_marker)
    match = re.search(pattern, stdout, re.DOTALL)
    if not match:
        return {
            "status": "error",
            "message": "Unable to parse test runner metrics from sandbox output.",
            "total": len(test_cases),
            "passed": 0,
            "failed": len(test_cases),
            "percentage": 0.0,
            "results": [
                {
                    "name": "Metrics Extraction",
                    "status": "FAILED",
                    "error": "Failed to extract test metrics from sandbox output.",
                }
            ],
            "execution": docker_res,
        }

    try:
        metrics = json.loads(match.group(1))
        # Synthesize a clean execution dictionary from the runner's captured stdout
        comp_err = metrics.get("compilation_error")
        exec_dict = {
            "status": "error" if comp_err else "success",
            "stdout": metrics.get("captured_stdout", ""),
            "stderr": comp_err or "",
            "exit_code": 1 if comp_err else 0,
            "output_truncated": docker_res.get("output_truncated", False),
        }
        return {
            "status": "success",
            "total": metrics["total"],
            "passed": metrics["passed"],
            "failed": metrics["failed"],
            "percentage": metrics["percentage"],
            "results": metrics["results"],
            "captured_stdout": metrics.get("captured_stdout", ""),
            "execution": exec_dict,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Metrics decode error: {e}",
            "total": len(test_cases),
            "passed": 0,
            "failed": len(test_cases),
            "percentage": 0.0,
            "results": [],
            "execution": docker_res,
        }
