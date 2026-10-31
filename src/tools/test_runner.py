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
import time
import select
import signal

# Bind safe primitive functions directly before untrusted code can monkey-patch them
_safe_fork = getattr(os, "fork", None)
_safe_waitpid = getattr(os, "waitpid", None)
_safe_pipe = getattr(os, "pipe", None)
_safe_close = getattr(os, "close", None)
_safe_write = getattr(os, "write", None)
_safe_read = getattr(os, "read", None)
_safe_kill = getattr(os, "kill", None)
_safe_exit = getattr(os, "_exit", None)
_safe_WIFEXITED = getattr(os, "WIFEXITED", None)
_safe_WEXITSTATUS = getattr(os, "WEXITSTATUS", None)
_safe_fdopen = getattr(os, "fdopen", None)
_safe_select = getattr(select, "select", None)
_safe_time = time.time
_safe_SIGKILL = getattr(signal, "SIGKILL", 9)

_safe_StringIO = io.StringIO
_safe_BaseException = BaseException
_safe_orig_stdout = sys.stdout
_safe_json_dumps = json.dumps
_safe_has_fork = callable(_safe_fork) and callable(_safe_waitpid) and callable(_safe_pipe) and callable(_safe_exit) and callable(_safe_select) and callable(_safe_read)

# Neutralize stack introspection vectors to prevent student code from walking/clearing parent frames
for _attr in ("_getframe", "_current_frames"):
    if hasattr(sys, _attr):
        try:
            delattr(sys, _attr)
        except Exception:
            try:
                setattr(sys, _attr, None)
            except Exception:
                pass

_code_str = {repr(code)}
_start_marker = {repr(start_marker)}
_end_marker = {repr(end_marker)}
_test_cases = {repr(test_cases)}

_results = []
_captured_stdout = ""
_compilation_error = None

if _safe_has_fork:
    # Dedicated Worker Process Isolation:
    # The parent harness process NEVER executes untrusted student code!
    _m_r_fd, _m_w_fd = _safe_pipe()
    _worker_pid = _safe_fork()
    if _worker_pid == 0:
        # INSIDE DEDICATED WORKER CHILD PROCESS
        _safe_close(_m_r_fd)
        try:
            # 1. Neutralize stack frame reflection and inspection
            for _attr in ("_getframe", "_current_frames"):
                if hasattr(sys, _attr):
                    try:
                        delattr(sys, _attr)
                    except Exception:
                        pass
            if hasattr(sys, "settrace"):
                try:
                    sys.settrace(None)
                except Exception:
                    pass
            if hasattr(sys, "setprofile"):
                try:
                    sys.setprofile(None)
                except Exception:
                    pass

            # 2. Block sensitive execution vector modules
            sys.modules["ctypes"] = None
            sys.modules["_ctypes"] = None

            # 3. Compile and execute student code inside worker
            _w_compiled = None
            _w_comp_err = None
            try:
                _w_compiled = compile(_code_str, "<learner_code>", "exec")
            except _safe_BaseException as _ce:
                _w_comp_err = f"{{type(_ce).__name__}}: {{_ce}}"

            _w_captured = ""
            _w_base_globals = {{"__builtins__": __builtins__}}

            if not _w_comp_err:
                _w_buf = _safe_StringIO()
                sys.stdout = _w_buf
                try:
                    exec(_w_compiled, _w_base_globals)
                except _safe_BaseException as _re:
                    _w_comp_err = f"{{type(_re).__name__}}: {{_re}}"
                finally:
                    sys.stdout = _safe_orig_stdout
                    _w_captured = _w_buf.getvalue()

            _w_base_globals["_captured_stdout"] = _w_captured

            # 4. Evaluate each assertion in an assertion child process
            _w_results = []
            if _w_comp_err:
                for _name, _ in _test_cases:
                    _w_results.append({{"name": _name, "status": "FAILED", "error": _w_comp_err}})
            else:
                for _name, _case_src in _test_cases:
                    _c_r_fd, _c_w_fd = _safe_pipe()
                    _c_pid = _safe_fork()
                    if _c_pid == 0:
                        _safe_close(_c_r_fd)
                        try:
                            exec(_case_src, _w_base_globals)
                            _safe_write(_c_w_fd, b"OK")
                            _safe_exit(0)
                        except AssertionError as _ae:
                            _msg = (str(_ae) or "Assertion failed").encode("utf-8", "replace")
                            _safe_write(_c_w_fd, b"FAIL:" + _msg)
                            _safe_exit(1)
                        except _safe_BaseException as _be:
                            _msg = f"{{type(_be).__name__}}: {{_be}}".encode("utf-8", "replace")
                            _safe_write(_c_w_fd, b"ERR:" + _msg)
                            _safe_exit(2)
                    else:
                        _safe_close(_c_w_fd)
                        # Concurrently drain assertion IPC pipe with timeout to prevent buffer deadlocks
                        _c_chunks = []
                        _c_total = 0
                        _c_MAX = 64 * 1024
                        _c_start = _safe_time()
                        _c_timeout = 2.0
                        while _c_total < _c_MAX:
                            _c_rem = max(0.01, _c_timeout - (_safe_time() - _c_start))
                            _c_rlist, _, _ = _safe_select([_c_r_fd], [], [], _c_rem)
                            if not _c_rlist:
                                if _safe_kill:
                                    try:
                                        _safe_kill(_c_pid, _safe_SIGKILL)
                                    except Exception:
                                        pass
                                break
                            try:
                                _c_chk = _safe_read(_c_r_fd, min(16384, _c_MAX - _c_total))
                            except Exception:
                                break
                            if not _c_chk:
                                break
                            _c_chunks.append(_c_chk)
                            _c_total += len(_c_chk)

                        _safe_close(_c_r_fd)
                        try:
                            _, _c_status = _safe_waitpid(_c_pid, 0)
                        except Exception:
                            _c_status = -1

                        _pipe_out = b"".join(_c_chunks).decode("utf-8", "replace")
                        _c_exit = _safe_WEXITSTATUS(_c_status) if (_safe_WIFEXITED and _safe_WIFEXITED(_c_status)) else -1
                        if _c_exit == 0 and _pipe_out == "OK":
                            _w_results.append({{"name": _name, "status": "PASSED", "error": ""}})
                        elif _pipe_out.startswith("FAIL:"):
                            _w_results.append({{"name": _name, "status": "FAILED", "error": _pipe_out[5:]}})
                        elif _pipe_out.startswith("ERR:"):
                            _w_results.append({{"name": _name, "status": "FAILED", "error": _pipe_out[4:]}})
                        else:
                            _w_results.append({{
                                "name": _name,
                                "status": "FAILED",
                                "error": f"Abnormal process exit (exit code {{_c_exit}})",
                            }})

            _w_payload = _safe_json_dumps({{
                "results": _w_results,
                "captured_stdout": _w_captured,
                "compilation_error": _w_comp_err,
            }}).encode("utf-8")
            _w_offset = 0
            while _w_offset < len(_w_payload):
                _written = _safe_write(_m_w_fd, _w_payload[_w_offset:_w_offset + 32768])
                if _written is None or _written <= 0:
                    break
                _w_offset += _written
            _safe_close(_m_w_fd)
            _safe_exit(0)
        except _safe_BaseException as _fatal:
            _safe_exit(3)
    else:
        # PARENT HARNESS PROCESS (Immune to student code tampering)
        _safe_close(_m_w_fd)
        # Concurrently drain worker IPC pipe with timeout before/alongside waitpid to prevent pipe deadlocks
        _w_chunks = []
        _w_total = 0
        _w_MAX = 512 * 1024
        _w_start = _safe_time()
        _w_timeout = 5.0
        while _w_total < _w_MAX:
            _w_rem = max(0.01, _w_timeout - (_safe_time() - _w_start))
            _w_rlist, _, _ = _safe_select([_m_r_fd], [], [], _w_rem)
            if not _w_rlist:
                if _safe_kill:
                    try:
                        _safe_kill(_worker_pid, _safe_SIGKILL)
                    except Exception:
                        pass
                break
            try:
                _w_chk = _safe_read(_m_r_fd, min(32768, _w_MAX - _w_total))
            except Exception:
                break
            if not _w_chk:
                break
            _w_chunks.append(_w_chk)
            _w_total += len(_w_chk)

        _safe_close(_m_r_fd)
        try:
            _, _w_status = _safe_waitpid(_worker_pid, 0)
        except Exception:
            _w_status = -1

        _raw_worker_out = b"".join(_w_chunks).decode("utf-8", "replace")
        _w_exit = _safe_WEXITSTATUS(_w_status) if (_safe_WIFEXITED and _safe_WIFEXITED(_w_status)) else -1
        _worker_data = None
        if _w_exit == 0 and _raw_worker_out:
            try:
                _worker_data = json.loads(_raw_worker_out)
            except Exception:
                pass

        if _worker_data:
            _results = _worker_data.get("results", [])
            _captured_stdout = _worker_data.get("captured_stdout", "")
            _compilation_error = _worker_data.get("compilation_error")
        else:
            _results = [
                {{"name": _name, "status": "FAILED", "error": f"Abnormal worker process exit (exit code {{_w_exit}})"}}
                for _name, _ in _test_cases
            ]
            _captured_stdout = ""
            _compilation_error = f"Abnormal worker termination ({{_w_exit}})"
else:
    # Non-fork fallback (Windows development / non-fork environments)
    try:
        _compiled_code = compile(_code_str, "<learner_code>", "exec")
    except _safe_BaseException as _ce:
        _compilation_error = f"{{type(_ce).__name__}}: {{_ce}}"

    _base_globals = {{"__builtins__": __builtins__}}
    if not _compilation_error:
        _io_buf = _safe_StringIO()
        sys.stdout = _io_buf
        try:
            exec(_compiled_code, _base_globals)
        except _safe_BaseException as _re:
            _compilation_error = f"{{type(_re).__name__}}: {{_re}}"
        finally:
            sys.stdout = _safe_orig_stdout
            _captured_stdout = _io_buf.getvalue()

    _base_globals["_captured_stdout"] = _captured_stdout
    if _compilation_error:
        for _name, _ in _test_cases:
            _results.append({{"name": _name, "status": "FAILED", "error": _compilation_error}})
    else:
        for _name, _case_src in _test_cases:
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
