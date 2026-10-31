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
    Execute deterministic unit tests against a learner's Python submission.

    Isolation & Validation Controls:
        - Executes inside an isolated, non-root Docker container prototype.
        - Randomized session token delimits test metrics from arbitrary student output.
        - Per-test namespace isolation: each test case executes against a clean,
          independent global namespace preventing state pollution between assertions.
        - Catches BaseException (including SystemExit) to prevent student code from
          aborting the test harness prematurely.
        - Safe reference binding protects json and sys channels from student tampering.
        - Concurrently evaluates both unit assertions and expected stdout matching
          when both are provided.
        - Unverified state: if neither assertions nor expected output are provided,
          returns an explicit 'unverified' status instead of fabricating false passes.

    Args:
        code: Python source code submitted by the learner.
        test_suite: Multiline string containing test assertions.
        expected_output: Optional expected standard output string.

    Returns:
        Structured test metrics dictionary including status, total, passed,
        failed, percentage, and individual test case outcomes.
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
import sys
import io
import json

# Bind safe primitives before executing untrusted student code
_safe_sys = sys
_safe_json_dumps = json.dumps
_safe_StringIO = io.StringIO
_safe_BaseException = BaseException
_safe_orig_stdout = sys.stdout

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
if not _compilation_error:
    _io_buf = _safe_StringIO()
    _safe_sys.stdout = _io_buf
    try:
        _smoke_ns = {{"__builtins__": __builtins__}}
        exec(_compiled_code, _smoke_ns)
    except _safe_BaseException as _re:
        _compilation_error = f"{{type(_re).__name__}}: {{_re}}"
    finally:
        _safe_sys.stdout = _safe_orig_stdout
        _captured_stdout = _io_buf.getvalue()

_results = []
if _compilation_error:
    for _name, _ in _test_cases:
        _results.append({{"name": _name, "status": "FAILED", "error": _compilation_error}})
else:
    for _name, _case_src in _test_cases:
        # Provide clean, independent namespace per test case (true isolation)
        _test_ns = {{
            "__builtins__": __builtins__,
            "_captured_stdout": _captured_stdout,
        }}
        try:
            exec(_compiled_code, _test_ns)
            exec(_case_src, _test_ns)
            _results.append({{"name": _name, "status": "PASSED", "error": ""}})
        except AssertionError as _ae:
            _err_msg = str(_ae) or "Assertion failed"
            _results.append({{"name": _name, "status": "FAILED", "error": _err_msg}})
        except _safe_BaseException as _be:
            _results.append({{"name": _name, "status": "FAILED", "error": f"{{type(_be).__name__}}: {{_be}}"}})

_passed = sum(1 for _r in _results if _r["status"] == "PASSED")
_total = len(_results)
_pct = round((_passed / _total * 100), 1) if _total > 0 else 0.0

# Emit output using pre-bound safe stdout writer
_output_json = _safe_json_dumps({{
    "total": _total,
    "passed": _passed,
    "failed": _total - _passed,
    "percentage": _pct,
    "results": _results
}})

_safe_sys.stdout.write(f"\\n{{_start_marker}}\\n{{_output_json}}\\n{{_end_marker}}\\n")
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
        return {
            "status": "success",
            "total": metrics["total"],
            "passed": metrics["passed"],
            "failed": metrics["failed"],
            "percentage": metrics["percentage"],
            "results": metrics["results"],
            "execution": docker_res,
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
