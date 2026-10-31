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
    Execute deterministic unit tests against a learner's Python submission with anti-tampering.

    Security & Anti-Tampering Controls:
        - Executes inside an isolated, non-root Docker container prototype.
        - Randomized session token prevents learner code from spoofing test metrics delimiters.
        - Isolated namespace copies for each test case prevent inter-test state pollution.
        - Strict timeout and resource limits.
        - Explicit failure enforcement: if the code fails to compile or times out,
          zero passed tests are awarded.

    Args:
        code: Python source code submitted by the learner.
        test_suite: Multiline string containing test assertions.
        expected_output: Optional expected standard output string.

    Returns:
        Structured test metrics dictionary including total, passed, failed,
        percentage, and individual test case outcomes.
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

    if expected_output.strip() and not test_suite.strip():
        test_cases.append(
            (
                "Standard Output Match",
                f"assert _captured_stdout.strip() == {repr(expected_output.strip())}",
            )
        )

    if not test_cases:
        test_cases.append(("Execution Cleanliness (Smoke Test)", "pass"))

    session_token = uuid.uuid4().hex
    start_marker = f"__TEST_METRICS_{session_token}_START__"
    end_marker = f"__TEST_METRICS_{session_token}_END__"

    harness = f"""
import sys
import io
import json

_captured_io = io.StringIO()
_orig_stdout = sys.stdout
sys.stdout = _captured_io

_learner_ns = {{}}
_compilation_error = None

try:
    exec({repr(code)}, _learner_ns)
except Exception as _e:
    _compilation_error = f"{{type(_e).__name__}}: {{_e}}"

sys.stdout = _orig_stdout
_captured_stdout = _captured_io.getvalue()
_learner_ns["_captured_stdout"] = _captured_stdout

_results = []
if _compilation_error:
    _results.append({{"name": "Code Compilation / Execution", "status": "FAILED", "error": _compilation_error}})
else:
    _test_cases = {repr(test_cases)}
    for _name, _case_src in _test_cases:
        _test_ns = _learner_ns.copy()
        _test_ns["_captured_stdout"] = _captured_stdout
        try:
            exec(_case_src, _test_ns)
            _results.append({{"name": _name, "status": "PASSED", "error": ""}})
        except AssertionError as _ae:
            _err_msg = str(_ae) or "Assertion failed"
            _results.append({{"name": _name, "status": "FAILED", "error": _err_msg}})
        except Exception as _e:
            _results.append({{"name": _name, "status": "FAILED", "error": f"{{type(_e).__name__}}: {{_e}}"}})

_passed = sum(1 for _r in _results if _r["status"] == "PASSED")
_total = len(_results)
_pct = round((_passed / _total * 100), 1) if _total > 0 else 0.0

print({repr(start_marker)})
print(json.dumps({{
    "total": _total,
    "passed": _passed,
    "failed": _total - _passed,
    "percentage": _pct,
    "results": _results
}}))
print({repr(end_marker)})
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
            "message": "Unable to parse test runner metrics.",
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
