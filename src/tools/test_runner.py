import json
import re
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

    Runs within the restricted Docker container to guarantee isolation.
    Combines learner code with test assertions, executing each test
    independently to produce verifiable, deterministic pass/fail metrics.

    Args:
        code: Python source code submitted by the learner.
        test_suite: Multiline string containing test assertions or assertions.
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
                f"assert captured_stdout.strip() == {repr(expected_output.strip())}",
            )
        )

    if not test_cases:
        test_cases.append(("Execution Cleanliness (Smoke Test)", "pass"))

    harness = f"""
import sys
import io
import json

captured_io = io.StringIO()
sys_stdout_orig = sys.stdout
sys.stdout = captured_io

learner_ns = {{}}
learner_error = None

try:
    exec({repr(code)}, learner_ns)
except Exception as e:
    learner_error = f"{{type(e).__name__}}: {{e}}"

sys.stdout = sys_stdout_orig
captured_stdout = captured_io.getvalue()
learner_ns["captured_stdout"] = captured_stdout

results = []
if learner_error:
    results.append({{"name": "Code Compilation / Execution", "status": "FAILED", "error": learner_error}})
else:
    cases = {repr(test_cases)}
    for name, case_src in cases:
        try:
            exec(case_src, learner_ns)
            results.append({{"name": name, "status": "PASSED", "error": ""}})
        except AssertionError as ae:
            err_msg = str(ae) or "Assertion failed"
            results.append({{"name": name, "status": "FAILED", "error": err_msg}})
        except Exception as e:
            results.append({{"name": name, "status": "FAILED", "error": f"{{type(e).__name__}}: {{e}}"}})

passed = sum(1 for r in results if r["status"] == "PASSED")
total = len(results)
pct = round((passed / total * 100), 1) if total > 0 else 0.0

print("__TEST_METRICS_START__")
print(json.dumps({{
    "total": total,
    "passed": passed,
    "failed": total - passed,
    "percentage": pct,
    "results": results
}}))
print("__TEST_METRICS_END__")
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
    match = re.search(
        r"__TEST_METRICS_START__\s*(.*?)\s*__TEST_METRICS_END__",
        stdout,
        re.DOTALL,
    )
    if not match:
        return {
            "status": "error",
            "message": "Unable to parse test runner metrics.",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "percentage": 0.0,
            "results": [],
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
            "total": 0,
            "passed": 0,
            "failed": 0,
            "percentage": 0.0,
            "results": [],
            "execution": docker_res,
        }
