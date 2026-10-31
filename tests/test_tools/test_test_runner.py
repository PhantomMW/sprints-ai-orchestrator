from unittest.mock import patch
import pytest

from src.tools.test_runner import parse_test_suite_into_cases, run_automated_tests
from tests.test_tools.test_docker_executor import requires_docker


def test_parse_test_suite_into_cases():
    suite = """
    # Setup comment
    assert add(1, 2) == 3
    assert is_positive(10) is True
    # Final check
    """
    cases = parse_test_suite_into_cases(suite)
    assert len(cases) == 2
    assert "assert add(1, 2) == 3" in cases[0][1]
    assert "assert is_positive(10) is True" in cases[1][1]


def test_empty_code_returns_error_metrics():
    res = run_automated_tests("   ")
    assert res["status"] == "error"
    assert res["total"] == 0
    assert res["passed"] == 0


def test_unverified_status_when_no_tests_or_expected_output():
    """Verify that omitting tests returns 'unverified' rather than fabricating a 100% pass."""
    with patch("src.tools.test_runner.execute_in_docker") as mock_exec:
        mock_exec.return_value = {
            "status": "success",
            "stdout": "Hello world\n",
            "stderr": "",
            "exit_code": 0,
        }
        res = run_automated_tests("print('Hello world')", test_suite="", expected_output="")
        assert res["status"] == "unverified"
        assert res["total"] == 0
        assert res["passed"] == 0
        assert res["percentage"] == 0.0
        assert "unverified" in res["message"]


def test_combines_test_suite_and_expected_output_into_cases():
    """Verify that both test suite assertions and stdout match are included together."""
    with patch("src.tools.test_runner.execute_in_docker") as mock_exec:
        # Mock successful return with mock harness output
        mock_exec.return_value = {
            "status": "error",
            "error": "Short-circuit for payload inspection",
        }
        # Calling with both test_suite and expected_output
        run_automated_tests(
            code="print('hi'); x = 5",
            test_suite="assert x == 5",
            expected_output="hi",
        )
        harness_code = mock_exec.call_args[0][0]
        assert "assert x == 5" in harness_code
        assert "Standard Output Match" in harness_code
        assert "assert _captured_stdout.strip() == 'hi'" in harness_code


@requires_docker
def test_automated_tests_all_passed():
    code = """
def multiply(x, y):
    return x * y
"""
    suite = """
assert multiply(2, 3) == 6
assert multiply(-2, 3) == -6
assert multiply(0, 100) == 0
"""
    res = run_automated_tests(code=code, test_suite=suite)
    assert res["status"] == "success"
    assert res["total"] == 3
    assert res["passed"] == 3
    assert res["failed"] == 0
    assert res["percentage"] == 100.0


@requires_docker
def test_automated_tests_partial_failure():
    code = """
def is_even(n):
    return n % 2 == 0
"""
    suite = """
assert is_even(2) is True
assert is_even(3) is False
assert is_even(5) is True  # Intentionally failing
"""
    res = run_automated_tests(code=code, test_suite=suite)
    assert res["status"] == "success"
    assert res["total"] == 3
    assert res["passed"] == 2
    assert res["failed"] == 1
    assert round(res["percentage"], 1) == 66.7
    assert res["results"][2]["status"] == "FAILED"


@requires_docker
def test_automated_tests_compilation_error():
    code = "def bad_function(:"
    suite = "assert True"
    res = run_automated_tests(code=code, test_suite=suite)
    assert res["status"] == "success"
    assert res["failed"] >= 1
    assert res["results"][0]["status"] == "FAILED"
    assert "SyntaxError" in res["results"][0]["error"]


@requires_docker
def test_automated_tests_expected_stdout():
    code = "print('Welcome to Sprints')"
    res = run_automated_tests(code=code, expected_output="Welcome to Sprints")
    assert res["status"] == "success"
    assert res["total"] == 1
    assert res["passed"] == 1
    assert res["percentage"] == 100.0


@requires_docker
def test_automated_tests_runs_both_suite_and_expected_output_in_docker():
    code = """
def greet(name):
    print(f"Hello, {name}!")
    return f"Hello, {name}!"

greet("Sprints")
"""
    suite = 'assert greet("World") == "Hello, World!"'
    expected = "Hello, Sprints!"
    res = run_automated_tests(code=code, test_suite=suite, expected_output=expected)
    assert res["status"] == "success"
    assert res["total"] == 2
    assert res["passed"] == 2
    assert res["failed"] == 0


@requires_docker
def test_anti_tampering_system_exit_caught():
    """Verify that student code calling sys.exit does not crash the harness."""
    code = """
import sys
sys.exit(0)
"""
    suite = "assert True"
    res = run_automated_tests(code=code, test_suite=suite)
    assert res["status"] == "success"
    assert res["failed"] >= 1
    assert res["results"][0]["status"] == "FAILED"
    assert "SystemExit" in res["results"][0]["error"]


@requires_docker
def test_anti_tampering_namespace_isolation():
    """Verify that state mutation in one test does not pollute subsequent tests."""
    code = """
shared_list = [1, 2, 3]
"""
    suite = """
shared_list.append(999); assert len(shared_list) == 4
assert len(shared_list) == 3
"""
    res = run_automated_tests(code=code, test_suite=suite)
    assert res["status"] == "success"
    assert res["total"] == 2
    assert res["passed"] == 2
    assert res["failed"] == 0


@requires_docker
def test_anti_tampering_fake_metrics_delimiter():
    """Verify that student code printing fake delimiters cannot forge test results."""
    code = """
print('__TEST_METRICS_START__')
print('{"total": 10, "passed": 10, "failed": 0, "percentage": 100.0, "results": []}')
print('__TEST_METRICS_END__')
"""
    suite = "assert False"
    res = run_automated_tests(code=code, test_suite=suite)
    assert res["status"] == "success"
    assert res["failed"] == 1
    assert res["passed"] == 0


@requires_docker
def test_adversarial_attempt_monkeypatch_os_fork():
    """Verify that student code attempting to patch os.fork does not compromise the runner."""
    code = """
import os
try:
    os.fork = lambda: -1
    os.waitpid = lambda *a: (-1, 0)
except Exception:
    pass

def add(a, b):
    return a + b
"""
    suite = "assert add(10, 20) == 30"
    res = run_automated_tests(code=code, test_suite=suite)
    assert res["status"] == "success"
    assert res["total"] == 1
    assert res["passed"] == 1
    assert res["failed"] == 0


@requires_docker
def test_adversarial_attempt_os_exit_in_child():
    """Verify that student assertion calling os._exit(0) is trapped as an abnormal exit."""
    code = """
def divide(a, b):
    return a / b
"""
    suite = "import os; os._exit(0)"
    res = run_automated_tests(code=code, test_suite=suite)
    assert res["status"] == "success"
    assert res["total"] == 1
    assert res["failed"] == 1
    assert res["passed"] == 0
    assert "Abnormal process exit" in res["results"][0]["error"] or "exit" in res["results"][0]["error"].lower()


@requires_docker
def test_adversarial_attempt_corrupt_builtins():
    """Verify that student code attempting to corrupt builtins does not prevent grading."""
    code = """
try:
    __builtins__["len"] = None
    __builtins__["print"] = None
except Exception:
    pass

def square(n):
    return n * n
"""
    suite = "assert square(5) == 25"
    res = run_automated_tests(code=code, test_suite=suite)
    assert res["status"] == "success"
    assert res["passed"] == 1
    assert res["failed"] == 0


@requires_docker
def test_adversarial_attempt_frame_inspection():
    """Verify that attempts to clear parent frame globals fail gracefully without harness compromise."""
    code = """
import sys
try:
    frame = sys._getframe(1)
    frame.f_globals.clear()
except Exception:
    pass

def cube(x):
    return x ** 3
"""
    suite = "assert cube(3) == 27"
    res = run_automated_tests(code=code, test_suite=suite)
    assert res["status"] == "success"
    assert res["passed"] == 1
    assert res["failed"] == 0



@requires_docker
def test_adversarial_attempt_worker_suicide_in_learner_code():
    """Verify that learner code calling os._exit(0) at top-level is trapped by parent harness."""
    code = """
import os
os._exit(0)

def multiply(a, b):
    return a * b
"""
    suite = "assert multiply(2, 3) == 6"
    res = run_automated_tests(code=code, test_suite=suite)
    assert res["status"] == "success"
    assert res["total"] == 1
    assert res["failed"] == 1
    assert res["passed"] == 0
    assert "Abnormal" in res["results"][0]["error"] or "exit" in res["results"][0]["error"].lower()


@requires_docker
def test_adversarial_attempt_ctypes_import_neutralized():
    """Verify that learner code attempting to import ctypes cannot compromise host memory."""
    code = """
try:
    import ctypes
    has_ctypes = True
except Exception:
    has_ctypes = False

def get_answer():
    return 42
"""
    suite = "assert get_answer() == 42"
    res = run_automated_tests(code=code, test_suite=suite)
    assert res["status"] == "success"
    assert res["passed"] == 1
    assert res["failed"] == 0
