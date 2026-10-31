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
