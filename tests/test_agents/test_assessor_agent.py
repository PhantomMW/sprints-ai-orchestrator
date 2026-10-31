from unittest.mock import MagicMock, patch

from src.agents.assessor_agent import assess_python_code
from src.state import GradingFeedback


def _mock_grading_feedback(score: int = 85, pass_status: bool = True):
    return GradingFeedback(
        score=score,
        pass_status=pass_status,
        feedback_points=["Code satisfies requirements.", "Clean structure."],
    )


def test_assess_python_code_with_automated_test_runner():
    """Verify that assess_python_code incorporates automated test runner metrics."""
    with patch("src.agents.assessor_agent.python_repl_tool") as mock_repl_tool, \
         patch("src.agents.assessor_agent.run_automated_tests") as mock_runner, \
         patch("src.agents.assessor_agent.create_assessor_agent") as mock_assessor_factory:

        mock_repl_tool.invoke.return_value = {
            "status": "success",
            "stdout": "6\n",
            "stderr": "",
            "exit_code": 0,
        }

        mock_runner.return_value = {
            "status": "success",
            "total": 2,
            "passed": 2,
            "failed": 0,
            "percentage": 100.0,
            "results": [
                {"name": "test_1", "status": "PASSED", "error": ""},
                {"name": "test_2", "status": "PASSED", "error": ""},
            ],
        }

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _mock_grading_feedback(score=95, pass_status=True)
        mock_assessor_factory.return_value = mock_agent

        result = assess_python_code(
            code="def add(a, b): return a + b",
            assignment="Implement add function",
            test_suite="assert add(1, 2) == 3\nassert add(2, 4) == 6",
        )

        assert result["status"] == "success"
        assert result["test_runner"]["total"] == 2
        assert result["test_runner"]["passed"] == 2
        assert result["test_runner"]["percentage"] == 100.0
        assert result["grading"]["score"] == 95
        assert result["grading"]["pass_status"] is True

        # Verify prompt received the test runner metrics
        call_prompt = mock_agent.invoke.call_args[0][0]
        assert "AUTOMATED TEST RUNNER VERIFICATION" in call_prompt
        assert "- Tests Passed: 2" in call_prompt
        assert "- Tests Total: 2" in call_prompt


def test_assess_python_code_execution_failure():
    """Verify that execution failure aborts early before LLM assessment."""
    with patch("src.agents.assessor_agent.python_repl_tool") as mock_repl_tool:
        mock_repl_tool.invoke.return_value = {
            "status": "error",
            "error": "SyntaxError",
            "stdout": "",
            "stderr": "SyntaxError: invalid syntax",
        }

        result = assess_python_code(
            code="def broken(",
            assignment="Implement function",
        )

        assert result["status"] == "error"
        assert "Python execution failed" in result["message"]
        assert result["test_runner"]["total"] == 0


def test_assess_python_code_runner_failure_aborts_early():
    """Verify that test runner technical failure halts grading immediately before LLM call."""
    with patch("src.agents.assessor_agent.python_repl_tool") as mock_repl_tool, \
         patch("src.agents.assessor_agent.run_automated_tests") as mock_runner, \
         patch("src.agents.assessor_agent.create_assessor_agent") as mock_assessor_factory:

        mock_repl_tool.invoke.return_value = {
            "status": "success",
            "stdout": "done",
            "stderr": "",
            "exit_code": 0,
        }

        # Test runner failed technically (e.g. timeout or docker failure)
        mock_runner.return_value = {
            "status": "error",
            "message": "Docker execution timed out after 5 seconds.",
            "total": 3,
            "passed": 0,
            "failed": 3,
            "percentage": 0.0,
            "results": [],
        }

        result = assess_python_code(
            code="import time; time.sleep(10)",
            assignment="Quick task",
            test_suite="assert True",
        )

        assert result["status"] == "error"
        assert "Automated test runner execution failed" in result["message"]
        mock_assessor_factory.assert_not_called()


def test_assess_python_code_assertion_failure_caps_score():
    """Verify that failing test assertions enforce pass_status=False and score capping."""
    with patch("src.agents.assessor_agent.python_repl_tool") as mock_repl_tool, \
         patch("src.agents.assessor_agent.run_automated_tests") as mock_runner, \
         patch("src.agents.assessor_agent.create_assessor_agent") as mock_assessor_factory:

        mock_repl_tool.invoke.return_value = {
            "status": "success",
            "stdout": "4",
            "stderr": "",
            "exit_code": 0,
        }

        # 1 test failed
        mock_runner.return_value = {
            "status": "success",
            "total": 2,
            "passed": 1,
            "failed": 1,
            "percentage": 50.0,
            "results": [
                {"name": "test_1", "status": "PASSED", "error": ""},
                {"name": "test_2", "status": "FAILED", "error": "AssertionError"},
            ],
        }

        mock_agent = MagicMock()
        # LLM mistakenly tries to give high score and pass
        mock_agent.invoke.return_value = _mock_grading_feedback(score=90, pass_status=True)
        mock_assessor_factory.return_value = mock_agent

        result = assess_python_code(
            code="def add(a, b): return a + b",
            assignment="Implement add",
            test_suite="assert add(1, 2) == 3\nassert add(2, 2) == 5",
        )

        assert result["status"] == "success"
        # Guardrail should have overridden pass_status and capped score
        assert result["grading"]["pass_status"] is False
        assert result["grading"]["score"] <= 50
        assert any("test_2" in f for f in result["grading"]["feedback_points"])
