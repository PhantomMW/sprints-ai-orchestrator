from src.llm import get_llm
from src.state import GradingFeedback
from src.tools.code_repl import python_repl_tool
from src.tools.test_runner import run_automated_tests


def create_assessor_agent():
    """
    Create a Gemini-powered Python assessment agent.

    The agent uses Pydantic structured output to ensure
    consistent grading feedback grounded in deterministic test execution.
    """
    llm = get_llm(temperature=0)
    return llm.with_structured_output(GradingFeedback)


def assess_python_code(
    code: str,
    assignment: str,
    expected_output: str = "",
    test_suite: str = "",
) -> dict:
    """
    Execute, test, and assess a learner's Python submission.

    Unified Execution & Single-Pass Optimization:
        - When a test suite or expected output is provided, the automated test runner executes
          the code inside Docker once with true OS process isolation, capturing execution
          metrics and evaluating all assertions without duplicate container spin-ups.
        - If no assertions are provided, execution is handled directly via python_repl_tool.
        - Fail-Closed Safety: Aborts grading immediately with an error response before calling
          Gemini if the test runner or execution encounters technical failures.
        - Gemini qualitative code evaluation grounded in deterministic test metrics.

    Args:
        code: Python code submitted by the learner.
        assignment: Description of the programming task.
        expected_output: Expected stdout, if provided.
        test_suite: Optional assertions or test cases.

    Returns:
        Structured grading feedback, deterministic test metrics, or error.
    """
    # 1. Deterministic unit test verification (or execution capture)
    test_metrics = run_automated_tests(
        code=code,
        test_suite=test_suite,
        expected_output=expected_output,
    )

    # 2. Extract execution result from test runner, or execute via repl tool if not present
    execution_result = test_metrics.get("execution")
    if not execution_result:
        execution_result = python_repl_tool.invoke({"code": code})

    # Fail-closed on base execution failure
    if execution_result.get("status") != "success" and test_metrics.get("status") != "unverified":
        return {
            "status": "error",
            "message": "Unable to verify: Python execution failed.",
            "execution": execution_result,
            "test_runner": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "percentage": 0.0,
                "results": [],
            },
        }

    # Fail-closed on test runner technical failure
    if test_metrics.get("status") == "error":
        return {
            "status": "error",
            "message": f"Unable to verify: Automated test runner execution failed: {test_metrics.get('message', 'Runner failure')}",
            "execution": execution_result,
            "test_runner": test_metrics,
        }

    # Base execution failure when no tests were provided
    if execution_result.get("status") != "success":
        return {
            "status": "error",
            "message": "Unable to verify: Python execution failed.",
            "execution": execution_result,
            "test_runner": test_metrics,
        }

    assessor = create_assessor_agent()

    prompt = f"""
    You are a professional Python programming assessor for Sprints learners.

    Evaluate the learner's submission using both the deterministic test results
    and the code quality information below.

    ASSIGNMENT:
    {assignment}

    LEARNER CODE:
    {code}

    SANDBOX EXECUTION:
    {execution_result}

    AUTOMATED TEST RUNNER VERIFICATION:
    - Status: {test_metrics.get('status')}
    - Tests Total: {test_metrics.get('total')}
    - Tests Passed: {test_metrics.get('passed')}
    - Tests Failed: {test_metrics.get('failed')}
    - Pass Percentage: {test_metrics.get('percentage')}%
    - Test Case Details: {test_metrics.get('results')}

    EXPECTED OUTPUT:
    {expected_output if expected_output else "Not provided."}

    GRADING RULES:

    1. Ground the evaluation score in the Automated Test Runner verification:
       - If automated tests failed, the score and feedback must reflect the test failures.
       - If all automated tests passed, award points for correctness and evaluate code quality.
    2. Consider execution output, exception handling, and edge cases.
    3. Evaluate code structure, readability, and Python idiomatic practices (PEP 8).
    4. Provide specific, constructive strengths and actionable improvement feedback.
    5. Assign a final score between 0 and 100.
    6. pass_status must be True only when the score is 60 or higher AND the solution
       satisfies the assignment requirements without breaking core tests.

    Return the evaluation using the required structured schema.
    """

    try:
        feedback = assessor.invoke(prompt)
        grading_dict = feedback.model_dump()

        # Programmatic guardrail: failing test assertions prevent pass_status=True
        if test_metrics.get("status") == "success" and test_metrics.get("failed", 0) > 0:
            grading_dict["pass_status"] = False
            grading_dict["score"] = min(grading_dict.get("score", 0), 50)
            failing_names = [
                r["name"] for r in test_metrics.get("results", []) if r.get("status") == "FAILED"
            ]
            if failing_names:
                grading_dict["feedback_points"].append(
                    f"Automated test runner failed on: {', '.join(failing_names)}"
                )
        elif test_metrics.get("status") == "unverified":
            grading_dict["feedback_points"].append(
                "Notice: No automated unit tests were provided for this submission; pass status is based on qualitative code review."
            )

        return {
            "status": "success",
            "execution": execution_result,
            "test_runner": test_metrics,
            "grading": grading_dict,
        }

    except Exception as error:
        return {
            "status": "error",
            "message": f"Unable to verify grading result: {error}",
            "execution": execution_result,
            "test_runner": test_metrics,
        }
