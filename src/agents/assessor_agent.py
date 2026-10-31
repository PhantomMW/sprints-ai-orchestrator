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

    Combines:
        1. Automated deterministic test runner verification.
        2. Sandbox execution output capture.
        3. Gemini qualitative code evaluation.

    Args:
        code: Python code submitted by the learner.
        assignment: Description of the programming task.
        expected_output: Expected stdout, if provided.
        test_suite: Optional assertions or test cases.

    Returns:
        Structured grading feedback, deterministic test metrics, or error.
    """

    # 1. Base execution in Docker sandbox
    execution_result = python_repl_tool.invoke({
        "code": code
    })

    if execution_result.get("status") != "success":
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

    # 2. Automated Test Runner (Deterministic unit test verification)
    test_metrics = run_automated_tests(
        code=code,
        test_suite=test_suite,
        expected_output=expected_output,
    )

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
        if test_metrics.get("failed", 0) > 0 or test_metrics.get("status") != "success":
            grading_dict["pass_status"] = False
            grading_dict["score"] = min(grading_dict.get("score", 0), 50)
            failing_names = [
                r["name"] for r in test_metrics.get("results", []) if r.get("status") == "FAILED"
            ]
            if failing_names:
                grading_dict["feedback_points"].append(
                    f"Automated test runner failed on: {', '.join(failing_names)}"
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
