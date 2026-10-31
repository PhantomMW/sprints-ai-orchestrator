from src.llm import get_llm
from src.state import GradingFeedback
from src.tools.code_repl import python_repl_tool


def create_assessor_agent():
    """
    Create a Gemini-powered Python assessment agent.

    The agent uses Pydantic structured output to ensure
    consistent grading feedback.
    """

    llm = get_llm(temperature=0)

    return llm.with_structured_output(GradingFeedback)


def assess_python_code(
    code: str,
    assignment: str,
    expected_output: str = ""
) -> dict:
    """
    Execute and assess a learner's Python submission.

    The Python REPL executes the submitted code.
    Gemini evaluates the assignment requirements,
    execution results, and code quality.

    Args:
        code: Python code submitted by the learner.
        assignment: Description of the programming task.
        expected_output: Expected output, if provided.

    Returns:
        Structured grading feedback or an error response.
    """

    execution_result = python_repl_tool.invoke({
        "code": code
    })

    if execution_result["status"] != "success":
        return {
            "status": "error",
            "message": "Unable to verify: Python execution failed.",
            "execution": execution_result
        }

    assessor = create_assessor_agent()

    prompt = f"""
    You are a professional Python programming assessor.

    Evaluate the learner's submission using the information below.

    ASSIGNMENT:
    {assignment}

    LEARNER CODE:
    {code}

    EXECUTION RESULT:
    {execution_result}

    EXPECTED OUTPUT:
    {expected_output if expected_output else "Not provided."}

    GRADING RULES:

    1. Evaluate correctness against the assignment.
    2. Consider the actual execution output.
    3. Check whether the solution satisfies the requirements.
    4. Identify strengths and weaknesses.
    5. Consider readability and code structure.
    6. Do not claim that tests passed unless their results
       are available in the execution information.
    7. Assign a score between 0 and 100.
    8. pass_status must be True only when the score is 60
       or higher and the solution satisfies the requirements.
    9. Provide specific and actionable feedback.

    Return the evaluation using the required structured schema.
    """

    try:
        feedback = assessor.invoke(prompt)

        return {
            "status": "success",
            "execution": execution_result,
            "grading": feedback.model_dump()
        }

    except Exception as error:
        return {
            "status": "error",
            "message": f"Unable to verify grading result: {error}",
            "execution": execution_result
        }