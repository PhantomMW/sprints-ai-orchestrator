from typing import Any

from pydantic import BaseModel, Field

from src.llm import get_llm


class EvaluationResult(BaseModel):
    """Structured quality evaluation of an agent response."""

    score: int = Field(
        ge=0,
        le=100,
        description="Quality score between 0 and 100.",
    )

    passed: bool = Field(
        description="Whether the response meets all quality requirements.",
    )

    feedback: list[str] = Field(
        min_length=1,
        description="Specific issues that should be improved.",
    )


class OptimizedResponse(BaseModel):
    """Structured improved response."""

    response: str = Field(
        min_length=1,
        description="The improved response.",
    )


DEFAULT_MAX_ITERATIONS = 3
PASSING_SCORE = 85


def evaluate_response(
    response: str,
    task_context: str,
) -> EvaluationResult:
    """
    Evaluate an agent response against its task context.

    The evaluator checks:
        - Factual consistency.
        - Relevance to the original task.
        - Clarity and completeness.
        - Unsupported claims or invented information.

    Args:
        response: The response produced by an agent.
        task_context: Verified information relevant to the task.

    Returns:
        Structured evaluation result.
    """

    llm = get_llm(temperature=0)

    evaluator = llm.with_structured_output(EvaluationResult)

    prompt = f"""
You are a strict quality evaluator for a multi-agent AI system.

Evaluate the response using ONLY the supplied task context.

TASK CONTEXT:
{task_context}

RESPONSE:
{response}

EVALUATION CRITERIA:

1. Factual accuracy:
   The response must not contradict the task context.

2. Grounding:
   The response must not invent facts, qualifications,
   database records, job vacancies, or execution results.

3. Relevance:
   The response must address the original task.

4. Clarity:
   The response must be understandable and well organized.

5. Completeness:
   The response must cover the important requirements.

SCORING:

- 90-100: Excellent.
- 85-89: Acceptable.
- 70-84: Needs improvement.
- 0-69: Unsatisfactory.

Set passed=True only when:
- The score is at least 85.
- There are no critical factual or grounding problems.

If the context does not contain enough information
to verify a claim, identify that limitation.

Return specific and actionable feedback.
"""

    return evaluator.invoke(prompt)


def optimize_response(
    response: str,
    task_context: str,
    feedback: list[str],
) -> str:
    """
    Improve a response using evaluator feedback.

    The optimizer must preserve all verified facts
    and must not invent missing information.

    Args:
        response: Original response.
        task_context: Verified task information.
        feedback: Issues identified by the evaluator.

    Returns:
        Improved response as plain text.
    """

    llm = get_llm(temperature=0)

    optimizer = llm.with_structured_output(OptimizedResponse)

    prompt = f"""
You are a response optimization agent.

Improve the response using the evaluator's feedback.

VERIFIED TASK CONTEXT:
{task_context}

ORIGINAL RESPONSE:
{response}

EVALUATOR FEEDBACK:
{feedback}

STRICT RULES:

1. Preserve every verified factual detail.
2. Never invent information.
3. Never add unsupported qualifications or job vacancies.
4. Never change numerical scores or execution results.
5. Do not remove important information.
6. Improve clarity, structure, and completeness.
7. If a claim cannot be verified, preserve its uncertainty.
8. Do not claim that missing information is verified.

Return only the improved response in the required schema.
"""

    result = optimizer.invoke(prompt)

    return result.response


def evaluate_and_optimize(
    response: str,
    task_context: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> dict[str, Any]:
    """
    Run the Evaluator-Optimizer loop with a strict iteration limit.

    Workflow:
        1. Evaluate the initial response.
        2. Stop if it passes.
        3. Otherwise, optimize the response.
        4. Evaluate the improved response.
        5. Repeat until it passes or reaches the iteration limit.

    Args:
        response: Initial agent response.
        task_context: Verified information for evaluation.
        max_iterations: Maximum optimization iterations.

    Returns:
        Final response, evaluation details, and iteration count.
    """

    if not isinstance(response, str) or not response.strip():
        return {
            "status": "error",
            "message": "Unable to verify: Response is empty.",
            "response": response,
            "iteration_count": 0,
            "evaluation": None,
        }

    if not isinstance(task_context, str) or not task_context.strip():
        return {
            "status": "error",
            "message": "Unable to verify: Task context is missing.",
            "response": response,
            "iteration_count": 0,
            "evaluation": None,
        }

    if not isinstance(max_iterations, int) or max_iterations < 0:
        return {
            "status": "error",
            "message": "max_iterations must be a non-negative integer.",
            "response": response,
            "iteration_count": 0,
            "evaluation": None,
        }

    max_iterations = min(max_iterations, DEFAULT_MAX_ITERATIONS)

    current_response = response
    iteration_count = 0

    try:
        evaluation = evaluate_response(
            current_response,
            task_context,
        )

        while not evaluation.passed and iteration_count < max_iterations:

            current_response = optimize_response(
                response=current_response,
                task_context=task_context,
                feedback=evaluation.feedback,
            )

            iteration_count += 1

            evaluation = evaluate_response(
                current_response,
                task_context,
            )

        return {
            "status": "success",
            "response": current_response,
            "iteration_count": iteration_count,
            "max_iterations": max_iterations,
            "evaluation": evaluation.model_dump(),
            "passed": evaluation.passed,
        }

    except Exception as error:
        return {
            "status": "error",
            "message": f"Unable to verify optimization result: {error}",
            "response": current_response,
            "iteration_count": iteration_count,
            "evaluation": None,
        }