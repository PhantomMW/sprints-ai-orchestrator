import json
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.advisor_agent import generate_career_advice
from src.agents.assessor_agent import assess_python_code
from src.agents.evaluator_optimizer import evaluate_and_optimize
from src.state import AgentState


MAX_OPTIMIZATION_ITERATIONS = 3
TIER_1_EMPLOYERS = {"fawry"}


def router_node(state: AgentState) -> dict:
    """
    Determine which agent should handle the user request.

    Routing priority:
        1. Explicit request_type.
        2. Submitted Python code and assignment.
        3. Career advice request.
    """

    request_type = state.get("request_type", "unknown")

    if request_type in ("advisor", "assessor"):
        selected_agent = request_type

    elif state.get("submitted_code") and state.get("assignment"):
        selected_agent = "assessor"

    elif state.get("student_id"):
        selected_agent = "advisor"

    else:
        selected_agent = "unknown"

    return {
        "next_agent": selected_agent,
        "request_type": selected_agent,
    }


def advisor_node(state: AgentState) -> dict:
    """
    Execute the Career Advisor agent.
    """

    student_id = state.get("student_id")

    if not student_id:
        response = "Unable to verify: Student ID is missing."

        return {
            "career_advice": {"status": "error"},
            "final_response": response,
            "errors": ["Missing student_id."],
            "messages": [AIMessage(content=response)],
        }

    result = generate_career_advice(student_id)

    if result.get("status") != "success":
        response = result.get(
            "message",
            "Unable to verify career advice.",
        )

        return {
            "career_advice": result,
            "final_response": response,
            "errors": [response],
            "messages": [AIMessage(content=response)],
        }

    advice = result["advice"]

    response = (
        f"Career advice generated for "
        f"{advice['learner_name']}.\n\n"
        f"Current Track: {advice['current_track']}\n"
        f"Recommended Roles: "
        f"{', '.join(advice['recommended_roles'])}\n\n"
        f"Overall Recommendation:\n"
        f"{advice['overall_recommendation']}"
    )

    return {
        "career_advice": result,
        "final_response": response,
        "messages": [AIMessage(content=response)],
    }


def assessor_node(state: AgentState) -> dict:
    """
    Execute the Python Assessor agent.

    Python execution takes place inside the Docker sandbox.
    """

    code = state.get("submitted_code", "")
    assignment = state.get("assignment", "")
    expected_output = state.get("expected_output", "")

    if not code.strip() or not assignment.strip():
        response = (
            "Unable to verify: Python code or assignment is missing."
        )

        return {
            "grading_result": {"status": "error"},
            "final_response": response,
            "errors": [response],
            "messages": [AIMessage(content=response)],
        }

    result = assess_python_code(
        code=code,
        assignment=assignment,
        expected_output=expected_output,
    )

    if result.get("status") != "success":
        response = result.get(
            "message",
            "Unable to verify Python assessment.",
        )

        return {
            "grading_result": result,
            "final_response": response,
            "errors": [response],
            "messages": [AIMessage(content=response)],
        }

    grading = result["grading"]

    response = (
        f"Python Assessment Completed.\n\n"
        f"Score: {grading['score']}/100\n"
        f"Pass Status: {grading['pass_status']}\n\n"
        f"Feedback:\n"
        + "\n".join(
            f"- {point}"
            for point in grading["feedback_points"]
        )
    )

    return {
        "grading_result": result,
        "final_response": response,
        "messages": [AIMessage(content=response)],
    }


def error_node(state: AgentState) -> dict:
    """
    Handle requests that cannot be routed.
    """

    response = (
        "Unable to determine the appropriate agent. "
        "Please provide a learner ID for career advice "
        "or Python code with an assignment for assessment."
    )

    return {
        "final_response": response,
        "errors": [response],
        "messages": [AIMessage(content=response)],
    }


def route_to_agent(
    state: AgentState,
) -> Literal["advisor", "assessor", "error"]:
    """
    Select the destination agent based on router output.
    """

    next_agent = state.get("next_agent", "unknown")

    if next_agent == "advisor":
        return "advisor"

    if next_agent == "assessor":
        return "assessor"

    return "error"


def route_after_agent(
    state: AgentState,
) -> Literal["evaluate", "end"]:
    """
    Send successful agent results to the evaluator.

    Failed or unverifiable results bypass optimization.
    """

    career_result = state.get("career_advice", {})
    grading_result = state.get("grading_result", {})

    if career_result.get("status") == "success":
        return "evaluate"

    if grading_result.get("status") == "success":
        return "evaluate"

    return "end"


def evaluator_node(state: AgentState) -> dict:
    """
    Evaluate and optimize successful agent responses.

    The evaluator checks factual consistency, grounding,
    relevance, clarity, and completeness.

    The optimizer can improve the response but must preserve
    verified facts, execution results, and numerical scores.

    If evaluation fails because of an API or other error,
    the original agent response is preserved.
    """

    original_response = state.get("final_response", "")

    career_result = state.get("career_advice", {})
    grading_result = state.get("grading_result", {})

    if career_result.get("status") == "success":
        task_context = (
            "CAREER ADVISOR VERIFIED INFORMATION:\n"
            + json.dumps(
                career_result,
                ensure_ascii=False,
                default=str,
            )
            + "\n\n"
            "IMPORTANT:\n"
            "Do not invent qualifications, active vacancies, "
            "employer confirmations, or salary statistics."
        )

    elif grading_result.get("status") == "success":
        task_context = (
            "PYTHON ASSESSMENT VERIFIED INFORMATION:\n"
            + json.dumps(
                {
                    "assignment": state.get("assignment", ""),
                    "expected_output": state.get("expected_output", ""),
                    "submitted_code": state.get("submitted_code", ""),
                    "execution": grading_result.get("execution"),
                    "grading": grading_result.get("grading"),
                },
                ensure_ascii=False,
                default=str,
            )
            + "\n\n"
            "IMPORTANT:\n"
            "Never change the actual execution result, "
            "numerical score, or pass status."
        )

    else:
        return {
            "iteration_count": 0,
            "evaluation_passed": False,
            "evaluator_feedback": "No successful result to evaluate.",
        }

    result = evaluate_and_optimize(
        response=original_response,
        task_context=task_context,
        max_iterations=MAX_OPTIMIZATION_ITERATIONS,
    )

    if result.get("status") != "success":
        error_message = result.get(
            "message",
            "Unable to verify evaluator result.",
        )

        return {
            "final_response": original_response,
            "iteration_count": result.get("iteration_count", 0),
            "evaluation_passed": False,
            "evaluator_feedback": error_message,
            "errors": [error_message],
        }

    optimized_response = result["response"]
    evaluation = result.get("evaluation", {})

    feedback = evaluation.get("feedback", [])

    return {
        "final_response": optimized_response,
        "iteration_count": result["iteration_count"],
        "max_iterations": MAX_OPTIMIZATION_ITERATIONS,
        "evaluation_passed": result["passed"],
        "evaluator_feedback": "\n".join(feedback),
        "messages": [AIMessage(content=optimized_response)],
    }



def requires_human_placement_review(state: AgentState) -> bool:
    """
    Determine whether a successful career recommendation requires
    human review before it is finalized.

    The prototype currently treats Fawry as a tier-1 employer.
    Review is required only when the verified learner profile lists
    that employer and the learner is marked Ready for Placement.
    """
    career_result = state.get("career_advice", {})

    if career_result.get("status") != "success":
        return False

    learner = career_result.get("learner_profile", {})
    preferred = learner.get("preferred_employers", "")
    readiness = str(learner.get("readiness_status", "")).strip().casefold()

    if isinstance(preferred, str):
        employers = {
            employer.strip().casefold()
            for employer in preferred.split(",")
            if employer.strip()
        }
    elif isinstance(preferred, (list, tuple, set)):
        employers = {
            str(employer).strip().casefold()
            for employer in preferred
            if str(employer).strip()
        }
    else:
        employers = set()

    return bool(
        employers.intersection(TIER_1_EMPLOYERS)
        and readiness == "ready for placement"
    )


def route_after_evaluation(
    state: AgentState,
) -> Literal["placement_review", "end"]:
    """Pause for human review only for qualifying tier-1 recommendations."""
    if requires_human_placement_review(state):
        return "placement_review"
    return "end"


def placement_review_node(state: AgentState) -> dict:
    """
    Pause the graph until a human approves or rejects a tier-1
    employer recommendation.

    This approval applies only to the recommendation. It does not
    confirm a vacancy, submit an application, or contact an employer.
    """
    career_result = state.get("career_advice", {})
    learner = career_result.get("learner_profile", {})
    advice = career_result.get("advice", {})

    review_request = {
        "type": "tier_1_placement_recommendation_review",
        "message": (
            "Human review required before finalizing this placement "
            "recommendation. This is not confirmation of an active vacancy."
        ),
        "student_id": career_result.get("student_id", state.get("student_id")),
        "learner_name": learner.get("name", advice.get("learner_name", "Unknown")),
        "preferred_employers": learner.get("preferred_employers", ""),
        "readiness_status": learner.get("readiness_status", ""),
        "recommended_roles": advice.get("recommended_roles", []),
        "active_vacancy_verified": career_result.get("evidence", {}).get(
            "active_vacancy_verified", False
        ),
        "decision_required": {
            "approved": "true or false",
            "reviewer": "optional reviewer identifier",
            "comments": "optional review comments",
        },
    }

    # LangGraph suspends this node here and resumes it with Command(resume=...).
    decision = interrupt(review_request)

    approved = isinstance(decision, dict) and decision.get("approved") is True
    reviewer = decision.get("reviewer", "") if isinstance(decision, dict) else ""
    comments = decision.get("comments", "") if isinstance(decision, dict) else ""

    decision_label = "APPROVED" if approved else "REJECTED"
    response = state.get("final_response", "")
    response += (
        f"\n\nHuman placement recommendation review: {decision_label}."
    )

    if reviewer:
        response += f"\nReviewer: {reviewer}."
    if comments:
        response += f"\nReview comments: {comments}"
    response += (
        "\nThis decision concerns the recommendation only; "
        "no active vacancy or employer acceptance has been verified."
    )

    return {
        "human_approval_required": True,
        "human_approval": approved,
        "approval_details": {
            "approved": approved,
            "reviewer": reviewer,
            "comments": comments,
            "student_id": review_request["student_id"],
            "employer_review": "Fawry",
            "active_vacancy_verified": False,
        },
        "final_response": response,
        "messages": [AIMessage(content=response)],
    }


def build_orchestrator():
    """
    Build and compile the Sprints LangGraph orchestrator.

    Features:
        - Centralized request routing.
        - Career Advisor integration.
        - Python Assessor integration.
        - Docker-based Python execution through the assessor.
        - Evaluator-Optimizer integration.
        - Maximum three optimization iterations.
        - MemorySaver checkpointing.
        - Human-in-the-Loop review for eligible tier-1 employer recommendations.
    """

    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("advisor", advisor_node)
    workflow.add_node("assessor", assessor_node)
    workflow.add_node("error", error_node)
    workflow.add_node("evaluate", evaluator_node)
    workflow.add_node("placement_review", placement_review_node)

    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "advisor": "advisor",
            "assessor": "assessor",
            "error": "error",
        },
    )

    workflow.add_conditional_edges(
        "advisor",
        route_after_agent,
        {
            "evaluate": "evaluate",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "assessor",
        route_after_agent,
        {
            "evaluate": "evaluate",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "placement_review": "placement_review",
            "end": END,
        },
    )

    workflow.add_edge("placement_review", END)
    workflow.add_edge("error", END)

    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)


# Compiled graph instance
orchestrator = build_orchestrator()