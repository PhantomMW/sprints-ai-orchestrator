from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field


# ==================================================
# 1. PYTHON GRADING SCHEMA
# ==================================================

class GradingFeedback(BaseModel):
    """
    Structured grading feedback for learner Python assignments.

    This schema ensures consistent assessment results
    from the Python Assessor Agent.
    """

    score: int = Field(
        ge=0,
        le=100,
        description="Numerical assessment score between 0 and 100."
    )

    feedback_points: list[str] = Field(
        min_length=1,
        description=(
            "Specific strengths, mistakes, "
            "and improvement points."
        )
    )

    pass_status: bool = Field(
        description="True if the learner passes the assessment."
    )


# ==================================================
# 2. CAREER ADVICE SCHEMA
# ==================================================

class CareerAdvice(BaseModel):
    """
    Structured career recommendations for a Sprints learner.

    Recommendations must be grounded in verified learner
    information and available knowledge sources.
    """

    learner_name: str

    current_track: str

    skill_assessment: str

    recommended_roles: list[str] = Field(
        min_length=1
    )

    skill_gaps: list[str]

    learning_roadmap: list[str] = Field(
        min_length=1
    )

    salary_insight: str

    overall_recommendation: str


# ==================================================
# 3. REQUEST TYPES
# ==================================================

RequestType = Literal[
    "advisor",
    "assessor",
    "general",
    "unknown",
]


# ==================================================
# 4. SHARED LANGGRAPH STATE
# ==================================================

class AgentState(TypedDict, total=False):
    """
    Shared state for the Sprints Multi-Agent AI Orchestrator.

    The state is passed between LangGraph nodes and contains
    learner information, routing decisions, retrieved context,
    assessment results, career recommendations, evaluation
    feedback, human approval status, and final responses.

    The messages field uses the add_messages reducer to
    preserve conversation history across graph nodes.

    total=False allows individual nodes to update only
    the fields they are responsible for.
    """

    # --------------------------------------------------
    # Conversation
    # --------------------------------------------------

    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]

    # --------------------------------------------------
    # Learner Information
    # --------------------------------------------------

    student_id: str

    user_request: str

    # --------------------------------------------------
    # Routing
    # --------------------------------------------------

    request_type: RequestType

    next_agent: str

    # --------------------------------------------------
    # RAG Knowledge Retrieval
    # --------------------------------------------------

    retrieved_context: str

    source_documents: list[dict[str, Any]]

    retrieval_sufficient: bool

    # --------------------------------------------------
    # Advisor Agent Results
    # --------------------------------------------------

    learner_profile: dict[str, Any]

    career_advice: dict[str, Any]

    # --------------------------------------------------
    # Assessor Agent Results
    # --------------------------------------------------

    submitted_code: str

    assignment: str

    expected_output: str

    test_suite: str

    test_runner: dict[str, Any]

    grading_result: dict[str, Any]

    # --------------------------------------------------
    # Evaluator-Optimizer
    # --------------------------------------------------

    iteration_count: int

    max_iterations: int

    evaluator_feedback: str

    evaluation_passed: bool

    # --------------------------------------------------
    # Human-in-the-Loop
    # --------------------------------------------------

    human_approval_required: bool

    human_approval: bool | None

    approval_details: dict[str, Any]

    # --------------------------------------------------
    # Error Handling
    # --------------------------------------------------

    errors: list[str]

    # --------------------------------------------------
    # Final Output
    # --------------------------------------------------

    final_response: str