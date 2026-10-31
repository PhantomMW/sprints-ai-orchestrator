from typing import Any

from pydantic import BaseModel, Field

from src.llm import get_llm
from src.state import CareerAdvice
from src.tools.db_queries import get_learner_by_id
from src.tools.rag_retriever import retrieve_context


MAX_RETRIEVAL_RETRIES = 2
RETRIEVAL_TOP_K = 4


class RetrievalAssessment(BaseModel):
    """Structured evaluation of retrieved knowledge."""

    sufficient: bool = Field(
        description=(
            "Whether the retrieved documents contain enough "
            "relevant evidence for the requested career advice."
        )
    )

    relevance_score: int = Field(
        ge=0,
        le=100,
        description="Relevance score between 0 and 100.",
    )

    reasoning: str = Field(
        min_length=1,
        description="Explanation of the retrieval quality.",
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description="Important missing information.",
    )


class QueryReformulation(BaseModel):
    """Structured reformulation of an insufficient RAG query."""

    improved_query: str = Field(
        min_length=5,
        description="Improved search query for the next retrieval.",
    )

    reasoning: str = Field(
        min_length=1,
        description="Why the new query may retrieve better evidence.",
    )


def create_advisor_agent():
    """
    Create a Gemini-powered Career Advisor
    with structured Pydantic output.
    """

    llm = get_llm(temperature=0.2)

    return llm.with_structured_output(CareerAdvice)


def create_retrieval_evaluator():
    """Create a structured retrieval quality evaluator."""

    llm = get_llm(temperature=0)

    return llm.with_structured_output(RetrievalAssessment)


def create_query_reformulator():
    """Create a structured query reformulation agent."""

    llm = get_llm(temperature=0)

    return llm.with_structured_output(QueryReformulation)


def get_learner_profile(student_id: str) -> dict[str, Any]:
    """Retrieve a learner profile from SQLite."""

    return get_learner_by_id(student_id)


def retrieve_career_knowledge(
    query: str,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant career knowledge from ChromaDB.

    Args:
        query: Search query.

    Returns:
        Retrieved documents with source and trust metadata.
    """

    documents = retrieve_context(
        query,
        k=RETRIEVAL_TOP_K,
    )

    return [
        {
            "content": document.page_content,
            "source": document.metadata.get(
                "source",
                "Unknown source",
            ),
            "trust_level": document.metadata.get(
                "trust_level",
                "unspecified",
            ),
        }
        for document in documents
    ]


def assess_retrieval_quality(
    query: str,
    documents: list[dict[str, Any]],
    learner: dict[str, Any],
) -> RetrievalAssessment:
    """
    Evaluate whether retrieved knowledge is sufficient.

    The evaluator must distinguish between:
        - Relevant evidence.
        - Missing information.
        - Unsupported assumptions.
    """

    evaluator = create_retrieval_evaluator()

    retrieved_context = "\n\n".join(
        [
            (
                f"Source: {item['source']}\n"
                f"Trust: {item['trust_level']}\n"
                f"Content: {item['content']}"
            )
            for item in documents
        ]
    )

    prompt = f"""
You are an Agentic RAG retrieval evaluator.

Determine whether the retrieved documents contain
sufficient evidence to support career recommendations.

LEARNER TRACK:
{learner.get('current_track', 'Unknown')}

LEARNER PROFILE:
{learner}

SEARCH QUERY:
{query}

RETRIEVED DOCUMENTS:
{retrieved_context if retrieved_context else "No documents retrieved."}

EVALUATION REQUIREMENTS:

1. Evaluate relevance to the learner's current track.
2. Evaluate whether the documents contain useful
   technical or career-related information.
3. Identify missing evidence.
4. Do not assume that project requirements are
   official employer hiring requirements.
5. Do not treat missing information as verified.
6. Be strict about evidence quality.

Set sufficient=True only if the retrieved evidence
is genuinely useful for the requested career advice.

Return a structured evaluation.
"""

    return evaluator.invoke(prompt)


def reformulate_retrieval_query(
    original_query: str,
    previous_documents: list[dict[str, Any]],
    assessment: RetrievalAssessment,
    learner: dict[str, Any],
) -> QueryReformulation:
    """
    Reformulate an insufficient retrieval query.

    Uses the evaluator's feedback to improve the next search.
    """

    reformulator = create_query_reformulator()

    previous_sources = [
        item["source"]
        for item in previous_documents
    ]

    prompt = f"""
You are a search query reformulation agent.

The previous RAG retrieval did not provide sufficient evidence.

LEARNER TRACK:
{learner.get('current_track', 'Unknown')}

LEARNER PROFILE:
{learner}

ORIGINAL QUERY:
{original_query}

PREVIOUS SOURCES:
{previous_sources}

RETRIEVAL EVALUATION:
{assessment.model_dump()}

YOUR TASK:

Create a more specific search query that addresses
the missing information identified by the evaluator.

RULES:

1. Focus on the learner's actual track.
2. Include relevant technical skills and career concepts.
3. Address the missing information.
4. Avoid unrelated search terms.
5. Do not invent facts about the learner.
6. Produce one focused search query.

Return the improved query and reasoning.
"""

    return reformulator.invoke(prompt)


def agentic_retrieve_career_knowledge(
    initial_query: str,
    learner: dict[str, Any],
) -> dict[str, Any]:
    """
    Perform Agentic RAG with retrieval self-reflection.

    Workflow:
        1. Retrieve initial documents.
        2. Evaluate retrieval quality.
        3. Reformulate the query if evidence is insufficient.
        4. Retry retrieval up to two times.
        5. Return the final retrieval result and trace.

    The maximum number of retrieval attempts is three:
    one initial attempt and two retries.
    """

    current_query = initial_query

    retrieval_trace = []

    final_documents = []
    final_assessment = None

    for attempt in range(MAX_RETRIEVAL_RETRIES + 1):

        documents = retrieve_career_knowledge(
            current_query
        )

        assessment = assess_retrieval_quality(
            query=current_query,
            documents=documents,
            learner=learner,
        )

        retrieval_trace.append(
            {
                "attempt": attempt + 1,
                "query": current_query,
                "documents_count": len(documents),
                "sources": [
                    item["source"]
                    for item in documents
                ],
                "sufficient": assessment.sufficient,
                "relevance_score": assessment.relevance_score,
                "reasoning": assessment.reasoning,
                "missing_information": assessment.missing_information,
            }
        )

        final_documents = documents
        final_assessment = assessment

        if assessment.sufficient:
            return {
                "status": "success",
                "documents": final_documents,
                "final_query": current_query,
                "retrieval_attempts": attempt + 1,
                "retry_count": attempt,
                "retrieval_sufficient": True,
                "assessment": assessment.model_dump(),
                "retrieval_trace": retrieval_trace,
            }

        if attempt >= MAX_RETRIEVAL_RETRIES:
            break

        reformulation = reformulate_retrieval_query(
            original_query=current_query,
            previous_documents=documents,
            assessment=assessment,
            learner=learner,
        )

        current_query = reformulation.improved_query

        retrieval_trace[-1]["reformulation_reasoning"] = (
            reformulation.reasoning
        )

    return {
        "status": "insufficient_evidence",
        "documents": final_documents,
        "final_query": current_query,
        "retrieval_attempts": len(retrieval_trace),
        "retry_count": MAX_RETRIEVAL_RETRIES,
        "retrieval_sufficient": False,
        "assessment": (
            final_assessment.model_dump()
            if final_assessment
            else None
        ),
        "retrieval_trace": retrieval_trace,
    }


def generate_career_advice(
    student_id: str,
) -> dict[str, Any]:
    """
    Generate evidence-based career recommendations.

    Combines:
        - Verified SQLite learner information.
        - Agentic RAG retrieval.
        - Retrieval self-reflection.
        - Structured Gemini career advice.

    Retrieval is retried at most twice when evidence
    is insufficient.
    """

    # 1. Retrieve learner profile.
    profile_result = get_learner_profile(student_id)

    if profile_result.get("status") != "success":
        return {
            "status": "error",
            "message": "Unable to verify learner information.",
        }

    if profile_result.get("row_count", 0) == 0:
        return {
            "status": "error",
            "message": "Learner ID not found.",
        }

    learner = profile_result["data"][0]

    # 2. Normalize the database track field.
    learner["current_track"] = learner.get(
        "bootcamp_track",
        "Unknown track",
    )

    # 3. Build the initial retrieval query.
    learner_query = (
        f"Career recommendations for a learner in "
        f"{learner['current_track']}. "
        f"Relevant technical skills, completed modules, "
        f"assessment scores, career roles, skill gaps, "
        f"and learning roadmap for this track."
    )

    # 4. Execute Agentic RAG with self-reflection.
    try:
        retrieval_result = agentic_retrieve_career_knowledge(
            initial_query=learner_query,
            learner=learner,
        )

    except Exception as error:
        return {
            "status": "error",
            "message": (
                "Unable to verify career knowledge retrieval."
            ),
            "details": str(error),
        }

    if retrieval_result["status"] != "success":
        return {
            "status": "error",
            "message": (
                "Unable to verify career recommendations: "
                "retrieved knowledge remained insufficient "
                "after the maximum retrieval attempts."
            ),
            "retrieval": retrieval_result,
        }

    knowledge_documents = retrieval_result["documents"]

    if not knowledge_documents:
        return {
            "status": "error",
            "message": (
                "Unable to verify career recommendations: "
                "no relevant knowledge was retrieved."
            ),
            "retrieval": retrieval_result,
        }

    # 5. Prepare retrieved context.
    retrieved_context = "\n\n".join(
        [
            (
                f"Source: {item['source']}\n"
                f"Trust level: {item['trust_level']}\n"
                f"Content:\n{item['content']}"
            )
            for item in knowledge_documents
        ]
    )

    # 6. Deduplicate source metadata.
    unique_sources = []
    seen_sources = set()

    for item in knowledge_documents:

        source_key = (
            item["source"],
            item["trust_level"],
        )

        if source_key not in seen_sources:
            seen_sources.add(source_key)

            unique_sources.append(
                {
                    "source": item["source"],
                    "trust_level": item["trust_level"],
                }
            )

    # 7. Prepare evidence-based career advice prompt.
    prompt = f"""
You are a professional AI Career Advisor for Sprints learners.

VERIFIED LEARNER PROFILE:
{learner}

RETRIEVED KNOWLEDGE:
{retrieved_context}

RETRIEVAL QUALITY:
{retrieval_result['assessment']}

INSTRUCTIONS:

1. Use the verified learner profile as the source of learner facts.

2. The learner's current track is:
   {learner['current_track']}

3. Recommend suitable entry-level roles based on the
   available learner information.

4. Identify skill gaps only when supported by evidence.
   Clearly state when information is insufficient.

5. Create a practical learning roadmap.

6. Consider preferred employers when making recommendations.

7. Analyze the target salary without inventing market statistics.

8. Never invent qualifications, certifications, work experience,
   completed courses, technical skills, or assessment results.

9. Retrieved project requirements must not be treated as
   verified employer hiring requirements.

10. Do not claim that an employer has an active vacancy.
    No live job-opening database is currently connected.

11. Distinguish verified facts from AI-generated recommendations.

12. Return all fields required by the CareerAdvice schema.

13. Use the exact learner name and current track from the
    verified learner profile.
"""

    # 8. Generate structured career advice.
    try:

        advisor = create_advisor_agent()

        result = advisor.invoke(prompt)

        return {
            "status": "success",
            "student_id": student_id,
            "learner_profile": learner.copy(),
            "advice": result.model_dump(),
            "sources": unique_sources,
            "retrieval": {
                "retrieval_sufficient": True,
                "retrieval_attempts": (
                    retrieval_result["retrieval_attempts"]
                ),
                "retry_count": retrieval_result["retry_count"],
                "final_query": retrieval_result["final_query"],
                "assessment": retrieval_result["assessment"],
                "retrieval_trace": retrieval_result["retrieval_trace"],
            },
            "evidence": {
                "learner_profile_verified": True,
                "knowledge_retrieved": True,
                "retrieval_quality_evaluated": True,
                "active_vacancy_verified": False,
            },
        }

    except Exception as error:

        return {
            "status": "error",
            "message": "Unable to verify career advice.",
            "details": str(error),
        }