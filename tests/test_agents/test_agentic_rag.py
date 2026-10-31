from unittest.mock import patch

from src.agents.advisor_agent import (
    RetrievalAssessment,
    agentic_retrieve_career_knowledge,
)


def test_agentic_rag_retries_when_evidence_is_insufficient():
    """Verify that RAG retries twice and stops after three attempts."""

    insufficient = RetrievalAssessment(
        sufficient=False,
        relevance_score=30,
        reasoning="Retrieved documents are not sufficiently relevant.",
        missing_information=["Track-specific technical skills."],
    )

    module = "src.agents.advisor_agent"

    learner = {
        "current_track": "Advanced Agentic AI",
    }

    documents = [
        {
            "content": "General project information.",
            "source": "project_brief.md",
            "trust_level": "high",
        }
    ]

    with (
        patch(
            f"{module}.retrieve_career_knowledge",
            return_value=documents,
        ) as retriever,
        patch(
            f"{module}.assess_retrieval_quality",
            return_value=insufficient,
        ),
        patch(
            f"{module}.reformulate_retrieval_query",
        ) as reformulator,
    ):

        from src.agents.advisor_agent import QueryReformulation

        reformulator.return_value = QueryReformulation(
            improved_query="Advanced Agentic AI technical skills and roles",
            reasoning="Focus the search on track-specific evidence.",
        )

        result = agentic_retrieve_career_knowledge(
            initial_query="Career recommendations",
            learner=learner,
        )

    assert result["status"] == "insufficient_evidence"
    assert result["retrieval_attempts"] == 3
    assert result["retry_count"] == 2
    assert retriever.call_count == 3
    assert reformulator.call_count == 2


def test_agentic_rag_stops_when_evidence_is_sufficient():
    """Verify that RAG stops immediately when evidence is sufficient."""

    sufficient = RetrievalAssessment(
        sufficient=True,
        relevance_score=95,
        reasoning="Retrieved documents contain relevant evidence.",
        missing_information=[],
    )

    module = "src.agents.advisor_agent"

    documents = [
        {
            "content": "Relevant career information.",
            "source": "project_brief.md",
            "trust_level": "high",
        }
    ]

    with (
        patch(
            f"{module}.retrieve_career_knowledge",
            return_value=documents,
        ) as retriever,
        patch(
            f"{module}.assess_retrieval_quality",
            return_value=sufficient,
        ),
        patch(
            f"{module}.reformulate_retrieval_query",
        ) as reformulator,
    ):

        result = agentic_retrieve_career_knowledge(
            initial_query="Career recommendations",
            learner={"current_track": "Advanced Agentic AI"},
        )

    assert result["status"] == "success"
    assert result["retrieval_attempts"] == 1
    assert result["retry_count"] == 0
    assert result["retrieval_sufficient"] is True

    retriever.assert_called_once()
    reformulator.assert_not_called()