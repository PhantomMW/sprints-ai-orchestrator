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

def test_rag_retrieval_across_multiple_bootcamp_tracks():
    """Verify RAG retrieval evaluation across diverse curriculum tracks."""
    module = "src.agents.advisor_agent"
    from src.agents.advisor_agent import RetrievalAssessment, agentic_retrieve_career_knowledge

    tracks_to_test = [
        ("Advanced Agentic AI", ["LangGraph", "Multi-Agent Orchestration"]),
        ("Data Engineering", ["PySpark", "Kafka", "Data Modeling"]),
        ("Cyber Security Operations", ["SIEM", "Incident Response", "Network Defense"]),
        ("Generative AI Engineering", ["Fine-tuning", "RAG Architecture", "Prompting"]),
        ("Full Stack Web Development", ["React", "FastAPI", "PostgreSQL"]),
    ]

    for track_name, expected_skills in tracks_to_test:
        sufficient_eval = RetrievalAssessment(
            sufficient=True,
            relevance_score=90,
            reasoning=f"High match for {track_name} skills.",
            missing_information=[],
        )
        mock_docs = [
            {"content": f"Curriculum for {track_name} covering {', '.join(expected_skills)}", "source": "project_brief.md", "trust_level": "high"}
        ]
        with (
            patch(f"{module}.retrieve_career_knowledge", return_value=mock_docs),
            patch(f"{module}.assess_retrieval_quality", return_value=sufficient_eval),
        ):
            res = agentic_retrieve_career_knowledge(
                initial_query=f"Career paths for {track_name}",
                learner={"current_track": track_name},
            )
            assert res["status"] == "success"
            assert res["retrieval_sufficient"] is True
            assert len(res["documents"]) > 0


def test_rag_evaluates_trust_levels_and_source_prioritization():
    """Verify that RAG documents carry verified trust levels and source metadata."""
    module = "src.agents.advisor_agent"
    from src.agents.advisor_agent import RetrievalAssessment, agentic_retrieve_career_knowledge

    mock_docs = [
        {"content": "Primary project brief requirements", "source": "project_brief.md", "trust_level": "high"},
        {"content": "Kickoff deck summary notes", "source": "client_kickoff_deck.md", "trust_level": "medium"},
    ]
    assessment = RetrievalAssessment(sufficient=True, relevance_score=85, reasoning="Good coverage.", missing_information=[])

    with (
        patch(f"{module}.retrieve_career_knowledge", return_value=mock_docs),
        patch(f"{module}.assess_retrieval_quality", return_value=assessment),
    ):
        res = agentic_retrieve_career_knowledge(
            initial_query="Placement requirements",
            learner={"current_track": "Advanced Agentic AI"},
        )
        assert res["status"] == "success"
        sources = {d["source"]: d["trust_level"] for d in res["documents"]}
        assert sources["project_brief.md"] == "high"
        assert sources["client_kickoff_deck.md"] == "medium"


def test_rag_detects_out_of_domain_queries():
    """Verify that queries unrelated to tech career advice receive low scores and trigger reformulations."""
    module = "src.agents.advisor_agent"
    from src.agents.advisor_agent import RetrievalAssessment, QueryReformulation, agentic_retrieve_career_knowledge

    irrelevant = RetrievalAssessment(
        sufficient=False,
        relevance_score=15,
        reasoning="Query regarding cooking recipes has no relevance to software engineering.",
        missing_information=["Technical skills", "Career pathways"],
    )
    reform = QueryReformulation(
        improved_query="Advanced Agentic AI technical career roadmap and employer partnerships",
        reasoning="Redirecting out-of-domain query back to verified curriculum domain.",
    )

    with (
        patch(f"{module}.retrieve_career_knowledge", return_value=[{"content": "irrelevant text", "source": "unknown", "trust_level": "low"}]),
        patch(f"{module}.assess_retrieval_quality", return_value=irrelevant),
        patch(f"{module}.reformulate_retrieval_query", return_value=reform),
    ):
        res = agentic_retrieve_career_knowledge(
            initial_query="How to cook pasta carbonara",
            learner={"current_track": "Advanced Agentic AI"},
        )
        assert res["status"] == "insufficient_evidence"
        assert res["retrieval_attempts"] == 3
