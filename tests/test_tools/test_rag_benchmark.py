"""
Test suite verifying quantitative RAG retrieval benchmarks.
Enforces Precision @ 3 >= 0.80, Domain Recall = 1.0, Factual Grounding >= 0.90,
Expected Source Match >= 0.80, Fact Recall >= 0.80, Unsupported Claim Detection = 1.0,
and Latency < 1000ms.
"""

from unittest.mock import patch
import pytest
from src.tools.rag_retriever import CHROMA_DIR
from scripts.benchmark_rag import BENCHMARK_DATASET, NEGATIVE_DATASET, run_rag_benchmark

CHROMA_EXISTS = CHROMA_DIR.exists() and (CHROMA_DIR / "chroma.sqlite3").exists()


def test_rag_benchmark_metric_computation():
    """Verify quantitative metric computation (precision, recall, factual grounding, provenance, negative controls) operates accurately."""
    from langchain_core.documents import Document

    def fake_retriever(query: str, k: int = 3):
        # If negative query, return documents without negative concepts
        if any(neg["query"] == query for neg in NEGATIVE_DATASET):
            return [
                Document(
                    page_content="Standard Sprints curriculum for agentic AI and Python",
                    metadata={"source": "project_brief.md", "trust_level": "high"}
                )
                for _ in range(k)
            ]
        return [
            Document(
                page_content=(
                    f"{query} langgraph stategraph dynamic routing typeddict agent state "
                    "python repl docker sandbox timeout stdout runtime error grading "
                    "human-in-the-loop hitl tier-1 fawry approval escalation interrupt "
                    "sprints_learner_db salary egp module learner employer "
                    "chromadb chunk self-reflection reformulat retriev "
                    "evaluator optimizer refinement iteration rubric criteria latency"
                ),
                metadata={"source": "project_brief.md", "trust_level": "high"}
            )
            for _ in range(k)
        ]

    with patch("scripts.benchmark_rag.retrieve_context", side_effect=fake_retriever):
        summary = run_rag_benchmark(k=3)
        assert summary["total_queries"] == len(BENCHMARK_DATASET)
        assert summary["k_eval"] == 3
        assert summary["provenance_ratio"] == 1.0
        assert summary["factual_grounding_ratio"] == 1.0
        assert summary["strict_groundedness_ratio"] == 1.0
        assert summary["expected_source_match_ratio"] == 1.0
        assert summary["mean_fact_recall"] >= 0.80
        assert summary["domain_recall"] == 1.0
        assert summary["mean_precision_at_k"] >= 0.80
        assert summary["unsupported_claim_detection_rate"] == 1.0


def test_rag_benchmark_unsupported_claim_detection():
    """Verify negative and out-of-domain queries correctly detect unsupported claims without hallucinating facts."""
    from langchain_core.documents import Document

    def benign_curriculum_retriever(query: str, k: int = 3):
        return [
            Document(
                page_content="Sprints software engineering curriculum covering Python, LangGraph, and Docker.",
                metadata={"source": "project_brief.md", "trust_level": "high"}
            )
            for _ in range(k)
        ]

    with patch("scripts.benchmark_rag.retrieve_context", side_effect=benign_curriculum_retriever):
        summary = run_rag_benchmark(k=3)
        assert summary["unsupported_claim_detection_rate"] == 1.0
        for neg in summary["negative_controls"]:
            assert neg["unsupported_claim_detected"] is True
            assert len(neg["spurious_matches"]) == 0


def test_rag_retrieval_benchmark_precision_and_recall():
    """Verify live ChromaDB retrieval achieves high precision, source matching, fact recall, and low latency."""
    summary = run_rag_benchmark(k=3)
    assert summary["mean_precision_at_k"] >= 0.80, (
        f"RAG Mean Precision @ 3 ({summary['mean_precision_at_k'] * 100:.1f}%) below 80.0%"
    )
    assert summary["domain_recall"] == 1.0
    assert summary["provenance_ratio"] == 1.0
    assert summary["expected_source_match_ratio"] >= 0.80
    assert summary["mean_fact_recall"] >= 0.80
    assert summary["factual_grounding_ratio"] >= 0.85
    assert summary["strict_groundedness_ratio"] >= 0.85
    assert summary["unsupported_claim_detection_rate"] == 1.0
    assert summary["mean_latency_ms"] < 1000.0
