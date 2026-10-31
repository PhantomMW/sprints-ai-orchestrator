"""
Test suite verifying quantitative RAG retrieval benchmarks.
Enforces Precision @ 3 >= 0.80, Domain Recall = 1.0, Factual Grounding >= 0.90, and Latency < 1000ms.
"""

from unittest.mock import patch
import pytest
from src.tools.rag_retriever import CHROMA_DIR
from scripts.benchmark_rag import BENCHMARK_DATASET, run_rag_benchmark

CHROMA_EXISTS = CHROMA_DIR.exists() and (CHROMA_DIR / "chroma.sqlite3").exists()


def test_rag_benchmark_metric_computation():
    """Verify quantitative metric computation (precision@k, recall, factual grounding, provenance) operates accurately."""
    from langchain_core.documents import Document

    def fake_retriever(query: str, k: int = 3):
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
        assert summary["domain_recall"] == 1.0
        assert summary["mean_precision_at_k"] >= 0.80


@pytest.mark.skipif(
    not CHROMA_EXISTS,
    reason="ChromaDB storage not present in CI checkout (local data excluded from git). Run document ingestion locally."
)
def test_rag_retrieval_benchmark_precision_and_recall():
    """Verify live ChromaDB retrieval achieves high precision, full domain recall, factual grounding, and low latency."""
    summary = run_rag_benchmark(k=3)
    assert summary["mean_precision_at_k"] >= 0.80, (
        f"RAG Mean Precision @ 3 ({summary['mean_precision_at_k'] * 100:.1f}%) below 80.0%"
    )
    assert summary["domain_recall"] == 1.0
    assert summary["provenance_ratio"] == 1.0
    assert summary["factual_grounding_ratio"] >= 0.90
    assert summary["strict_groundedness_ratio"] >= 0.90
    assert summary["mean_latency_ms"] < 1000.0
