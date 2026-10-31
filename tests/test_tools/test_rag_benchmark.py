"""
Test suite verifying quantitative RAG retrieval benchmarks.
Enforces Precision @ 3 >= 0.80, Domain Recall = 1.0, and Groundedness = 1.0.
"""

import pytest
from scripts.benchmark_rag import run_rag_benchmark


def test_rag_retrieval_benchmark_precision_and_recall():
    """Verify RAG retrieval achieves high precision, full domain recall, and perfect groundedness."""
    summary = run_rag_benchmark(k=3)

    # 1. Precision @ 3 must meet or exceed target threshold (80%)
    assert summary["mean_precision_at_k"] >= 0.80, (
        f"RAG Mean Precision @ 3 ({summary['mean_precision_at_k'] * 100:.1f}%) "
        f"is below required target of 80.0%"
    )

    # 2. Domain Recall across all curriculum architecture topics must be 100%
    assert summary["domain_recall"] == 1.0, (
        f"Domain recall ({summary['domain_recall'] * 100:.1f}%) did not achieve 100% coverage."
    )

    # 3. Groundedness ratio must be 100% (all retrieved chunks have verified sources and trust levels)
    assert summary["groundedness_ratio"] == 1.0, (
        f"Groundedness ratio ({summary['groundedness_ratio'] * 100:.1f}%) must be 100.0%."
    )

    # 4. Latency verification: vector retrieval should be fast (< 1000ms per query as per brief criteria)
    assert summary["mean_latency_ms"] < 1000.0, (
        f"Mean retrieval latency ({summary['mean_latency_ms']:.1f}ms) exceeds 1000ms SLA."
    )
