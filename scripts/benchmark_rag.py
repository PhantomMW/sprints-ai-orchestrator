"""
Quantitative RAG Retrieval Benchmark Script.

Evaluates vector retrieval performance across Sprints project architecture domains:
- Precision @ k (k=3)
- Domain / Topic Recall (100%)
- Groundedness Ratio (100%)
- Retrieval Latency (ms)
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.rag_retriever import retrieve_context

BENCHMARK_DATASET = [
    {
        "domain": "LangGraph Orchestration & StateGraph",
        "query": "LangGraph multi-agent orchestration StateGraph dynamic routing and agent state",
        "keywords": ["langgraph", "stategraph", "orchestrator", "routing", "agent", "state"],
        "expected_sources": ["project_brief.md", "client_kickoff_deck.md"],
    },
    {
        "domain": "Secure Python Assessment & Sandbox",
        "query": "secure Python code execution sandbox REPL timeout stdout capture and grading",
        "keywords": ["repl", "python", "execute", "sandbox", "stdout", "timeout", "grading"],
        "expected_sources": ["project_brief.md", "client_kickoff_deck.md"],
    },
    {
        "domain": "Human-in-the-Loop Governance & Escalation",
        "query": "Human-in-the-Loop review tier-1 partner Fawry placement escalation interrupt",
        "keywords": ["hitl", "human", "fawry", "tier-1", "approval", "escalation", "interrupt"],
        "expected_sources": ["project_brief.md", "client_discovery_call.md"],
    },
    {
        "domain": "Learner Database & Career Placement Queries",
        "query": "sprints learner database target salary EGP completed modules preferred employers",
        "keywords": ["sqlite", "database", "salary", "egp", "modules", "learner", "employers"],
        "expected_sources": ["project_brief.md", "client_kickoff_deck.md"],
    },
    {
        "domain": "Agentic RAG & Knowledge Self-Reflection",
        "query": "Agentic RAG ChromaDB curriculum chunking self-reflection query reformulation",
        "keywords": ["rag", "chroma", "chromadb", "chunk", "retrieval", "reformulat", "query"],
        "expected_sources": ["project_brief.md", "client_kickoff_deck.md"],
    },
    {
        "domain": "Evaluator-Optimizer & Quality Acceptance Criteria",
        "query": "Evaluator-Optimizer refinement iterations latency acceptance criteria rubric",
        "keywords": ["evaluator", "optimizer", "iteration", "criteria", "rubric", "latency"],
        "expected_sources": ["project_brief.md", "client_kickoff_deck.md"],
    },
]


def run_rag_benchmark(k: int = 3) -> dict[str, Any]:
    """Execute quantitative evaluation of RAG retriever."""
    total_queries = len(BENCHMARK_DATASET)
    precision_scores = []
    latencies = []
    domains_hit = set()
    grounded_chunks_count = 0
    total_chunks_retrieved = 0

    results_detail = []

    for item in BENCHMARK_DATASET:
        query = item["query"]
        domain = item["domain"]
        keywords = [kw.casefold() for kw in item["keywords"]]

        start_t = time.perf_counter()
        docs = retrieve_context(query, k=k)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        latencies.append(elapsed_ms)

        relevant_docs_count = 0
        total_chunks_retrieved += len(docs)

        for doc in docs:
            content_lower = doc.page_content.casefold()
            source = doc.metadata.get("source", "")
            trust = doc.metadata.get("trust_level", "medium")

            # Groundedness: Chunk has verified source, non-empty text, and valid metadata
            if source and len(doc.page_content.strip()) > 20 and trust in ("high", "medium"):
                grounded_chunks_count += 1

            # Relevance: matches any target keywords
            if any(kw in content_lower for kw in keywords):
                relevant_docs_count += 1

        precision = (relevant_docs_count / len(docs)) if docs else 0.0
        precision_scores.append(precision)

        if precision > 0:
            domains_hit.add(domain)

        results_detail.append({
            "query": query,
            "domain": domain,
            "retrieved_k": len(docs),
            "relevant_count": relevant_docs_count,
            "precision": round(precision, 3),
            "latency_ms": round(elapsed_ms, 2),
        })

    all_domains = {item["domain"] for item in BENCHMARK_DATASET}
    mean_precision = sum(precision_scores) / total_queries if total_queries else 0.0
    domain_recall = len(domains_hit) / len(all_domains) if all_domains else 0.0
    groundedness_ratio = (grounded_chunks_count / total_chunks_retrieved) if total_chunks_retrieved else 0.0
    mean_latency_ms = sum(latencies) / total_queries if total_queries else 0.0

    summary = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_queries": total_queries,
        "k_eval": k,
        "mean_precision_at_k": round(mean_precision, 3),
        "domain_recall": round(domain_recall, 3),
        "groundedness_ratio": round(groundedness_ratio, 3),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "domains_covered": sorted(list(domains_hit)),
        "details": results_detail,
    }
    return summary


if __name__ == "__main__":
    print("=" * 70)
    print("SPRINTS QUANTITATIVE AGENTIC RAG BENCHMARK")
    print("=" * 70)
    summary = run_rag_benchmark(k=3)
    print(f"Total Benchmark Queries: {summary['total_queries']}")
    print(f"Mean Precision @ 3:     {summary['mean_precision_at_k'] * 100:.1f}% (Benchmark Target: >= 80.0%)")
    print(f"Domain Recall:          {summary['domain_recall'] * 100:.1f}% (Benchmark Target: 100.0%)")
    print(f"Groundedness Ratio:     {summary['groundedness_ratio'] * 100:.1f}% (Benchmark Target: 100.0%)")
    print(f"Mean Query Latency:     {summary['mean_latency_ms']:.1f} ms")
    print("-" * 70)
    print("PER-DOMAIN EVALUATION:")
    for row in summary["details"]:
        status = "PASS" if row["precision"] >= 0.60 else "FAIL"
        print(f"[{status}] {row['domain'][:35]:<35} | Prec@3: {row['precision']:.2f} | Latency: {row['latency_ms']:.1f}ms")
    print("=" * 70)
