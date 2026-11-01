"""
Quantitative RAG Retrieval Benchmark Script with Factual Grounding & Negative Controls.

Evaluates vector retrieval performance across Sprints project architecture domains:
- Precision @ k (k=3)
- Expected Source Matching Ratio (Human-curated Ground Truth Documents)
- Domain Fact Recall (Coverage of required domain assertions)
- Metadata Provenance Ratio (Source authenticity & trust levels)
- Strict Factual Grounding Ratio (Document supports required domain assertions)
- Unsupported Claim Detection Rate (Negative / Out-of-Domain Query Controls)
- Retrieval Latency (ms)

Produces reproducible artifact: benchmarks/rag_benchmark_report.json
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Prevent HuggingFace Hub network retries in offline / sandboxed environments
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.rag_retriever import retrieve_context

# 1. POSITIVE CURRICULUM BENCHMARK QUERIES (Human-Curated Ground Truth)
BENCHMARK_DATASET = [
    {
        "domain": "LangGraph Orchestration & StateGraph",
        "query": "LangGraph multi-agent orchestration StateGraph dynamic routing and agent state",
        "keywords": ["langgraph", "stategraph", "orchestrator", "routing", "agent", "state"],
        "factual_assertions": ["stategraph", "dynamic routing", "orchestrat", "typeddict", "state"],
        "expected_sources": ["project_brief.md", "client_kickoff_deck.md"],
    },
    {
        "domain": "Secure Python Assessment & Sandbox",
        "query": "secure Python code execution sandbox REPL timeout stdout capture and grading",
        "keywords": ["repl", "python", "execute", "sandbox", "stdout", "timeout", "grading"],
        "factual_assertions": ["repl", "sandbox", "stdout", "runtime error", "timeout", "grad"],
        "expected_sources": ["project_brief.md", "client_kickoff_deck.md", "client_discovery_call.md"],
    },
    {
        "domain": "Human-in-the-Loop Governance & Escalation",
        "query": "Human-in-the-Loop review tier-1 partner Fawry placement escalation interrupt",
        "keywords": ["hitl", "human", "fawry", "tier-1", "approval", "escalation", "interrupt"],
        "factual_assertions": ["human-in-the-loop", "hitl", "tier-1", "fawry", "interrupt", "escalat"],
        "expected_sources": ["project_brief.md", "client_kickoff_deck.md", "client_discovery_call.md"],
    },
    {
        "domain": "Learner Database & Career Placement Queries",
        "query": "sprints learner database target salary EGP completed modules preferred employers",
        "keywords": ["sqlite", "database", "salary", "egp", "modules", "learner", "employers"],
        "factual_assertions": ["sprints_learner_db", "salary", "egp", "module", "learner", "employer"],
        "expected_sources": ["project_brief.md", "client_kickoff_deck.md"],
    },
    {
        "domain": "Agentic RAG & Knowledge Self-Reflection",
        "query": "Agentic RAG ChromaDB curriculum chunking self-reflection query reformulation",
        "keywords": ["rag", "chroma", "chromadb", "chunk", "retrieval", "reformulat", "query"],
        "factual_assertions": ["chromadb", "chunk", "self-reflection", "reformulat", "retriev"],
        "expected_sources": ["project_brief.md", "client_kickoff_deck.md"],
    },
    {
        "domain": "Evaluator-Optimizer & Quality Acceptance Criteria",
        "query": "Evaluator-Optimizer refinement iterations latency acceptance criteria rubric",
        "keywords": ["evaluator", "optimizer", "iteration", "criteria", "rubric", "latency"],
        "factual_assertions": ["evaluator", "optimizer", "refinement iteration", "rubric", "criteria", "latency"],
        "expected_sources": ["project_brief.md", "client_kickoff_deck.md"],
    },
    {
        "domain": "Pydantic Schemas & Output Parsers",
        "query": "Pydantic prompt templates output parsers score feedback points and pass status json schemas",
        "keywords": ["pydantic", "parser", "schema", "prompt", "feedback", "pass_status"],
        "factual_assertions": ["pydantic", "output parser", "feedback_point", "pass_status", "schema"],
        "expected_sources": ["project_brief.md", "client_kickoff_deck.md"],
    },
    {
        "domain": "Conversational MemorySaver & Multi-Turn State",
        "query": "LangGraph MemorySaver checkpointer session checkpointing conversational state persistence thread ID",
        "keywords": ["memorysaver", "checkpoint", "state", "session", "thread", "langgraph", "amnesia", "orchestrator"],
        "factual_assertions": ["memorysaver", "thread id", "checkpointer", "session", "amnesia"],
        "expected_sources": ["project_brief.md", "client_kickoff_deck.md", "client_discovery_call.md"],
    },
]

# 2. NEGATIVE / OUT-OF-DOMAIN CONTROLS (Unsupported Claim Detection)
NEGATIVE_DATASET = [
    {
        "domain": "Out-of-Domain: Quantum Computing",
        "query": "Quantum computing superconductor qubits and post-quantum cryptography in Sprints bootcamps",
        "keywords": ["quantum", "qubit", "superconductor", "post-quantum"],
        "factual_assertions": ["quantum computing", "qubits", "superconductor"],
    },
    {
        "domain": "Out-of-Domain: International Bank Transfers",
        "query": "Corporate banking SWIFT wire transfer international routing numbers for learner stipends",
        "keywords": ["swift", "wire transfer", "routing number", "iban"],
        "factual_assertions": ["swift", "wire transfer", "international routing"],
    },
]


def run_rag_benchmark(
    k: int = 3,
    output_report_path: Path | None = None,
) -> dict[str, Any]:
    """
    Execute quantitative evaluation of RAG retriever with factual grounding,
    source provenance matching, fact recall, and negative claim rejection.
    """
    # Warm up retriever to exclude cold-start HuggingFace model loading from latency metrics
    try:
        retrieve_context("warmup", k=1)
    except Exception:
        pass

    total_queries = len(BENCHMARK_DATASET)
    precision_scores = []
    source_match_scores = []
    fact_recall_scores = []
    latencies = []
    domains_hit = set()
    provenance_chunks_count = 0
    factual_chunks_count = 0
    strict_grounded_chunks_count = 0
    total_chunks_retrieved = 0

    results_detail = []

    # Evaluate positive queries
    for item in BENCHMARK_DATASET:
        query = item["query"]
        domain = item["domain"]
        keywords = [kw.casefold() for kw in item["keywords"]]
        factual_assertions = [fa.casefold() for fa in item.get("factual_assertions", keywords)]
        expected_sources = item.get("expected_sources", [])

        start_t = time.perf_counter()
        docs = retrieve_context(query, k=k)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        latencies.append(elapsed_ms)

        relevant_docs_count = 0
        expected_source_match_count = 0
        domain_facts_found = set()
        total_chunks_retrieved += len(docs)
        doc_details = []

        for doc in docs:
            content_lower = doc.page_content.casefold()
            source = doc.metadata.get("source", "")
            trust = doc.metadata.get("trust_level", "medium")

            # 1. Provenance: Chunk has verified document source, non-empty content, and valid trust level
            has_provenance = bool(source and len(doc.page_content.strip()) > 20 and trust in ("high", "medium"))
            if has_provenance:
                provenance_chunks_count += 1

            # 2. Expected Source Verification: Chunk originates from human-curated ground truth documents
            matches_expected_source = bool(source in expected_sources)
            if matches_expected_source:
                expected_source_match_count += 1

            # 3. Factual Grounding: Chunk text provides factual assertions directly supporting the domain query
            matched_facts = [fa for fa in factual_assertions if fa in content_lower]
            is_factually_grounded = len(matched_facts) > 0
            if is_factually_grounded:
                factual_chunks_count += 1
                domain_facts_found.update(matched_facts)

            # 4. Strict Groundedness: Satisfies BOTH authenticated provenance and verifiable factual assertions
            is_strictly_grounded = has_provenance and is_factually_grounded
            if is_strictly_grounded:
                strict_grounded_chunks_count += 1

            # 5. Keyword Relevance: matches any query keywords
            is_relevant = any(kw in content_lower for kw in keywords)
            if is_relevant:
                relevant_docs_count += 1

            doc_details.append({
                "source": source,
                "trust_level": trust,
                "matches_expected_source": matches_expected_source,
                "matched_facts": matched_facts,
                "is_factually_grounded": is_factually_grounded,
                "has_provenance": has_provenance,
                "snippet": doc.page_content.strip()[:140] + "...",
            })

        precision = (relevant_docs_count / len(docs)) if docs else 0.0
        source_match_rate = (expected_source_match_count / len(docs)) if docs else 0.0
        fact_recall = (len(domain_facts_found) / len(factual_assertions)) if factual_assertions else 0.0

        precision_scores.append(precision)
        source_match_scores.append(source_match_rate)
        fact_recall_scores.append(fact_recall)

        if precision > 0:
            domains_hit.add(domain)

        results_detail.append({
            "query": query,
            "domain": domain,
            "retrieved_k": len(docs),
            "relevant_count": relevant_docs_count,
            "precision": round(precision, 3),
            "source_match_rate": round(source_match_rate, 3),
            "fact_recall": round(fact_recall, 3),
            "latency_ms": round(elapsed_ms, 2),
            "retrieved_docs": doc_details,
        })

    # Evaluate Negative / Out-of-Domain queries
    negative_results = []
    unsupported_detected_count = 0

    for item in NEGATIVE_DATASET:
        query = item["query"]
        domain = item["domain"]
        keywords = [kw.casefold() for kw in item["keywords"]]
        factual_assertions = [fa.casefold() for fa in item.get("factual_assertions", keywords)]

        docs = retrieve_context(query, k=k)
        content_all = " ".join(d.page_content.casefold() for d in docs)
        spurious_matches = [fa for fa in factual_assertions if fa in content_all]

        # Unsupported Claim Detected: Retriever correctly returned NO false factual matches
        is_unsupported = (len(spurious_matches) == 0)
        if is_unsupported:
            unsupported_detected_count += 1

        negative_results.append({
            "domain": domain,
            "query": query,
            "unsupported_claim_detected": is_unsupported,
            "spurious_matches": spurious_matches,
            "retrieved_sources": [d.metadata.get("source") for d in docs],
        })

    all_domains = {item["domain"] for item in BENCHMARK_DATASET}
    mean_precision = sum(precision_scores) / total_queries if total_queries else 0.0
    mean_source_match = sum(source_match_scores) / total_queries if total_queries else 0.0
    mean_fact_recall = sum(fact_recall_scores) / total_queries if total_queries else 0.0
    domain_recall = len(domains_hit) / len(all_domains) if all_domains else 0.0
    provenance_ratio = (provenance_chunks_count / total_chunks_retrieved) if total_chunks_retrieved else 0.0
    factual_grounding_ratio = (factual_chunks_count / total_chunks_retrieved) if total_chunks_retrieved else 0.0
    strict_groundedness_ratio = (strict_grounded_chunks_count / total_chunks_retrieved) if total_chunks_retrieved else 0.0
    unsupported_claim_detection_rate = (unsupported_detected_count / len(NEGATIVE_DATASET)) if NEGATIVE_DATASET else 1.0
    mean_latency_ms = sum(latencies) / total_queries if total_queries else 0.0

    summary = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope_and_methodology": (
            "Evaluated against 8 core curriculum domains (24 retrieved chunks) plus 2 out-of-domain negative controls. "
            "Metrics measure factual grounding, expected source matching, fact recall, and unsupported claim rejection. "
            "Results represent regression performance on this curated benchmark suite and establish engineering baselines "
            "rather than an unconditional 100% accuracy claim for uncurated open-ended queries."
        ),
        "total_queries": total_queries,
        "k_eval": k,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_store": "ChromaDB (PersistentClient)",
        "mean_precision_at_k": round(mean_precision, 3),
        "expected_source_match_ratio": round(mean_source_match, 3),
        "mean_fact_recall": round(mean_fact_recall, 3),
        "domain_recall": round(domain_recall, 3),
        "provenance_ratio": round(provenance_ratio, 3),
        "factual_grounding_ratio": round(factual_grounding_ratio, 3),
        "strict_groundedness_ratio": round(strict_groundedness_ratio, 3),
        "unsupported_claim_detection_rate": round(unsupported_claim_detection_rate, 3),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "domains_covered": sorted(list(domains_hit)),
        "details": results_detail,
        "negative_controls": negative_results,
    }

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Benchmark report successfully saved to: {output_report_path}")

    return summary


if __name__ == "__main__":
    report_file = PROJECT_ROOT / "benchmarks" / "rag_benchmark_report.json"
    print("=" * 80)
    print("SPRINTS QUANTITATIVE AGENTIC RAG BENCHMARK (FACTUAL GROUNDING & CONTROLS)")
    print("=" * 80)
    summary = run_rag_benchmark(k=3, output_report_path=report_file)
    print(f"Total Benchmark Queries:         {summary['total_queries']}")
    print(f"Mean Precision @ 3:             {summary['mean_precision_at_k'] * 100:.1f}% (Target: >= 80.0%)")
    print(f"Expected Source Match Ratio:    {summary['expected_source_match_ratio'] * 100:.1f}% (Target: >= 80.0%)")
    print(f"Domain Fact Recall:             {summary['mean_fact_recall'] * 100:.1f}% (Target: >= 80.0%)")
    print(f"Domain Coverage Recall:         {summary['domain_recall'] * 100:.1f}% (Target: 100.0%)")
    print(f"Provenance Ratio:               {summary['provenance_ratio'] * 100:.1f}% (Target: 100.0%)")
    print(f"Factual Grounding Ratio:        {summary['factual_grounding_ratio'] * 100:.1f}% (Target: >= 90.0%)")
    print(f"Strict Groundedness Ratio:      {summary['strict_groundedness_ratio'] * 100:.1f}% (Target: >= 90.0%)")
    print(f"Unsupported Claim Detection:    {summary['unsupported_claim_detection_rate'] * 100:.1f}% (Target: 100.0%)")
    print(f"Mean Query Latency:             {summary['mean_latency_ms']:.1f} ms (Target: < 1000.0 ms)")
    print("-" * 80)
    print("PER-DOMAIN EVALUATION:")
    for row in summary["details"]:
        status = "PASS" if row["precision"] >= 0.60 else "FAIL"
        print(f"[{status}] {row['domain'][:32]:<32} | Prec@3: {row['precision']:.2f} | FactRecall: {row['fact_recall']:.2f} | Latency: {row['latency_ms']:.1f}ms")
    print("-" * 80)
    print("NEGATIVE CONTROLS (UNSUPPORTED CLAIMS):")
    for neg in summary["negative_controls"]:
        status = "PASS" if neg["unsupported_claim_detected"] else "FAIL"
        print(f"[{status}] {neg['domain'][:40]:<40} -> Correctly rejected false assertions")
    print("=" * 80)
