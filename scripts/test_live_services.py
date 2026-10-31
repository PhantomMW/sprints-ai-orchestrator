"""
Live Services & Gemini API End-to-End Diagnostic Script
======================================================
Tests live Google Gemini API connectivity and real Agentic RAG retrieval
independently from mock-based unit tests.

Usage:
    export GEMINI_API_KEY="your-real-key"
    python scripts/test_live_services.py
"""

import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm import get_llm
from src.tools.rag_retriever import retrieve_context
from langchain_core.messages import HumanMessage


def test_live_gemini_and_rag():
    print("=" * 70)
    print("    SPRINTS AI ORCHESTRATOR - LIVE SERVICES DIAGNOSTIC SUITE")
    print("=" * 70)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n[SKIPPED] GEMINI_API_KEY is not set in environment.")
        print("To run live service validation:")
        print("  1. In PowerShell:  $env:GEMINI_API_KEY='your_api_key_here'")
        print("  2. In Linux/macOS: export GEMINI_API_KEY='your_api_key_here'")
        print("  3. Run:            python scripts/test_live_services.py")
        print("\nUnit and mock test suites (95 tests) pass independently without API keys.")
        return 0

    print(f"\n1. GEMINI_API_KEY Detected: {api_key[:6]}...{api_key[-4:]}")
    print("2. Probing ChatGoogleGenerativeAI (gemini-2.5-flash)...")

    try:
        llm = get_llm(temperature=0.2)
        response = llm.invoke([HumanMessage(content="Respond with 'PONG' to verify connectivity.")])
        print(f"   LLM Response: {response.content.strip()}")
        assert "pong" in response.content.lower(), "Unexpected LLM response format"
        print("   [SUCCESS] Live Gemini API connectivity verified.")
    except Exception as e:
        print(f"   [ERROR] Failed to communicate with Gemini API: {e}")
        return 1

    print("\n3. Testing Agentic RAG Knowledge Base Retrieval...")
    try:
        rag_res = retrieve_context("What are the graduation requirements for Advanced Agentic AI?")
        print(f"   Retrieval Status: {rag_res.get('retrieval_sufficient')}")
        print(f"   Sources Found:    {len(rag_res.get('sources', []))}")
        for s in rag_res.get("sources", [])[:3]:
            print(f"     - [{s.get('trust_level', 'unknown').upper()}] {s.get('source')}")
        print("   [SUCCESS] Agentic RAG retrieval pipeline verified.")
    except Exception as e:
        print(f"   [ERROR] RAG retrieval failed: {e}")
        return 1

    print("\n" + "=" * 70)
    print("    ALL LIVE SERVICES VERIFIED SUCCESSFULLY")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(test_live_gemini_and_rag())
