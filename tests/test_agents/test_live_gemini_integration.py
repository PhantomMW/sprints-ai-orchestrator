import os
import pytest
from src.llm import get_llm


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GEMINI_API_KEY") == "dummy-key-for-tests",
    reason="Live Gemini API key not configured in environment.",
)
def test_live_gemini_api_connectivity():
    """
    Live Integration Test: Validates real connectivity and structured generation
    against the upstream Google Gemini API.

    This test is executed when a live GEMINI_API_KEY is available in the environment
    (e.g., in authenticated CI runs or local developer testing). It is safely skipped
    in standard offline/synthetic CI environments.
    """
    llm = get_llm(temperature=0)
    response = llm.invoke("Respond with the exact word 'VERIFIED' only.")
    assert "VERIFIED" in response.content.strip().upper()
