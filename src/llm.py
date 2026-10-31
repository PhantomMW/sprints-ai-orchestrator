import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


def get_llm(temperature: float = 0):
    """
    Initialize and return the Gemini language model.

    Args:
        temperature: Controls response randomness.

    Returns:
        Configured ChatGoogleGenerativeAI instance.

    Raises:
        ValueError: If GOOGLE_API_KEY is missing.
    """

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY is missing. Check your .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=temperature,
    )