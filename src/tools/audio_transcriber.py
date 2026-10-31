import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_DIR = PROJECT_ROOT / "data" / "curriculum_docs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

AUDIO_FILE = AUDIO_DIR / "client_discovery_call.mp3"


def transcribe_discovery_call() -> Path:
    """
    Transcribe the client discovery call using Gemini.

    Preserves the conversation's original meaning and details.
    The transcript is saved as a Markdown document for RAG ingestion.

    Returns:
        Path to the generated transcript file.

    Raises:
        ValueError: If the API key or audio file is missing.
        RuntimeError: If Gemini transcription fails.
    """

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Check your .env file."
        )

    if not AUDIO_FILE.exists():
        raise ValueError(
            f"Audio file not found: {AUDIO_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)

    print("Uploading discovery call to Gemini...")

    try:
        uploaded_file = client.files.upload(
            file=str(AUDIO_FILE),
            config=types.UploadFileConfig(
                mime_type="audio/mpeg"
            )
        )

        print("Audio uploaded successfully.")
        print("Transcribing discovery call...")

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                uploaded_file,
                """
                Transcribe this entire client discovery call.

                Requirements:
                1. Preserve the original spoken language.
                2. Do not summarize or omit details.
                3. Preserve technical terminology and project names.
                4. Identify different speakers when possible.
                5. Use timestamps when they can be identified reliably.
                6. Mark unclear speech as [Unclear].
                7. Do not invent missing words or information.
                8. Return the complete transcript only.
                """
            ]
        )

        transcript = response.text

        if not transcript or not transcript.strip():
            raise RuntimeError(
                "Gemini returned an empty transcript."
            )

        output_file = (
            OUTPUT_DIR / "client_discovery_call.md"
        )

        content = (
            "# Client Discovery Call Transcript\n\n"
            "Source: client_discovery_call.mp3\n\n"
            "Source Type: Client Discovery Conversation\n\n"
            "Transcription Method: Gemini Audio Understanding\n\n"
            "---\n\n"
            f"{transcript.strip()}\n"
        )

        output_file.write_text(
            content,
            encoding="utf-8"
        )

        print("\nTranscription completed successfully.")
        print(f"Saved to: {output_file}")
        print(f"Transcript characters: {len(transcript)}")

        return output_file

    except Exception as error:
        raise RuntimeError(
            f"Discovery call transcription failed: {error}"
        ) from error


if __name__ == "__main__":
    transcribe_discovery_call()