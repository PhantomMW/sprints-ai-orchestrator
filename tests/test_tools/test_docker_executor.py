from io import BytesIO

from src.tools.docker_executor import (
    MAX_OUTPUT_BYTES,
    _read_limited,
    execute_in_docker,
)


def test_read_limited_keeps_only_configured_bytes():
    payload = b"x" * (MAX_OUTPUT_BYTES + 100)

    captured, truncated = _read_limited(BytesIO(payload), MAX_OUTPUT_BYTES)

    assert len(captured) == MAX_OUTPUT_BYTES
    assert truncated is True


def test_read_limited_reports_untruncated_stream():
    payload = b"assessment output"

    captured, truncated = _read_limited(BytesIO(payload), MAX_OUTPUT_BYTES)

    assert captured == payload
    assert truncated is False


def test_empty_submission_is_rejected_without_docker():
    result = execute_in_docker("  ")

    assert result["status"] == "error"
    assert result["stdout"] == ""
    assert result["stderr"] == ""
    assert result["output_truncated"] is False
