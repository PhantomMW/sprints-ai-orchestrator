import json
import logging
from src.utils.logger import JSONFormatter, get_structured_logger, log_event


def test_json_formatter_outputs_valid_json():
    """Verify that JSONFormatter produces valid structured JSON records."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_component",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test structured log message",
        args=(),
        exc_info=None,
    )
    record.event = "TEST_EVENT"
    record.trace_id = "trace-12345"
    record.duration_ms = 45.2
    record.details = {"user": "admin", "items": 3}

    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["component"] == "test_component"
    assert parsed["message"] == "Test structured log message"
    assert parsed["event"] == "TEST_EVENT"
    assert parsed["trace_id"] == "trace-12345"
    assert parsed["duration_ms"] == 45.2
    assert parsed["details"]["user"] == "admin"


def test_log_event_executes_without_error():
    """Verify log_event helper runs cleanly across different log levels."""
    log_event(
        event="ROUTER_EVALUATION",
        component="router_node",
        message="Request routed to advisor",
        trace_id="req-999",
        details={"agent": "advisor"},
        duration_ms=12.5,
        status="info",
    )
