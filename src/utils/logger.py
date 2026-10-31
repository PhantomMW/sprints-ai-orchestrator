import json
import logging
import os
import sys
import time
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as structured single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }

        # Include custom structured attributes if provided
        for attr in ("event", "trace_id", "details", "duration_ms", "status"):
            val = getattr(record, attr, None)
            if val is not None:
                log_payload[attr] = val

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload, default=str)


def get_structured_logger(name: str = "sprints_orchestrator") -> logging.Logger:
    """
    Configure and return a structured JSON logger for observability.

    Log levels can be configured via the LOG_LEVEL environment variable (default: INFO).
    """
    logger = logging.getLogger(name)
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ"))
        logger.addHandler(handler)

    return logger


# Global default logger instance
logger = get_structured_logger()


def log_event(
    event: str,
    component: str,
    message: str,
    trace_id: str | None = None,
    details: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    status: str = "info",
) -> None:
    """Log an observable system event with structured telemetry attributes."""
    extra = {
        "event": event,
        "trace_id": trace_id,
        "details": details or {},
        "duration_ms": duration_ms,
        "status": status,
    }
    log_func = getattr(logger, status.lower(), logger.info)
    log_func(f"[{component}] {message}", extra=extra)
