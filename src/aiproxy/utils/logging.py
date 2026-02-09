"""Structured logging helpers."""
import json
import logging
import os
import random
import sys
import time
from logging.handlers import RotatingFileHandler

from ..core.config import get_logging_config

logger = logging.getLogger("aiproxy")
_file_logger = None


def _resolve_log_dir(log_dir: str | None) -> str | None:
    if log_dir:
        return log_dir
    env_dir = os.getenv("LOG_DIR")
    if env_dir:
        return env_dir
    return None


def _file_logging_enabled() -> bool:
    return os.getenv("LOG_TO_FILE", "True").lower() in ("1", "true", "yes", "on")


def _build_rotating_handler(log_dir: str, filename: str) -> RotatingFileHandler:
    os.makedirs(log_dir, exist_ok=True)
    max_mb = float(os.getenv("LOG_FILE_MAX_MB", "10"))
    backup_count = int(os.getenv("LOG_FILE_BACKUPS", "5"))
    max_bytes = max(1, int(max_mb * 1024 * 1024))
    backup_count = max(1, backup_count)
    log_path = os.path.join(log_dir, filename)
    handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def setup_logging(level: str, log_dir: str | None = None) -> None:
    """Configure root logging level and handlers."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    resolved_dir = _resolve_log_dir(log_dir)
    if resolved_dir and _file_logging_enabled():
        try:
            handlers.append(_build_rotating_handler(resolved_dir, "aiproxy.log"))
        except Exception as exc:
            # Fall back to stdout-only if file logging can't be initialized.
            print(f"[aiproxy] file logging disabled: {exc}", file=sys.stderr)
    # Force handlers so docker/compose can always capture logs.
    logging.basicConfig(level=numeric, handlers=handlers, force=True)


def get_file_logger(log_dir: str, filename: str = "chat_completions.log") -> logging.Logger:
    """Return a rotating file logger for request logs."""
    global _file_logger
    if _file_logger is not None:
        return _file_logger
    if not _file_logging_enabled():
        null_logger = logging.getLogger("aiproxy.file.null")
        null_logger.addHandler(logging.NullHandler())
        null_logger.propagate = False
        _file_logger = null_logger
        return _file_logger
    handler = _build_rotating_handler(log_dir, filename)
    file_logger = logging.getLogger("aiproxy.file")
    file_logger.setLevel(logging.INFO)
    file_logger.addHandler(handler)
    file_logger.propagate = False
    _file_logger = file_logger
    return file_logger


def log_event(level: int, message: str, **fields) -> None:
    """Emit a structured log line."""
    payload = {"message": message, "ts": int(time.time())}
    payload.update(fields)
    logger.log(level, json.dumps(payload, ensure_ascii=False))


def redact_headers(headers, redact_list):
    """Return headers dict with sensitive keys masked."""
    redact_set = {item.lower() for item in redact_list}
    safe = {}
    for key, value in headers.items():
        if key.lower() in redact_set:
            safe[key] = "***"
        else:
            safe[key] = value
    return safe


def redact_payload(payload, redact_keys):
    """Recursively mask sensitive keys in JSON-like payloads."""
    if isinstance(payload, dict):
        masked = {}
        for key, value in payload.items():
            if key in redact_keys:
                masked[key] = "***"
            else:
                masked[key] = redact_payload(value, redact_keys)
        return masked
    if isinstance(payload, list):
        return [redact_payload(item, redact_keys) for item in payload]
    return payload


def should_log_request(status_code: int) -> bool:
    """Apply sampling rules; always log errors."""
    if status_code >= 400:
        return True
    cfg = get_logging_config()
    sample_rate = cfg.get("sample_rate", 1.0)
    return random.random() <= sample_rate
