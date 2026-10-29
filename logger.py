# ============================================================
# WebMind Production Logger & Log Sanitizer
#
# Provides bounded, rotating log files under %LOCALAPPDATA%\WebMind\logs
# with automatic sanitization to prevent accidental credential or
# token leaks into logs.
# ============================================================

import logging
from logging.handlers import RotatingFileHandler
import os
import re
from typing import Optional

_RE_BEARER = re.compile(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{8,}", re.IGNORECASE)
_RE_SK_TOKEN = re.compile(r"sk-[a-zA-Z0-9_\-]{8,}")
_RE_COOKIE = re.compile(r"(cookie\s*:\s*)[^\r\n]+", re.IGNORECASE)
_RE_AUTH_HEADER = re.compile(r"(['\"]?authorization['\"]?\s*:\s*['\"][^'\"]+['\"])", re.IGNORECASE)


class SanitizingFilter(logging.Filter):
    """Redacts sensitive API keys, bearer tokens, and cookies from log output."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.sanitize(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(self.sanitize(str(a)) if isinstance(a, str) else a for a in record.args)
            elif isinstance(record.args, dict):
                record.args = {k: self.sanitize(str(v)) if isinstance(v, str) else v for k, v in record.args.items()}
        return True

    @staticmethod
    def sanitize(text: str) -> str:
        if not text:
            return text
        text = _RE_BEARER.sub(r"\1[REDACTED]", text)
        text = _RE_SK_TOKEN.sub(r"sk-[REDACTED]", text)
        text = _RE_COOKIE.sub(r"\1[COOKIES REDACTED]", text)
        text = _RE_AUTH_HEADER.sub(r"'authorization': '[REDACTED]'", text)
        return text


def get_log_dir() -> str:
    """Returns the persistent logs directory."""
    custom_dir = os.environ.get("WEBMIND_LOG_DIR")
    if custom_dir:
        return os.path.abspath(custom_dir)

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            base = os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")), "AppData", "Local")
        log_dir = os.path.join(base, "WebMind", "logs")
    else:
        base = os.path.expanduser("~/.config")
        log_dir = os.path.join(base, "webmind", "logs")

    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_log_file() -> str:
    """Returns the path to the primary webmind.log file."""
    return os.path.join(get_log_dir(), "webmind.log")


_initialized_loggers = {}


def get_logger(
    name: str = "webmind",
    log_file: Optional[str] = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Returns a configured, rotating logger with sanitization filter applied.
    Max log size is bounded to max_bytes * (backup_count + 1).
    """
    if name in _initialized_loggers:
        return _initialized_loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    target_file = log_file or get_log_file()
    os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)

    sanitizer = SanitizingFilter()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Rotating File Handler
    rf_handler = RotatingFileHandler(
        target_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    rf_handler.setFormatter(formatter)
    rf_handler.addFilter(sanitizer)
    logger.addHandler(rf_handler)

    # 2. Console Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(sanitizer)
    logger.addHandler(stream_handler)

    _initialized_loggers[name] = logger
    return logger


def get_recent_logs(lines: int = 200) -> str:
    """Reads the last N lines of the active log file safely."""
    path = get_log_file()
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"[Failed to read logs: {e}]"