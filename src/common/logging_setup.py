"""Logging setup and JSON formatter."""

from __future__ import annotations
import logging
import os
import sys
import json
from typing import Optional, Mapping, Any


class JsonFormatter(logging.Formatter):
    """Lightweight JSON formatter for structured logs."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_mode: bool = False) -> None:
    """Set up root logger with console handler and optional JSON formatting."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    for h in list(root.handlers):
        root.removeHandler(h)
    
    handler = logging.StreamHandler(stream=sys.stdout)
    if json_mode:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    
    root.addHandler(handler)
    
    # Set third-party loggers to WARNING level
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)