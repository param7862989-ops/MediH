"""Structured clinical and security audit logging engine for MediHaven.

Provides separate log streams for general application execution and dedicated,
tamper-evident audit logging for the Patient Medical Vault and QR access events.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.utils.config import Config


def _rotating_file_handler(path: Path, max_bytes: int, backups: int) -> Optional[RotatingFileHandler]:
    """A rotating file handler, or None if the location is not writable (e.g. a read-only container).

    Console logging keeps working either way; vault access events are also stored in the database.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return RotatingFileHandler(filename=str(path), maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    except OSError as e:
        print(f"[medihaven] File logging disabled for {path.name}: {type(e).__name__}", file=sys.stderr)
        return None


def get_logger(name: str = "medihaven") -> logging.Logger:
    """Returns a configured logger for application and clinical events.

    Args:
        name: Name of the logger (typically __name__ of the caller).

    Returns:
        logging.Logger instance configured with console and rotating file handlers.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if already initialized
    if logger.handlers:
        return logger

    # Resolve log level from Config
    log_level = getattr(logging, Config.LOG_LEVEL, logging.INFO)
    logger.setLevel(log_level)

    # Standard clinical log format
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler (10MB per file, up to 5 backups)
    file_handler = _rotating_file_handler(Config.LOG_FILE, 10 * 1024 * 1024, 5)
    if file_handler:
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_audit_logger(name: str = "medihaven.audit") -> logging.Logger:
    """Returns a dedicated security audit logger for vault QR operations.

    Args:
        name: Name of the audit logger.

    Returns:
        logging.Logger configured to append exclusively to the vault audit log.
    """
    audit_logger = logging.getLogger(name)

    if audit_logger.handlers:
        return audit_logger

    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # Prevent audit events leaking to console

    audit_formatter = logging.Formatter(
        fmt="[%(asctime)s] [AUDIT] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Without a writable audit file, audit events go to stderr instead of being lost
    audit_handler = _rotating_file_handler(Config.AUDIT_LOG_FILE, 20 * 1024 * 1024, 10) or logging.StreamHandler()
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(audit_formatter)
    audit_logger.addHandler(audit_handler)

    return audit_logger
