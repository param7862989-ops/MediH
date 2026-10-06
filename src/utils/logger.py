"""Structured clinical and security audit logging engine for MediHaven.

Provides separate log streams for general application execution and dedicated,
tamper-evident audit logging for the Patient Medical Vault and QR access events.
"""

import logging
from logging.handlers import RotatingFileHandler
from src.utils.config import Config


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
    Config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=str(Config.LOG_FILE),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
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

    Config.AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    audit_file_handler = RotatingFileHandler(
        filename=str(Config.AUDIT_LOG_FILE),
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    audit_file_handler.setLevel(logging.INFO)
    audit_file_handler.setFormatter(audit_formatter)
    audit_logger.addHandler(audit_file_handler)

    return audit_logger
