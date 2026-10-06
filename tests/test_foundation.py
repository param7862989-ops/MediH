"""Foundation Smoke Tests for Phase 1.

Validates that repository directories, environment configuration, structured logging,
and baseline UI assets exist and operate without errors.
"""

from pathlib import Path
from src.utils.config import Config
from src.utils.logger import get_logger, get_audit_logger


def test_directory_structure_exists():
    """Verify that all architectural directories required by Section 10 are present."""
    required_directories = [
        Config.DATA_DIR / "raw",
        Config.DATA_DIR / "processed",
        Config.DATA_DIR / "external",
        Config.DATABASE_DIR,
        Config.MODELS_STORE_DIR,
        Config.LOGS_DIR,
        Config.TEMPLATES_DIR,
        Config.STATIC_DIR / "css",
        Config.STATIC_DIR / "js",
        Config.BASE_DIR / "src" / "utils",
        Config.BASE_DIR / "src" / "database",
        Config.BASE_DIR / "src" / "data",
        Config.BASE_DIR / "src" / "models",
        Config.BASE_DIR / "src" / "vault",
        Config.BASE_DIR / "src" / "api",
        Config.BASE_DIR / "src" / "evaluation",
        Config.BASE_DIR / "tests",
    ]

    for directory in required_directories:
        assert directory.exists(), f"Mandatory directory missing: {directory}"
        assert directory.is_dir(), f"Path is not a directory: {directory}"


def test_config_constants_and_types():
    """Verify that Config loads valid parameters and types."""
    assert Config.BASE_DIR.exists()
    assert isinstance(Config.DATABASE_PATH, Path)
    assert len(Config.JWT_SECRET_KEY) >= 16, "JWT Secret Key must be at least 16 chars"
    assert Config.JWT_ALGORITHM == "HS256"
    assert Config.DEFAULT_TOKEN_TTL_MINUTES > 0
    assert Config.KNN_NEIGHBORS == 5
    assert Config.KMEANS_CLUSTERS == 4
    assert 0.0 < Config.HIGH_RISK_THRESHOLD < 1.0
    assert 0.0 < Config.CRITICAL_RISK_THRESHOLD <= 1.0
    assert Config.HIGH_RISK_THRESHOLD < Config.CRITICAL_RISK_THRESHOLD


def test_application_and_audit_loggers():
    """Verify that clinical application logger and security audit logger write to disk."""
    app_logger = get_logger("tests.app")
    test_message = "Foundation test log entry"
    app_logger.info(test_message)

    assert Config.LOG_FILE.exists(), f"Log file not created: {Config.LOG_FILE}"
    with open(Config.LOG_FILE, "r", encoding="utf-8") as f:
        log_content = f.read()
    assert test_message in log_content

    audit_logger = get_audit_logger("tests.audit")
    test_audit_event = "QR_TOKEN_VERIFY_TEST_ACTION"
    audit_logger.info(test_audit_event)

    assert Config.AUDIT_LOG_FILE.exists(), f"Audit file not created: {Config.AUDIT_LOG_FILE}"
    with open(Config.AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        audit_content = f.read()
    assert test_audit_event in audit_content


def test_static_and_template_assets_exist():
    """Verify that design tokens and semantic layout templates exist."""
    tokens_file = Config.STATIC_DIR / "css" / "tokens.css"
    assert tokens_file.exists(), "static/css/tokens.css missing"
    with open(tokens_file, "r", encoding="utf-8") as f:
        tokens_content = f.read()
    assert "--bg-primary:" in tokens_content
    assert "--risk-critical:" in tokens_content
    assert "--color-brand:" in tokens_content

    base_template = Config.TEMPLATES_DIR / "base.html"
    assert base_template.exists(), "templates/base.html missing"
    with open(base_template, "r", encoding="utf-8") as f:
        template_content = f.read()
    assert "MediHaven" in template_content
    assert "{% block content %}" in template_content
