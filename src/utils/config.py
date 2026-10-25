"""Centralized configuration management for MediHaven.

Resolves paths dynamically across operating systems, loads environment variables
from .env, and provides typed constants for the database, ML models, cryptographic
QR tokens, and application logging.
"""

import os
import warnings
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from src.utils.research_paths import unsafe_reasons

# Base Project Root Directory (resolved relative to this file)
# file -> utils -> src -> MediHaven root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env file if present
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# "synthetic" runs the seeded demo; "mimic" is the research mode for real de-identified data
DATA_MODES = ("synthetic", "mimic")


def parse_data_mode(raw_value: Optional[str]) -> str:
    """Normalizes DATA_MODE, defaulting to the synthetic demo and rejecting unknown modes."""
    mode = (raw_value or "synthetic").strip().lower()
    if mode not in DATA_MODES:
        raise ValueError(f"Invalid DATA_MODE '{raw_value}'. Expected one of: {', '.join(DATA_MODES)}.")
    return mode


def env_path(name: str, default: Path) -> Path:
    """Reads a directory/file path from the environment; relative paths resolve against the project root."""
    raw_value = os.getenv(name, "").strip()
    path = Path(raw_value).expanduser() if raw_value else default
    return path if path.is_absolute() else BASE_DIR / path


def optional_env_path(name: str) -> Optional[Path]:
    """Like env_path, but returns None when the variable is unset or empty."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None
    path = Path(raw_value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


class Config:
    """Master configuration class for MediHaven."""

    # Project Paths
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = env_path("PROCESSED_DATA_DIR", DATA_DIR / "processed")
    EXTERNAL_DATA_DIR = DATA_DIR / "external"

    DATABASE_DIR = BASE_DIR / "database"
    DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DATABASE_DIR / "medihaven.db")))
    if not DATABASE_PATH.is_absolute():
        DATABASE_PATH = BASE_DIR / DATABASE_PATH

    MODELS_STORE_DIR = env_path("MODELS_STORE_DIR", BASE_DIR / "models_store")
    LOGS_DIR = BASE_DIR / "logs"
    TEMPLATES_DIR = BASE_DIR / "templates"
    STATIC_DIR = BASE_DIR / "static"

    # Data mode & research data locations (MIMIC-IV inputs and generated research datasets)
    DATA_MODE = parse_data_mode(os.getenv("DATA_MODE"))
    MIMIC_IV_DIR = optional_env_path("MIMIC_IV_DIR")
    MIMIC_OUTPUT_DIR = optional_env_path("MIMIC_OUTPUT_DIR")

    # Application & Server Settings
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "1").lower() in ("1", "true", "yes")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    # Demo mode lets clinicians mint sample QR passes from the scanner portal.
    # It bypasses patient consent, so it defaults to on only in development.
    DEMO_MODE = os.getenv(
        "DEMO_MODE", "1" if FLASK_ENV == "development" else "0"
    ).lower() in ("1", "true", "yes")
    FLASK_SECRET_KEY = os.getenv(
        "FLASK_SECRET_KEY", "medihaven_flask_dev_sec_9941a8b7c2e34f6d1a0b5c8e7f2d3a4b"
    )

    # Cryptographic Vault Token & QR Security
    JWT_SECRET_KEY = os.getenv(
        "JWT_SECRET_KEY", "medihaven_jwt_vault_sig_7f8e9d0c1b2a3456789abcdef0123456"
    )
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    DEFAULT_TOKEN_TTL_MINUTES = int(os.getenv("DEFAULT_TOKEN_TTL_MINUTES", "1440"))  # 24h

    # Logging Paths and Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE = Path(os.getenv("LOG_FILE", str(LOGS_DIR / "medihaven.log")))
    if not LOG_FILE.is_absolute():
        LOG_FILE = BASE_DIR / LOG_FILE

    AUDIT_LOG_FILE = Path(
        os.getenv("AUDIT_LOG_FILE", str(LOGS_DIR / "vault_access_audit.log"))
    )
    if not AUDIT_LOG_FILE.is_absolute():
        AUDIT_LOG_FILE = BASE_DIR / AUDIT_LOG_FILE

    # Machine Learning & Clinical Parameters
    KNN_NEIGHBORS = int(os.getenv("KNN_NEIGHBORS", "5"))
    KMEANS_CLUSTERS = int(os.getenv("KMEANS_CLUSTERS", "4"))
    DECISION_TREE_MAX_DEPTH = int(os.getenv("DECISION_TREE_MAX_DEPTH", "5"))
    HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.70"))
    CRITICAL_RISK_THRESHOLD = float(os.getenv("CRITICAL_RISK_THRESHOLD", "0.85"))
    EARLY_WARNING_WINDOW_DAYS_MIN = int(os.getenv("EARLY_WARNING_WINDOW_DAYS_MIN", "7"))
    EARLY_WARNING_WINDOW_DAYS_MAX = int(os.getenv("EARLY_WARNING_WINDOW_DAYS_MAX", "14"))

    # Model Artifact Paths
    SCALER_PATH = MODELS_STORE_DIR / "scaler.joblib"
    KMEANS_MODEL_PATH = MODELS_STORE_DIR / "kmeans_model.joblib"
    DECISION_TREE_MODEL_PATH = MODELS_STORE_DIR / "decision_tree_model.joblib"
    KNN_MODEL_PATH = MODELS_STORE_DIR / "knn_model.joblib"
    NEURAL_NETWORK_MODEL_PATH = MODELS_STORE_DIR / "neural_network_model.joblib"
    NEURAL_NET_MODEL_PATH = NEURAL_NETWORK_MODEL_PATH

    @classmethod
    def ensure_directories(cls):
        """Ensures that all mandatory project directories exist."""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.EXTERNAL_DATA_DIR,
            cls.DATABASE_DIR,
            cls.MODELS_STORE_DIR,
            cls.LOGS_DIR,
            cls.TEMPLATES_DIR,
            cls.STATIC_DIR / "css",
            cls.STATIC_DIR / "js",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def warn_about_unsafe_research_paths(cls):
        """Warns at startup when research paths sit in the repository or a OneDrive folder.

        The dataset builder refuses such paths outright; this is an early heads-up.
        """
        for name, path in (("MIMIC_IV_DIR", cls.MIMIC_IV_DIR), ("MIMIC_OUTPUT_DIR", cls.MIMIC_OUTPUT_DIR)):
            if path is None:
                continue
            reasons = unsafe_reasons(path)
            if reasons:
                warnings.warn(
                    f"{name}={path} is unsafe for research data because " + " and ".join(reasons)
                    + ". The MIMIC-IV dataset builder will refuse to use it.",
                    stacklevel=2,
                )


# Auto-ensure directories on import
Config.ensure_directories()
Config.warn_about_unsafe_research_paths()
