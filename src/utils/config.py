"""Centralized configuration management for MediHaven.

Resolves paths dynamically across operating systems, loads environment variables
from .env, and provides typed constants for the database, ML models, cryptographic
QR tokens, and application logging.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Project Root Directory (resolved relative to this file)
# file -> utils -> src -> MediHaven root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env file if present
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Config:
    """Master configuration class for MediHaven."""

    # Project Paths
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"

    DATABASE_DIR = BASE_DIR / "database"
    DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DATABASE_DIR / "medihaven.db")))
    if not DATABASE_PATH.is_absolute():
        DATABASE_PATH = BASE_DIR / DATABASE_PATH

    MODELS_STORE_DIR = BASE_DIR / "models_store"
    LOGS_DIR = BASE_DIR / "logs"
    TEMPLATES_DIR = BASE_DIR / "templates"
    STATIC_DIR = BASE_DIR / "static"

    # Application & Server Settings
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "1").lower() in ("1", "true", "yes")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
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


# Auto-ensure directories on import
Config.ensure_directories()
