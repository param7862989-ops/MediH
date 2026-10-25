"""Shared PyTest configuration: every test runs against throwaway paths.

Before anything imports src.utils.config, this file points the database, model store,
processed-data directory and logs at a fresh temporary folder and switches off external
services. The session then seeds a temporary synthetic database and copies (read-only) the
existing model artifacts into the temporary model store, so no test can overwrite or reseed
the persistent demo database or model artifacts. At the end of the session the persistent
files are re-hashed and the run fails if anything changed.
"""

import logging
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "src.utils.config" in sys.modules:
    raise RuntimeError("src.utils.config was imported before test isolation was configured.")

from tests.isolation_utils import describe_changes, snapshot_persistent_artifacts

TEST_ROOT = Path(tempfile.mkdtemp(prefix="medihaven_pytest_"))
os.environ.update({
    "DATA_MODE": "synthetic",
    "DATABASE_PATH": str(TEST_ROOT / "database" / "medihaven_test.db"),
    "MODELS_STORE_DIR": str(TEST_ROOT / "models_store"),
    "PROCESSED_DATA_DIR": str(TEST_ROOT / "processed"),
    "LOG_FILE": str(TEST_ROOT / "logs" / "medihaven_test.log"),
    "AUDIT_LOG_FILE": str(TEST_ROOT / "logs" / "vault_access_audit_test.log"),
    # Empty values (not unset) so python-dotenv cannot fill them in from .env
    "MIMIC_IV_DIR": "",
    "MIMIC_OUTPUT_DIR": "",
    "GEMINI_API_KEY": "",
    "NO_BROWSER": "1",
    # Tests never read the real .env secrets: pin the documented public development values, so results do
    # not depend on (or expose) local secrets. Tests that need strong secrets set them explicitly.
    "FLASK_SECRET_KEY": "medihaven_flask_dev_sec_9941a8b7c2e34f6d1a0b5c8e7f2d3a4b",
    "JWT_SECRET_KEY": "medihaven_jwt_vault_sig_7f8e9d0c1b2a3456789abcdef0123456",
    # Ambient launcher settings must not leak into tests either
    "FLASK_RUN_HOST": "",
    "FLASK_RUN_FROM_CLI": "",
})

PERSISTENT_SNAPSHOT_BEFORE = snapshot_persistent_artifacts()

import pytest

from src.utils.config import Config

for _name, _path in (
    ("DATABASE_PATH", Config.DATABASE_PATH),
    ("MODELS_STORE_DIR", Config.MODELS_STORE_DIR),
    ("PROCESSED_DATA_DIR", Config.PROCESSED_DATA_DIR),
    ("LOG_FILE", Config.LOG_FILE),
):
    if TEST_ROOT not in Path(_path).parents:
        raise RuntimeError(f"Test isolation failed: Config.{_name}={_path} is outside {TEST_ROOT}")

# Existing synthetic model artifacts copied (never written) into the temporary model store
MODEL_ARTIFACTS = (
    "kmeans_model.joblib",
    "decision_tree_model.joblib",
    "knn_model.joblib",
    "neural_network_model.joblib",
    "ensemble_metadata.json",
)


def pytest_sessionstart(session):
    """Seeds the temporary synthetic database and prepares temporary artifacts."""
    from database.seed_data import seed_database
    from src.data.pipeline import run_batch_pipeline

    # seed_data seeds the RNG at import; re-seed so the temporary cohort matches the canonical demo
    random.seed(42)
    seed_database(100)
    run_batch_pipeline(test_size=0.2, random_state=42, save_artifacts=True)

    persistent_store = PROJECT_ROOT / "models_store"
    if all((persistent_store / name).is_file() for name in MODEL_ARTIFACTS):
        for name in MODEL_ARTIFACTS:
            shutil.copy2(persistent_store / name, Config.MODELS_STORE_DIR / name)
    else:
        # Fresh clone without artifacts: fit throwaway models inside the temporary store only
        from src.models.ensemble import EnsembleClinicalPredictor
        EnsembleClinicalPredictor.train_from_data()


def pytest_sessionfinish(session, exitstatus):
    """Fails the run if the persistent database or artifacts changed, then removes TEST_ROOT."""
    changes = describe_changes(PERSISTENT_SNAPSHOT_BEFORE, snapshot_persistent_artifacts())
    if changes:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        message = "PERSISTENT DATA CHANGED DURING TESTS:\n  " + "\n  ".join(changes)
        if reporter:
            reporter.write_line(message, red=True, bold=True)
        else:
            print(message, file=sys.stderr)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED

    logging.shutdown()
    shutil.rmtree(TEST_ROOT, ignore_errors=True)


@pytest.fixture(scope="session")
def project_config():
    """Provides access to the master configuration object."""
    return Config


@pytest.fixture(scope="session")
def test_root() -> Path:
    """The temporary folder holding this session's database, models, data and logs."""
    return TEST_ROOT


@pytest.fixture(scope="session")
def persistent_snapshot_before():
    """Checksums of the persistent database/artifacts taken before any test code ran."""
    return PERSISTENT_SNAPSHOT_BEFORE


@pytest.fixture(autouse=True)
def fresh_rate_limits():
    """Rate-limit counters are per process; each test starts with none, as a new client would."""
    from src.api.rate_limit import reset_rate_limits
    reset_rate_limits()
    yield
    reset_rate_limits()
