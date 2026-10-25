"""First-start preparation of the synthetic demo assets.

A fresh checkout or container has no database, feature store or trained models. This creates each
missing piece once (seed -> feature pipeline -> ensemble training) and leaves existing ones untouched,
so restarts are fast and never reseed a database that already exists.

Used by run_demo.py and wsgi.py; can also run on its own as a deploy step:
    python -m src.bootstrap
"""

import sqlite3
from contextlib import closing
from typing import Callable

from src.utils.config import Config
from src.utils.data_mode import require_synthetic_mode

SYNTHETIC_PATIENTS = 100


def database_seeded() -> bool:
    """True if the database has patients. An empty file (created by an early request) or a seed that
    crashed (seeding commits once, at the end) counts as not seeded, so it is seeded again."""
    if not Config.DATABASE_PATH.exists():
        return False
    try:
        uri = Config.DATABASE_PATH.resolve().as_uri() + '?mode=ro'
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            return conn.execute('SELECT COUNT(*) FROM patients').fetchone()[0] > 0
    except sqlite3.Error:
        return False


def model_artifacts_ready() -> bool:
    return all(p.exists() for p in (Config.KMEANS_MODEL_PATH, Config.DECISION_TREE_MODEL_PATH,
                                    Config.KNN_MODEL_PATH, Config.NEURAL_NETWORK_MODEL_PATH))


def prepare_synthetic_assets(report: Callable[[str], None] = print, total_steps: int = 3) -> None:
    """Creates whichever synthetic demo assets are missing. Refuses to run in research mode."""
    # Seeding and training synthetic assets is exactly what research mode must never do
    require_synthetic_mode("Preparing the synthetic demo assets (auto-seed and auto-train)")

    report(f" [1/{total_steps}] Checking Relational Database & Seed Data...")
    if not database_seeded():
        report(f"       -> Database missing or empty at {Config.DATABASE_PATH}. Auto-initializing...")
        from src.database.seed_data import seed_clinical_database
        seed_clinical_database(num_patients=SYNTHETIC_PATIENTS)
    else:
        report(f"       [OK] SQLite database verified ({Config.DATABASE_PATH.name})")

    report(f" [2/{total_steps}] Checking Clinical Feature Engineering Artifacts...")
    if not (Config.PROCESSED_DATA_DIR / "train.parquet").exists() or not Config.SCALER_PATH.exists():
        report("       -> Processed feature store missing. Running data engineering pipeline...")
        from src.data.pipeline import run_batch_pipeline
        run_batch_pipeline()
    else:
        report("       [OK] Processed clinical feature store verified (train & test parquets)")

    report(f" [3/{total_steps}] Checking Machine Learning Model Artifacts...")
    if not model_artifacts_ready():
        report("       -> Model weights missing. Fitting and serializing 4-model ensemble...")
        from src.models.ensemble import EnsembleClinicalPredictor
        EnsembleClinicalPredictor.train_from_data()
    else:
        report("       [OK] 4-Model Ensemble verified (K-Means, Decision Tree, KNN, MLP)")


if __name__ == "__main__":
    prepare_synthetic_assets()
