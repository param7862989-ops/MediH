"""Small synthetic Task 1 datasets with the exact v1.2.0 schema, for Phase 2 unit tests only.

Values are random numbers with an injected signal; they are not clinical data and are never used
for the real-data evaluation. Some subjects have two stays so subject grouping is exercised.
"""

import json
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.datasources.mimic.mappings import FEATURE_SPEC_VERSION, feature_columns
from src.tasks.build_mimic_t1 import DATASET_ID, ID_COLUMNS, LABEL_COLUMN, content_sha256
from src.tasks.mortality_icu_24h import ANOMALY_COLUMN, TASK_ID
from src.training.dataset import EXPECTED_DATASET_VERSION, default_dataset_dir

SUBJECT_BASE = 77000000


def make_task1_frame(n_subjects: int = 60, multi_stay_subjects: int = 10, positives: int = 14,
                     anomalies: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = [SUBJECT_BASE + i for i in range(n_subjects) for _ in range(2 if i < multi_stay_subjects else 1)]
    n = len(subjects)
    label = np.zeros(n, dtype="int8")
    label[rng.choice(n, size=positives, replace=False)] = 1
    anomaly = np.zeros(n, dtype="int8")
    anomaly[np.flatnonzero(label == 1)[:anomalies]] = 1

    frame = pd.DataFrame({
        "src_subject_id": np.array(subjects, dtype="int64"),
        "src_hadm_id": np.arange(n, dtype="int64") + 78000000,
        "src_stay_id": np.arange(n, dtype="int64") + 79000000,
        "index_time": pd.Timestamp("2150-01-02 08:00:00") + pd.to_timedelta(np.arange(n), unit="h"),
        LABEL_COLUMN: label,
        ANOMALY_COLUMN: anomaly,
    })
    for col in feature_columns():
        if col.endswith("_count"):
            frame[col] = rng.integers(0, 20, size=n).astype("int64")
        elif col.endswith("_missing"):
            frame[col] = np.zeros(n, dtype="int8")
        else:
            frame[col] = rng.normal(0, 1, size=n)
    frame["heart_rate_mean"] += 1.5 * label   # learnable signal
    frame["lactate_last"] += 1.0 * label
    lactate_missing = rng.random(n) < 0.3     # realistic missingness
    for agg in ("min", "max", "mean", "last"):
        frame.loc[lactate_missing, f"lactate_{agg}"] = np.nan
    frame.loc[lactate_missing, "lactate_count"] = 0
    frame.loc[lactate_missing, "lactate_missing"] = 1
    frame["dataset_version"] = EXPECTED_DATASET_VERSION
    frame["feature_spec_version"] = FEATURE_SPEC_VERSION
    return frame


def make_manifest(frame: pd.DataFrame) -> Dict[str, object]:
    return {
        "dataset_id": DATASET_ID,
        "dataset_version": EXPECTED_DATASET_VERSION,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "task": {"task_id": TASK_ID, "task_version": "3"},
        "source": {"release": "synthetic-unit-test-fixture"},
        "columns": {
            "identifiers": list(ID_COLUMNS),
            "index_time": "index_time",
            "label": LABEL_COLUMN,
            "label_quality": {"columns": [ANOMALY_COLUMN], "note": "fixture"},
            "features": feature_columns(),
            "versions": ["dataset_version", "feature_spec_version"],
        },
        "row_count": len(frame),
        "content_sha256": content_sha256(frame),
    }


def write_task1_dataset(
    research_dir: Path,
    frame: Optional[pd.DataFrame] = None,
    manifest_overrides: Optional[Dict[str, object]] = None,
    mutate_after_hash: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    **frame_kwargs,
) -> Path:
    """Writes dataset.parquet + manifest.json under <research_dir>/<dataset folder>; returns that folder."""
    frame = make_task1_frame(**frame_kwargs) if frame is None else frame
    manifest = make_manifest(frame)
    manifest.update(manifest_overrides or {})
    if mutate_after_hash is not None:
        frame = mutate_after_hash(frame)
    dataset_dir = default_dataset_dir(research_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dataset_dir / "dataset.parquet", index=False)
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return dataset_dir
