"""Loads and validates the Phase 1 Task 1 research dataset before any training.

Training is pinned to dataset version 1.2.0 / task version 3. Any mismatch in versions, schema,
labels, identifiers or content hash stops the run instead of training on unexpected data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json

import pandas as pd

from src.datasources.mimic.mappings import FEATURE_SPEC_VERSION, feature_columns
from src.tasks.build_mimic_t1 import DATASET_ID, ID_COLUMNS, LABEL_COLUMN, content_sha256
from src.tasks.mortality_icu_24h import ANOMALY_COLUMN, TASK_ID
from src.utils.research_paths import require_safe_research_path

EXPECTED_DATASET_VERSION = "1.2.0"
EXPECTED_TASK_VERSION = "3"
SUBJECT_COLUMN = "src_subject_id"
INDEX_TIME_COLUMN = "index_time"
VERSION_COLUMNS = ["dataset_version", "feature_spec_version"]


class DatasetValidationError(ValueError):
    """Raised when the Task 1 dataset is missing or does not match the expected contract."""


@dataclass
class Task1Dataset:
    frame: pd.DataFrame
    manifest: Dict[str, object]
    dataset_dir: Path
    features: List[str]

    @property
    def labels(self) -> pd.Series:
        return self.frame[LABEL_COLUMN]


def default_dataset_dir(research_dir: Path) -> Path:
    return Path(research_dir) / f"{DATASET_ID}_v{EXPECTED_DATASET_VERSION}"


def _fail(message: str) -> None:
    raise DatasetValidationError(message)


def load_task1_dataset(dataset_dir: Path) -> Task1Dataset:
    """Reads dataset.parquet + manifest.json and validates them against the expected contract."""
    dataset_dir = require_safe_research_path(dataset_dir, "the Task 1 dataset directory")
    data_file, manifest_file = dataset_dir / "dataset.parquet", dataset_dir / "manifest.json"
    if not data_file.is_file() or not manifest_file.is_file():
        _fail(f"Task 1 dataset not found in '{dataset_dir}'. Build it first with: "
              f"python -m src.tasks.build_mimic_t1 --mimic-dir <MIMIC-IV demo> --output-dir <research output>")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    frame = pd.read_parquet(data_file)
    _validate_frame(frame, manifest)
    return Task1Dataset(frame=frame, manifest=manifest, dataset_dir=dataset_dir,
                        features=list(manifest["columns"]["features"]))


def _validate_manifest(manifest: Dict[str, object]) -> None:
    expected = {
        "dataset_id": DATASET_ID,
        "dataset_version": EXPECTED_DATASET_VERSION,
        "feature_spec_version": FEATURE_SPEC_VERSION,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            _fail(f"Manifest {key} is '{manifest.get(key)}', expected '{value}'.")
    task = manifest.get("task") or {}
    if task.get("task_id") != TASK_ID or task.get("task_version") != EXPECTED_TASK_VERSION:
        _fail(f"Manifest task is '{task.get('task_id')}' v{task.get('task_version')}, "
              f"expected '{TASK_ID}' v{EXPECTED_TASK_VERSION}.")

    columns = manifest.get("columns") or {}
    if columns.get("identifiers") != ID_COLUMNS or columns.get("label") != LABEL_COLUMN:
        _fail("Manifest identifier/label columns do not match the Task 1 contract.")
    if columns.get("index_time") != INDEX_TIME_COLUMN or columns.get("versions") != VERSION_COLUMNS:
        _fail("Manifest index-time/version columns do not match the Task 1 contract.")
    if (columns.get("label_quality") or {}).get("columns") != [ANOMALY_COLUMN]:
        _fail(f"Manifest label-quality columns must be exactly ['{ANOMALY_COLUMN}'].")
    if columns.get("features") != feature_columns():
        _fail("Manifest feature list differs from feature spec "
              f"{FEATURE_SPEC_VERSION} in src/datasources/mimic/mappings.py.")


def _validate_frame(frame: pd.DataFrame, manifest: Dict[str, object]) -> None:
    columns = manifest["columns"]
    expected = (ID_COLUMNS + [INDEX_TIME_COLUMN, LABEL_COLUMN, ANOMALY_COLUMN]
                + list(columns["features"]) + VERSION_COLUMNS)
    missing, unexpected = sorted(set(expected) - set(frame.columns)), sorted(set(frame.columns) - set(expected))
    if missing or unexpected:
        _fail(f"Dataset columns differ from the manifest (missing={missing}, unexpected={unexpected}).")
    if len(frame) != manifest.get("row_count"):
        _fail(f"Dataset has {len(frame)} rows but the manifest records {manifest.get('row_count')}.")
    if content_sha256(frame) != manifest.get("content_sha256"):
        _fail("Dataset content hash does not match the manifest; the file may be modified or corrupt.")

    for column in (LABEL_COLUMN, ANOMALY_COLUMN):
        values = frame[column]
        if values.isna().any() or not set(values.unique()) <= {0, 1}:
            _fail(f"Column '{column}' must contain only 0/1 values with no missing entries.")
    if (frame.loc[frame[ANOMALY_COLUMN] == 1, LABEL_COLUMN] != 1).any():
        _fail("Outcome-anomaly rows must all be positive under task version 3.")
    if frame[ID_COLUMNS].isna().any().any():
        _fail("Identifier columns contain missing values.")
    if not frame["src_stay_id"].is_unique:
        _fail("Dataset must contain one row per ICU stay.")
    if set(frame["dataset_version"]) != {EXPECTED_DATASET_VERSION} or set(frame["feature_spec_version"]) != {FEATURE_SPEC_VERSION}:
        _fail("Row-level version stamps do not match the expected dataset/feature-spec versions.")
    non_numeric = [c for c in columns["features"] if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        _fail(f"Feature columns must be numeric: {non_numeric}")
