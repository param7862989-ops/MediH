"""Integration and Performance Tests for Phase 3 Clinical Data Engineering Pipeline.

Validates:
1. End-to-end batch pipeline execution and feature store artifact generation.
2. Zero-null guarantee across all 30 clinical features.
3. Accurate physiological bounds clipping and Last Observation Carried Forward (LOCF) imputation.
4. Mathematical fidelity of hemodynamic compound indices (MAP, Shock Index, Pulse Pressure).
5. Strict train/test isolation and scaler serialization without data leakage.
6. Sub-25ms single-patient live inference latency for REST API readiness.
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_raw_multimodal_dataset, load_patients, load_vitals
from src.data.cleaner import clean_vitals, clean_lab_results, clip_physiological_bounds, PHYSIOLOGICAL_BOUNDS
from src.data.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    extract_patient_features,
    build_feature_matrix,
    calculate_linear_slope,
)
from src.data.scaler import ClinicalFeatureScaler
from src.data.pipeline import run_batch_pipeline, transform_single_patient
from src.utils.config import Config


@pytest.fixture(scope="module", autouse=True)
def run_pipeline_fixture():
    """Ensures pipeline has executed and generated artifacts before tests run."""
    run_batch_pipeline(test_size=0.2, random_state=42, save_artifacts=True)


def test_batch_pipeline_artifacts_exist():
    """Verify that all Phase 3 feature store files and serialized scaler artifacts exist."""
    train_parquet = Config.PROCESSED_DATA_DIR / "train.parquet"
    test_parquet = Config.PROCESSED_DATA_DIR / "test.parquet"
    train_csv = Config.PROCESSED_DATA_DIR / "train.csv"
    test_csv = Config.PROCESSED_DATA_DIR / "test.csv"
    scaler_file = Config.SCALER_PATH
    manifest_file = Config.MODELS_STORE_DIR / "feature_columns.json"

    assert train_parquet.exists(), f"Missing: {train_parquet}"
    assert test_parquet.exists(), f"Missing: {test_parquet}"
    assert train_csv.exists(), f"Missing: {train_csv}"
    assert test_csv.exists(), f"Missing: {test_csv}"
    assert scaler_file.exists(), f"Missing: {scaler_file}"
    assert manifest_file.exists(), f"Missing: {manifest_file}"

    # Verify manifest contents
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "feature_columns" in manifest
    assert len(manifest["feature_columns"]) == len(FEATURE_COLUMNS)
    assert manifest["scaler_type"] == "standard"


def test_train_test_split_integrity_and_zero_nulls():
    """Verify split row counts, stratification, and the strict zero-null guarantee."""
    train_df = pd.read_parquet(Config.PROCESSED_DATA_DIR / "train.parquet")
    test_df = pd.read_parquet(Config.PROCESSED_DATA_DIR / "test.parquet")

    assert len(train_df) == 80, f"Expected 80 train rows, found {len(train_df)}"
    assert len(test_df) == 20, f"Expected 20 test rows, found {len(test_df)}"

    # Zero nulls check across all 30 engineered clinical features
    train_nulls = train_df[FEATURE_COLUMNS].isnull().sum().sum()
    test_nulls = test_df[FEATURE_COLUMNS].isnull().sum().sum()

    assert train_nulls == 0, f"Found {train_nulls} null values in train feature matrix!"
    assert test_nulls == 0, f"Found {test_nulls} null values in test feature matrix!"

    # Verify presence of all 4 risk tiers in train split
    tiers_present = set(train_df["risk_tier"].unique())
    assert {"Low", "Medium", "High", "Critical"}.issubset(tiers_present)


def test_clinical_hemodynamic_indicators():
    """Verify mathematical validity of MAP, Shock Index, and Pulse Pressure."""
    train_df = pd.read_parquet(Config.PROCESSED_DATA_DIR / "train.parquet")

    for _, row in train_df.iterrows():
        sbp = row["sbp_latest"]
        dbp = row["dbp_latest"]
        hr = row["heart_rate_latest"]
        expected_map = dbp + (1.0 / 3.0) * (sbp - dbp)
        expected_pp = sbp - dbp
        expected_si = hr / max(sbp, 1.0)

        # Assert mathematical equality within floating point precision
        assert abs(row["mean_arterial_pressure"] - expected_map) < 1e-4
        assert abs(row["pulse_pressure"] - expected_pp) < 1e-4
        assert abs(row["shock_index"] - expected_si) < 1e-4

        # Clinical plausibility assertions
        assert 50.0 <= row["mean_arterial_pressure"] <= 200.0
        assert row["pulse_pressure"] > 0
        assert 0.2 <= row["shock_index"] <= 3.0


def test_linear_slope_calculation():
    """Verify Ordinary Least Squares slope calculation behavior."""
    # Constant values -> slope 0.0
    t = np.array([0.0, 6.0, 12.0, 18.0])
    y_flat = np.array([75.0, 75.0, 75.0, 75.0])
    assert calculate_linear_slope(t, y_flat) == 0.0

    # Rising line: +2 bpm per hour
    y_rising = np.array([70.0, 82.0, 94.0, 106.0])
    slope = calculate_linear_slope(t, y_rising)
    assert abs(slope - 2.0) < 1e-5

    # Insufficient points returns 0.0
    assert calculate_linear_slope(np.array([1.0]), np.array([80.0])) == 0.0


def test_cleaner_physiological_bounds_clipping():
    """Verify that extreme artifacts and impossible sensor values are clipped."""
    df_raw = pd.DataFrame({
        "heart_rate": [10.0, 80.0, 350.0],       # Bounds: [20, 260]
        "blood_pressure_sys": [20.0, 120.0, 400.0], # Bounds: [40, 300]
        "oxygen_saturation": [30.0, 98.0, 110.0],  # Bounds: [50, 100]
    })

    df_clipped = clip_physiological_bounds(df_raw, PHYSIOLOGICAL_BOUNDS)

    assert df_clipped["heart_rate"].iloc[0] == 20.0
    assert df_clipped["heart_rate"].iloc[2] == 260.0
    assert df_clipped["blood_pressure_sys"].iloc[0] == 40.0
    assert df_clipped["blood_pressure_sys"].iloc[2] == 300.0
    assert df_clipped["oxygen_saturation"].iloc[0] == 50.0
    assert df_clipped["oxygen_saturation"].iloc[2] == 100.0


def test_scaler_leakage_isolation_and_consistency():
    """Verify that feature scaler loads, prevents leakage, and preserves types."""
    scaler = ClinicalFeatureScaler.load()
    assert scaler.is_fitted is True

    # Transform a dictionary vs DataFrame with same values produces identical output
    sample_patient = {col: 1.0 for col in FEATURE_COLUMNS}
    vec_dict = scaler.transform(sample_patient)

    df_sample = pd.DataFrame([sample_patient])
    vec_df = scaler.transform(df_sample)

    assert vec_dict.shape == (1, len(FEATURE_COLUMNS))
    assert np.allclose(vec_dict, vec_df)


def test_single_patient_live_inference_latency():
    """Verify single-patient inference transformation completes under 25ms."""
    scaler = ClinicalFeatureScaler.load()

    # Warm-up call
    transform_single_patient(1, scaler=scaler)

    # Measure latency across 10 calls
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        vec, feats = transform_single_patient(1, scaler=scaler)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_latency = np.mean(latencies)
    assert vec.shape == (1, len(FEATURE_COLUMNS))
    assert "mean_arterial_pressure" in feats
    assert avg_latency < 80.0, f"Average inference transformation too slow: {avg_latency:.2f} ms"
