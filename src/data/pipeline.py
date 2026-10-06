"""Master Clinical Data Engineering and Feature Pipeline for MediHaven.

Provides end-to-end execution of data extraction, physiological sanitization,
feature engineering, leak-free normalization, and artifact export:
- Batch Mode: Generates train/test feature stores (train.parquet, test.parquet)
- Inference Mode: Generates scaled 1D feature vectors for live REST API scoring
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_raw_multimodal_dataset, load_patients, load_vitals, load_lab_results, load_medications
from src.data.cleaner import clean_multimodal_data, clean_vitals, clean_lab_results
from src.data.feature_engineering import (
    build_feature_matrix,
    extract_patient_features,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
)
from src.data.scaler import ClinicalFeatureScaler
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("medihaven.data.pipeline")


def run_batch_pipeline(
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, ClinicalFeatureScaler]:
    """Executes the full clinical data pipeline across all patient cohorts.

    Args:
        test_size: Proportion of patients allocated to test split (default: 0.2).
        random_state: Deterministic random seed for reproducibility.
        save_artifacts: If True, writes parquet/csv and scaler artifacts to disk.

    Returns:
        Tuple of (train_df, test_df, fitted_scaler).
    """
    logger.info("=" * 60)
    logger.info("STARTING BATCH CLINICAL DATA ENGINEERING PIPELINE (Phase 3)")
    logger.info("=" * 60)

    # 1. Extraction Layer
    raw_data = load_raw_multimodal_dataset()
    if raw_data["patients"].empty:
        raise RuntimeError("No patients found in database. Please run seed_data.py first.")

    # 2. Cleaning & Imputation Layer
    clean_data = clean_multimodal_data(raw_data)

    # 3. Clinical Feature Engineering Layer
    feature_df = build_feature_matrix(clean_data)

    # 4. Stratified Train / Test Split
    stratify_col = (
        feature_df["risk_tier_encoded"]
        if "risk_tier_encoded" in feature_df.columns
        else None
    )

    train_df, test_df = train_test_split(
        feature_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )
    train_df = train_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    logger.info(
        f"Dataset split: Train={len(train_df)} patients, "
        f"Test={len(test_df)} patients (Stratified by risk_tier)."
    )

    # 5. Fit Feature Scaler (Strictly on Train set to avoid data leakage)
    scaler = ClinicalFeatureScaler(scaler_type="standard", feature_columns=FEATURE_COLUMNS)
    scaler.fit(train_df)

    # 6. Apply Scaling and Export Artifacts
    if save_artifacts:
        Config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.MODELS_STORE_DIR.mkdir(parents=True, exist_ok=True)

        # Export scaled copies alongside raw unscaled for inspection
        train_parquet_path = Config.PROCESSED_DATA_DIR / "train.parquet"
        test_parquet_path = Config.PROCESSED_DATA_DIR / "test.parquet"
        train_csv_path = Config.PROCESSED_DATA_DIR / "train.csv"
        test_csv_path = Config.PROCESSED_DATA_DIR / "test.csv"

        train_df.to_parquet(train_parquet_path, index=False)
        test_df.to_parquet(test_parquet_path, index=False)
        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)

        scaler.save()

        logger.info(f"Artifacts successfully saved:")
        logger.info(f" - Train Parquet: {train_parquet_path}")
        logger.info(f" - Test Parquet:  {test_parquet_path}")
        logger.info(f" - Train CSV:      {train_csv_path}")
        logger.info(f" - Test CSV:       {test_csv_path}")
        logger.info(f" - Scaler Artifact: {Config.SCALER_PATH}")

    logger.info("BATCH CLINICAL DATA PIPELINE COMPLETED SUCCESSFULLY.")
    return train_df, test_df, scaler


def transform_single_patient(
    patient_id: int,
    scaler: Optional[ClinicalFeatureScaler] = None,
    conn: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Transforms a single patient's longitudinal records into a model-ready vector.

    Args:
        patient_id: Patient database ID.
        scaler: Optional pre-loaded ClinicalFeatureScaler instance.
        conn: Optional active database connection.

    Returns:
        Tuple of (scaled_feature_array (1, 30), raw_feature_dictionary).
    """
    if scaler is None:
        scaler = ClinicalFeatureScaler.load()

    from src.database.db import get_connection

    should_close = False
    if conn is None:
        conn_context = get_connection()
        active_conn = conn_context.__enter__()
        should_close = True
    else:
        active_conn = conn

    try:
        # Ingest records for this patient using shared connection
        p_df = load_patients(patient_id, conn=active_conn)
        if p_df.empty:
            raise ValueError(f"Patient with ID {patient_id} not found in database.")

        p_row = p_df.iloc[0]
        raw_vitals = load_vitals(patient_id, conn=active_conn)
        raw_labs = load_lab_results(patient_id, conn=active_conn)
        raw_meds = load_medications(patient_id, conn=active_conn)
    finally:
        if should_close:
            conn_context.__exit__(None, None, None)

    # Clean records
    clean_v = clean_vitals(raw_vitals)
    clean_l = clean_lab_results(raw_labs)

    # Extract 28 features
    feats_dict = extract_patient_features(
        patient_id=patient_id,
        vitals_df=clean_v,
        labs_df=clean_l,
        meds_df=raw_meds,
        patient_row=p_row,
    )

    # Scale vector
    scaled_vector = scaler.transform(feats_dict)

    return scaled_vector, feats_dict


if __name__ == "__main__":
    try:
        train_split, test_split, clinical_scaler = run_batch_pipeline()
        print(f"\nPhase 3 Pipeline Verification:")
        print(f"Train Shape: {train_split.shape}")
        print(f"Test Shape:  {test_split.shape}")
        print(f"Feature count: {len(FEATURE_COLUMNS)}")
        print(f"Sample MAP (first 3 train): {train_split['mean_arterial_pressure'].head(3).tolist()}")
        print(f"Sample Shock Index (first 3 train): {train_split['shock_index'].head(3).tolist()}")
        print("Success: Phase 3 executed without errors.")
    except Exception as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        sys.exit(1)
