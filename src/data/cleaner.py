"""Clinical Data Sanitization and Imputation Engine for MediHaven.

Applies physiological plausibility bounds clipping, removes sensor artifacts,
and executes clinically valid imputation (Last Observation Carried Forward / LOCF
for time-series vitals, robust reference imputation for laboratory biomarkers).
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("medihaven.data.cleaner")

# Clinically validated physiological bounds
PHYSIOLOGICAL_BOUNDS = {
    # Vitals
    "heart_rate": (20.0, 260.0),
    "blood_pressure_sys": (40.0, 300.0),
    "blood_pressure_dia": (20.0, 200.0),
    "temperature": (30.0, 45.0),
    "oxygen_saturation": (50.0, 100.0),
    "respiratory_rate": (4.0, 70.0),
    # Laboratory Biomarkers
    "glucose": (10.0, 1000.0),
    "creatinine": (0.1, 25.0),
    "cholesterol": (50.0, 600.0),
    "wbc_count": (0.5, 100.0),
    "bun": (1.0, 200.0),
    "hemoglobin": (2.0, 25.0),
    "platelets": (10.0, 1500.0),
}

# Standard clinical baseline reference values (used for fallback imputation)
CLINICAL_REFERENCE_DEFAULTS = {
    "heart_rate": 75.0,
    "blood_pressure_sys": 120.0,
    "blood_pressure_dia": 80.0,
    "temperature": 36.8,
    "oxygen_saturation": 98.0,
    "respiratory_rate": 16.0,
    "glucose": 95.0,
    "creatinine": 0.95,
    "cholesterol": 180.0,
    "wbc_count": 6.8,
    "bun": 15.0,
    "hemoglobin": 14.0,
    "platelets": 250.0,
}


def clip_physiological_bounds(
    df: pd.DataFrame,
    bounds_dict: Dict[str, tuple],
) -> pd.DataFrame:
    """Clips numerical columns to their respective physiological limits.

    Args:
        df: Input DataFrame.
        bounds_dict: Dictionary mapping column names to (min, max) tuples.

    Returns:
        DataFrame with bounded values.
    """
    cleaned_df = df.copy()
    cols = [col for col in bounds_dict if col in cleaned_df.columns]
    if cols:
        # Clip all columns in one vectorized numpy call (np.clip preserves NaNs for imputation step)
        lower = np.array([bounds_dict[col][0] for col in cols])
        upper = np.array([bounds_dict[col][1] for col in cols])
        cleaned_df[cols] = np.clip(cleaned_df[cols].to_numpy(dtype=float), lower, upper)
    return cleaned_df


def clean_vitals(df_vitals: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes longitudinal vitals time series and executes LOCF imputation.

    Args:
        df_vitals: Raw vitals DataFrame.

    Returns:
        Cleaned and imputed vitals DataFrame.
    """
    if df_vitals.empty:
        return df_vitals.copy()

    df = df_vitals.copy()

    # Sort chronologically by patient and timestamp
    if "recorded_at" in df.columns:
        df = df.sort_values(by=["patient_id", "recorded_at"]).reset_index(drop=True)

    # 1. Clip physiological outliers to clinical limits
    vitals_bounds = {k: v for k, v in PHYSIOLOGICAL_BOUNDS.items() if k in df.columns}
    df = clip_physiological_bounds(df, vitals_bounds)

    # 2. Impute missing vitals per patient using Last Observation Carried Forward (LOCF)
    vitals_cols = [
        "heart_rate",
        "blood_pressure_sys",
        "blood_pressure_dia",
        "temperature",
        "oxygen_saturation",
        "respiratory_rate",
    ]
    present_cols = [c for c in vitals_cols if c in df.columns]

    # Imputation is a no-op on complete series, so skip the (slow) pandas fills entirely
    if present_cols and df[present_cols].isna().to_numpy().any():
        # Group by patient_id: forward fill then backward fill
        # Built-in grouped fills are equivalent to per-group ffill().bfill() but avoid a Python lambda
        filled = df.groupby("patient_id")[present_cols].ffill()
        filled["patient_id"] = df["patient_id"]
        df[present_cols] = filled.groupby("patient_id")[present_cols].bfill()

        # Fill any remaining NaNs with standard clinical reference values
        for col in present_cols:
            if col in CLINICAL_REFERENCE_DEFAULTS:
                df[col] = df[col].fillna(CLINICAL_REFERENCE_DEFAULTS[col])

    return df


def clean_lab_results(df_labs: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes laboratory biomarker measurements and imputes missing tests.

    Args:
        df_labs: Raw lab results DataFrame.

    Returns:
        Cleaned and imputed lab results DataFrame.
    """
    if df_labs.empty:
        return df_labs.copy()

    df = df_labs.copy()

    if "recorded_at" in df.columns:
        df = df.sort_values(by=["patient_id", "recorded_at"]).reset_index(drop=True)

    # 1. Clip laboratory outliers
    lab_bounds = {k: v for k, v in PHYSIOLOGICAL_BOUNDS.items() if k in df.columns}
    df = clip_physiological_bounds(df, lab_bounds)

    # 2. Impute missing tests: LOCF per patient first, then cohort median, then clinical default
    lab_cols = [
        "glucose",
        "creatinine",
        "cholesterol",
        "wbc_count",
        "bun",
        "hemoglobin",
        "platelets",
    ]
    present_cols = [c for c in lab_cols if c in df.columns]

    # Imputation is a no-op on complete series, so skip the (slow) pandas fills entirely
    if present_cols and df[present_cols].isna().to_numpy().any():
        # Built-in grouped fills are equivalent to per-group ffill().bfill() but avoid a Python lambda
        filled = df.groupby("patient_id")[present_cols].ffill()
        filled["patient_id"] = df["patient_id"]
        df[present_cols] = filled.groupby("patient_id")[present_cols].bfill()

        # Impute remaining missing values with median or clinical reference
        for col in present_cols:
            median_val = df[col].median()
            fallback_val = (
                median_val
                if not np.isnan(median_val)
                else CLINICAL_REFERENCE_DEFAULTS.get(col, 0.0)
            )
            df[col] = df[col].fillna(fallback_val)

    return df


def clean_multimodal_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Applies domain-specific cleaning to all clinical DataFrames.

    Args:
        data_dict: Raw multimodal dataset mapping from loader.

    Returns:
        Cleaned multimodal dataset dictionary.
    """
    logger.info("Executing clinical data cleaning and imputation pipeline...")
    cleaned = dict(data_dict)

    if "vitals" in cleaned:
        cleaned["vitals"] = clean_vitals(cleaned["vitals"])

    if "labs" in cleaned:
        cleaned["labs"] = clean_lab_results(cleaned["labs"])

    return cleaned
