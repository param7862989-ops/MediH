"""Clinical Feature Engineering and Longitudinal Trajectory Modeling for MediHaven.

Translates raw physiological time series and laboratory biomarkers into clinically
actionable, normalized mathematical features:
- Hemodynamic indicators (Mean Arterial Pressure, Pulse Pressure, Shock Index)
- 72-Hour Longitudinal Trajectory Slopes (Ordinary Least Squares linear regression)
- Organ Injury & Volatility Markers (AKI Creatinine Delta, BUN/Creatinine Ratio)
- Demographic & Medication Comorbidity Burden
"""

from datetime import timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("medihaven.data.features")

# Standard clinical feature column ordering for consistency across training & inference
FEATURE_COLUMNS: List[str] = [
    # 1. Demographics
    "age",
    "gender_encoded",
    # 2. Latest Vitals
    "heart_rate_latest",
    "sbp_latest",
    "dbp_latest",
    "temperature_latest",
    "oxygen_saturation_latest",
    "respiratory_rate_latest",
    # 3. Longitudinal 72h Vitals & Slopes
    "heart_rate_mean",
    "heart_rate_slope",
    "sbp_mean",
    "sbp_slope",
    "spo2_mean",
    "spo2_min",
    "spo2_slope",
    "temp_max",
    # 4. Compound Hemodynamic Indices
    "mean_arterial_pressure",
    "pulse_pressure",
    "shock_index",
    "modified_shock_index",
    # 5. Laboratory Biomarkers & Volatilities
    "glucose_latest",
    "glucose_delta",
    "creatinine_latest",
    "creatinine_delta",
    "wbc_latest",
    "bun_latest",
    "bun_creatinine_ratio",
    "hemoglobin_latest",
    "platelets_latest",
    # 6. Comorbidity Burden
    "active_medications_count",
]

TARGET_COLUMNS: List[str] = [
    "risk_tier_encoded",
    "risk_score",
    "readmitted_30d",
    "icu_transfer",
]

TIER_MAP = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3,
}
REVERSE_TIER_MAP = {v: k for k, v in TIER_MAP.items()}


def calculate_linear_slope(time_hours: np.ndarray, values: np.ndarray) -> float:
    """Calculates Ordinary Least Squares (OLS) slope (change in units per hour).

    Args:
        time_hours: 1D array of elapsed hours.
        values: 1D array of physiological measurements.

    Returns:
        Linear slope rate (beta_1). Returns 0.0 if fewer than 2 valid points.
    """
    if len(time_hours) < 2 or len(values) < 2:
        return 0.0

    # Avoid zero-variance in time
    dt = time_hours - np.mean(time_hours)
    var_t = np.sum(dt**2)
    if var_t < 1e-6:
        return 0.0

    dy = values - np.mean(values)
    slope = float(np.sum(dt * dy) / var_t)
    return slope


def extract_patient_features(
    patient_id: int,
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    meds_df: pd.DataFrame,
    patient_row: pd.Series,
) -> Dict[str, float]:
    """Extracts the complete 28-feature clinical vector for a single patient.

    Args:
        patient_id: Target patient ID.
        vitals_df: Cleaned vitals DataFrame (filtered or global).
        labs_df: Cleaned labs DataFrame (filtered or global).
        meds_df: Cleaned medications DataFrame (filtered or global).
        patient_row: Row containing patient demographics.

    Returns:
        Dictionary mapping feature names to numerical values.
    """
    # 1. Demographics
    age = float(patient_row.get("age", 50))
    gender = str(patient_row.get("gender", "M")).upper()
    gender_encoded = 1.0 if gender == "M" else 0.0

    # 2. Longitudinal Vitals Processing
    p_vitals = vitals_df[vitals_df["patient_id"] == patient_id].sort_values("recorded_at")

    if not p_vitals.empty:
        latest_vital = p_vitals.iloc[-1]
        hr_latest = float(latest_vital["heart_rate"])
        sbp_latest = float(latest_vital["blood_pressure_sys"])
        dbp_latest = float(latest_vital["blood_pressure_dia"])
        temp_latest = float(latest_vital["temperature"])
        spo2_latest = float(latest_vital["oxygen_saturation"])
        resp_latest = float(latest_vital["respiratory_rate"])

        # Longitudinal 72h window calculation
        max_time = p_vitals["recorded_at"].max()
        window_72h = max_time - timedelta(hours=72)
        vitals_72h = p_vitals[p_vitals["recorded_at"] >= window_72h]
        if vitals_72h.empty:
            vitals_72h = p_vitals

        hr_mean = float(vitals_72h["heart_rate"].mean())
        sbp_mean = float(vitals_72h["blood_pressure_sys"].mean())
        spo2_mean = float(vitals_72h["oxygen_saturation"].mean())
        spo2_min = float(vitals_72h["oxygen_saturation"].min())
        temp_max = float(vitals_72h["temperature"].max())

        # Elapsed hours for slope calculation
        t_start = vitals_72h["recorded_at"].iloc[0]
        elapsed_hours = (
            (vitals_72h["recorded_at"] - t_start).dt.total_seconds() / 3600.0
        ).values

        hr_slope = calculate_linear_slope(elapsed_hours, vitals_72h["heart_rate"].values)
        sbp_slope = calculate_linear_slope(
            elapsed_hours, vitals_72h["blood_pressure_sys"].values
        )
        spo2_slope = calculate_linear_slope(
            elapsed_hours, vitals_72h["oxygen_saturation"].values
        )
    else:
        # Default fallback values for patients with no vitals recordings
        hr_latest = hr_mean = 75.0
        sbp_latest = sbp_mean = 120.0
        dbp_latest = 80.0
        temp_latest = temp_max = 36.8
        spo2_latest = spo2_mean = spo2_min = 98.0
        resp_latest = 16.0
        hr_slope = sbp_slope = spo2_slope = 0.0

    # 3. Compound Hemodynamic Indices
    map_val = dbp_latest + (1.0 / 3.0) * (sbp_latest - dbp_latest)
    pulse_pressure = sbp_latest - dbp_latest
    shock_index = hr_latest / max(sbp_latest, 1.0)
    mod_shock_index = hr_latest / max(map_val, 1.0)

    # 4. Laboratory Biomarkers Processing
    p_labs = labs_df[labs_df["patient_id"] == patient_id].sort_values("recorded_at")

    if not p_labs.empty:
        latest_lab = p_labs.iloc[-1]
        first_lab = p_labs.iloc[0]

        gluc_latest = float(latest_lab["glucose"])
        gluc_delta = float(latest_lab["glucose"] - first_lab["glucose"])

        creat_latest = float(latest_lab["creatinine"])
        creat_delta = float(latest_lab["creatinine"] - first_lab["creatinine"])

        wbc_latest = float(latest_lab["wbc_count"])
        bun_latest = float(latest_lab.get("bun", 15.0))
        bun_creat_ratio = bun_latest / max(creat_latest, 0.1)

        hgb_latest = float(latest_lab.get("hemoglobin", 14.0))
        plt_latest = float(latest_lab.get("platelets", 250.0))
    else:
        gluc_latest, gluc_delta = 95.0, 0.0
        creat_latest, creat_delta = 0.95, 0.0
        wbc_latest = 6.8
        bun_latest = 15.0
        bun_creat_ratio = 15.0 / 0.95
        hgb_latest = 14.0
        plt_latest = 250.0

    # 5. Comorbidity Burden
    p_meds = meds_df[meds_df["patient_id"] == patient_id]
    active_meds_count = float(len(p_meds[p_meds["is_active"] == 1]))

    return {
        "age": age,
        "gender_encoded": gender_encoded,
        "heart_rate_latest": hr_latest,
        "sbp_latest": sbp_latest,
        "dbp_latest": dbp_latest,
        "temperature_latest": temp_latest,
        "oxygen_saturation_latest": spo2_latest,
        "respiratory_rate_latest": resp_latest,
        "heart_rate_mean": hr_mean,
        "heart_rate_slope": hr_slope,
        "sbp_mean": sbp_mean,
        "sbp_slope": sbp_slope,
        "spo2_mean": spo2_mean,
        "spo2_min": spo2_min,
        "spo2_slope": spo2_slope,
        "temp_max": temp_max,
        "mean_arterial_pressure": map_val,
        "pulse_pressure": pulse_pressure,
        "shock_index": shock_index,
        "modified_shock_index": mod_shock_index,
        "glucose_latest": gluc_latest,
        "glucose_delta": gluc_delta,
        "creatinine_latest": creat_latest,
        "creatinine_delta": creat_delta,
        "wbc_latest": wbc_latest,
        "bun_latest": bun_latest,
        "bun_creatinine_ratio": bun_creat_ratio,
        "hemoglobin_latest": hgb_latest,
        "platelets_latest": plt_latest,
        "active_medications_count": active_meds_count,
    }


def build_feature_matrix(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Builds the comprehensive feature matrix and labels for all patients.

    Args:
        data_dict: Cleaned multimodal dataset from cleaner.

    Returns:
        DataFrame containing all feature columns and target labels.
    """
    logger.info("Building clinical feature matrix...")

    df_patients = data_dict["patients"]
    df_vitals = data_dict["vitals"]
    df_labs = data_dict["labs"]
    df_meds = data_dict["medications"]
    df_outcomes = data_dict.get("outcomes", pd.DataFrame())
    df_preds = data_dict.get("predictions", pd.DataFrame())

    records = []

    for _, p_row in df_patients.iterrows():
        pid = int(p_row["patient_id"])
        feats = extract_patient_features(
            patient_id=pid,
            vitals_df=df_vitals,
            labs_df=df_labs,
            meds_df=df_meds,
            patient_row=p_row,
        )
        feats["patient_id"] = pid
        feats["mrn"] = p_row["mrn"]

        # Ground truth outcome targets
        if not df_outcomes.empty:
            p_out = df_outcomes[df_outcomes["patient_id"] == pid]
            if not p_out.empty:
                feats["readmitted_30d"] = int(p_out.iloc[-1].get("readmitted_30d", 0))
                feats["icu_transfer"] = int(p_out.iloc[-1].get("icu_transfer", 0))
            else:
                feats["readmitted_30d"] = 0
                feats["icu_transfer"] = 0

        # Prediction ground-truth tier targets
        if not df_preds.empty:
            p_pred = df_preds[df_preds["patient_id"] == pid]
            if not p_pred.empty:
                tier_str = str(p_pred.iloc[-1].get("risk_tier", "Low"))
                feats["risk_tier"] = tier_str
                feats["risk_tier_encoded"] = TIER_MAP.get(tier_str, 0)
                feats["risk_score"] = float(p_pred.iloc[-1].get("risk_score", 0.1))
            else:
                feats["risk_tier"] = "Low"
                feats["risk_tier_encoded"] = 0
                feats["risk_score"] = 0.1

        records.append(feats)

    feature_df = pd.DataFrame(records)
    logger.info(
        f"Feature matrix successfully constructed: {feature_df.shape[0]} patients, "
        f"{len(FEATURE_COLUMNS)} features per patient."
    )
    return feature_df
