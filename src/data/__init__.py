"""Clinical data ingestion, preprocessing, and feature engineering package."""

from src.data.loader import (
    load_patients,
    load_vitals,
    load_lab_results,
    load_medications,
    load_outcomes,
    load_predictions,
    load_raw_multimodal_dataset,
)
from src.data.cleaner import (
    clean_vitals,
    clean_lab_results,
    clean_multimodal_data,
    PHYSIOLOGICAL_BOUNDS,
)
from src.data.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TIER_MAP,
    extract_patient_features,
    build_feature_matrix,
)
from src.data.scaler import ClinicalFeatureScaler
from src.data.pipeline import run_batch_pipeline, transform_single_patient

__all__ = [
    "load_patients",
    "load_vitals",
    "load_lab_results",
    "load_medications",
    "load_outcomes",
    "load_predictions",
    "load_raw_multimodal_dataset",
    "clean_vitals",
    "clean_lab_results",
    "clean_multimodal_data",
    "PHYSIOLOGICAL_BOUNDS",
    "FEATURE_COLUMNS",
    "TARGET_COLUMNS",
    "TIER_MAP",
    "extract_patient_features",
    "build_feature_matrix",
    "ClinicalFeatureScaler",
    "run_batch_pipeline",
    "transform_single_patient",
]
