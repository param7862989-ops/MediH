"""Permitted model inputs for Task 1, with provenance and hard guards against leakage.

Only feature-spec columns whose availability at the prediction time (ICU intime + 24 h) can be
established are used. Identifiers, the index time, the label, the outcome-anomaly flag and the
version stamps are excluded by name, and any column whose name suggests outcome, discharge,
death, identifier or prior-prediction content is rejected outright.
"""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.datasources.mimic.mappings import AGGREGATIONS, VARIABLES
from src.tasks.build_mimic_t1 import ID_COLUMNS, LABEL_COLUMN
from src.tasks.mortality_icu_24h import ANOMALY_COLUMN

# Columns of the dataset that are never model inputs, with the reason
EXCLUDED_COLUMNS: Dict[str, str] = {
    **{col: "source identifier (used only for grouping/joins)" for col in ID_COLUMNS},
    "index_time": "prediction timestamp, not a clinical measurement",
    LABEL_COLUMN: "outcome label",
    ANOMALY_COLUMN: "outcome-derived label metadata (sensitivity analysis only)",
    "dataset_version": "version stamp",
    "feature_spec_version": "version stamp",
}

# Name fragments that must never appear in a model input
FORBIDDEN_NAME_PATTERNS = (
    r"label", r"anomal", r"mortal", r"death", r"(^|_)dod($|_)", r"disch", r"expire", r"outcome",
    r"^src_", r"subject", r"hadm", r"stay_id", r"(^|_)id$", r"index_time", r"risk_tier", r"risk_score",
    r"predict", r"outtime", r"(^|_)los($|_)", r"version",
)


class ForbiddenFeatureError(ValueError):
    """Raised when a forbidden column would enter the training matrix."""


def forbidden_reason(column: str) -> Optional[str]:
    """Returns why a column may not be a model input, or None if it is not forbidden by name."""
    if column in EXCLUDED_COLUMNS:
        return EXCLUDED_COLUMNS[column]
    lowered = column.lower()
    for pattern in FORBIDDEN_NAME_PATTERNS:
        if re.search(pattern, lowered):
            return f"name matches forbidden pattern '{pattern}'"
    return None


def feature_provenance(column: str) -> Optional[str]:
    """Explains how a feature is available at the prediction time, or None if it cannot be established."""
    window = ("aggregated from measurements with intime <= charttime <= index_time "
              "and (reliable) storetime <= index_time")
    for var in VARIABLES:
        if column in {f"{var.name}_{agg}" for agg in AGGREGATIONS} | {f"{var.name}_missing"}:
            return f"{var.name}: {window}"
    known = {
        "map_derived": f"whether MAP was derived from SBP/DBP pairs; {window}",
        "bp_source_invasive": f"whether an arterial-line BP reading exists; {window}",
        "age": "age at hospital admission (anchor_age + admission year - anchor_year), known at admission",
        "sex_male": "recorded sex, known at admission",
    }
    return known.get(column)


def select_features(candidates: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Returns (permitted features, excluded {column: reason}).

    A forbidden candidate raises ForbiddenFeatureError: it means the feature specification itself
    is wrong and must be fixed, not silently patched. Candidates without established provenance
    are excluded and reported.
    """
    forbidden = {c: forbidden_reason(c) for c in candidates if forbidden_reason(c)}
    if forbidden:
        raise ForbiddenFeatureError(f"Forbidden columns in the feature list: {forbidden}")
    permitted, excluded = [], {}
    for column in candidates:
        if feature_provenance(column) is None:
            excluded[column] = "availability at the prediction time cannot be established"
        else:
            permitted.append(column)
    return permitted, excluded


def build_feature_matrix(frame: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Selects the permitted features by name (never by position) and re-checks the guard."""
    bad = {c: forbidden_reason(c) for c in features if forbidden_reason(c)}
    if bad:
        raise ForbiddenFeatureError(f"Refusing to build a training matrix with forbidden columns: {bad}")
    return frame.loc[:, list(features)].astype("float64")
