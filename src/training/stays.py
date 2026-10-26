"""Stay-level view of the research cohort for the local research triage page (DATA_MODE=mimic only).

Each ICU stay's risk comes from its out-of-fold prediction: the model that scores a stay was trained
on folds that contain none of that patient's stays, exactly as in the Phase 2 evaluation. (The final
models were fitted on every stay, so their scores for these same stays would look far better than
the models really are.) The out-of-fold predictions are recomputed in memory from the verified
dataset with the recorded seed and folds, then checked against the saved Phase 2 metrics: the fold
assignment fingerprint and every model's pooled AUROC must match, or nothing is served.

Privacy: rows are numbered "Stay 1..N"; MIMIC identifiers, index times and death times never leave
this module, and nothing patient-level is logged or written to disk. Research mode itself only
serves this machine (loopback), so these rows never reach a network.
"""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.datasources.mimic.mappings import VARIABLES
from src.tasks.build_mimic_t1 import LABEL_COLUMN
from src.tasks.mortality_icu_24h import ANOMALY_COLUMN
from src.training.dataset import SUBJECT_COLUMN, DatasetValidationError, default_dataset_dir, load_task1_dataset
from src.training.features import build_feature_matrix
from src.training.models import MODEL_ORDER, REFERENCE_MODEL, candidate_models
from src.training.splits import assignment_fingerprint, make_grouped_folds
from src.utils.logger import get_logger

logger = get_logger("medihaven.research.stays")

REAL_MODELS = tuple(m for m in MODEL_ORDER if m != REFERENCE_MODEL)
TIERS = ("Low", "Medium", "High", "Critical")
# Display bands as multiples of the cohort death rate (fixed in advance; not clinical thresholds)
TIER_MULTIPLES = (1.0, 2.0, 4.0)
AUROC_TOLERANCE = 1e-9

VARIABLE_LABELS = {
    "heart_rate": "Heart rate", "resp_rate": "Respiratory rate", "spo2": "SpO₂", "temperature": "Temperature",
    "sbp": "Systolic BP", "dbp": "Diastolic BP", "map": "Mean arterial pressure", "creatinine": "Creatinine",
    "bun": "Blood urea nitrogen", "wbc": "White blood cells", "hemoglobin": "Hemoglobin",
    "platelets": "Platelets", "lactate": "Lactate", "glucose": "Glucose",
}
# Typical adult reference ranges, for highlighting only (not used by any model)
REFERENCE_RANGES = {
    "heart_rate": (60, 100), "resp_rate": (12, 20), "spo2": (95, 100), "temperature": (36.1, 37.8),
    "sbp": (90, 140), "dbp": (60, 90), "map": (70, 105), "creatinine": (0.6, 1.3), "bun": (7, 20),
    "wbc": (4, 11), "hemoglobin": (12, 17.5), "platelets": (150, 400), "lactate": (0.5, 2.0),
    "glucose": (70, 140),
}
ROSTER_VITALS = ("heart_rate", "sbp", "spo2", "resp_rate", "lactate")


class StayViewError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def tier_for(probability: float, death_rate: float) -> str:
    low, medium, high = (death_rate * m for m in TIER_MULTIPLES)
    if probability < low:
        return "Low"
    if probability < medium:
        return "Medium"
    if probability < high:
        return "High"
    return "Critical"


def _number(value) -> Optional[float]:
    return None if value is None or pd.isna(value) else round(float(value), 2)


@dataclass
class StayView:
    death_rate: float
    bands: Dict[str, float]
    rows: List[Dict[str, object]]
    details: Dict[int, Dict[str, object]]


def _out_of_fold(frame: pd.DataFrame, features: List[str], n_folds: int, seed: int):
    """Mirrors evaluation.cross_validate: same folds, same pipelines, one prediction per stay."""
    X = build_feature_matrix(frame, features)
    y = frame[LABEL_COLUMN].to_numpy(dtype=int)
    groups = frame[SUBJECT_COLUMN].to_numpy()
    folds = make_grouped_folds(y, groups, n_folds, seed)
    pooled = {name: np.full(len(y), np.nan) for name in MODEL_ORDER}
    fold_of = np.zeros(len(y), dtype=int)
    for k, (train_idx, test_idx) in enumerate(folds):
        fold_of[test_idx] = k + 1
        for name, prototype in candidate_models(seed).items():
            model = clone(prototype).fit(X.iloc[train_idx], y[train_idx])
            pooled[name][test_idx] = model.predict_proba(X.iloc[test_idx])[:, list(model.classes_).index(1)]
    return y, groups, folds, pooled, fold_of


def build_stay_view(research_dir: Path, features: List[str], metrics: Dict[str, object]) -> StayView:
    from sklearn.metrics import roc_auc_score

    try:
        dataset = load_task1_dataset(default_dataset_dir(research_dir))  # schema + content hash checks
    except DatasetValidationError as e:
        raise StayViewError("DATASET_INVALID", f"The research dataset failed validation: {e}") from e

    primary = metrics["analyses"]["primary"]
    split = primary["split"]
    # Stable numbering independent of file order; identifiers are used here only, never returned
    frame = dataset.frame.sort_values("src_stay_id").reset_index(drop=True)
    # The evaluation used the file's row order; reproduce it exactly, then map back to the numbering
    original = dataset.frame.reset_index(drop=True)
    y, groups, folds, pooled, fold_of = _out_of_fold(original, features, int(split["n_folds"]), int(split["seed"]))

    if assignment_fingerprint(folds, groups) != split["assignment_sha256"]:
        raise StayViewError("SPLIT_MISMATCH", "Recomputed folds differ from the verified Phase 2 run.")
    for name in MODEL_ORDER:
        saved = primary["models"][name]["pooled_out_of_fold"]["auroc"]
        recomputed = float(roc_auc_score(y, pooled[name]))
        if saved is None or abs(recomputed - saved) > AUROC_TOLERANCE:
            raise StayViewError("PREDICTION_MISMATCH",
                                "Recomputed out-of-fold predictions do not reproduce the verified Phase 2 metrics.")

    position = {stay_id: i for i, stay_id in enumerate(original["src_stay_id"])}
    death_rate = float(primary["population"]["prevalence"])
    rows, details = [], {}
    for number, r in enumerate(frame.to_dict("records"), start=1):
        i = position[r["src_stay_id"]]
        probabilities = {name: round(float(pooled[name][i]), 4) for name in MODEL_ORDER}
        consensus = round(float(np.mean([pooled[name][i] for name in REAL_MODELS])), 4)
        votes = {name: {"probability": probabilities[name], "tier": tier_for(probabilities[name], death_rate)}
                 for name in REAL_MODELS}
        sex = None if pd.isna(r.get("sex_male")) else ("M" if int(r["sex_male"]) == 1 else "F")
        measured = sum(int(r[f"{v.name}_missing"]) == 0 for v in VARIABLES)
        base = {
            "stay": number,
            "age": None if pd.isna(r.get("age")) else int(round(float(r["age"]))),
            "sex": sex,
            "risk": consensus,
            "tier": tier_for(consensus, death_rate),
            "models": votes,
            "died": bool(int(r[LABEL_COLUMN]) == 1),
            "variables_measured": measured,
            "latest": {v: _number(r[f"{v}_last"]) for v in ROSTER_VITALS},
        }
        rows.append(base)
        details[number] = {
            **base,
            "baseline_probability": probabilities[REFERENCE_MODEL],
            "fold": int(fold_of[i]),
            "outcome_note": ("Death recorded after the hospital discharge time (data anomaly; counted as a death)"
                             if int(r[ANOMALY_COLUMN]) == 1 else None),
            "bp_source": None if pd.isna(r.get("bp_source_invasive")) else
            ("arterial line" if int(r["bp_source_invasive"]) == 1 else "cuff"),
            "map_derived": None if pd.isna(r.get("map_derived")) else bool(int(r["map_derived"]) == 1),
            "variables": [
                {
                    "name": v.name, "label": VARIABLE_LABELS.get(v.name, v.name), "unit": v.unit,
                    "measured": int(r[f"{v.name}_missing"]) == 0,
                    "count": int(r[f"{v.name}_count"]),
                    "min": _number(r[f"{v.name}_min"]), "max": _number(r[f"{v.name}_max"]),
                    "mean": _number(r[f"{v.name}_mean"]), "last": _number(r[f"{v.name}_last"]),
                    "reference": REFERENCE_RANGES.get(v.name),
                    "plausible": (v.plausible_min, v.plausible_max),
                }
                for v in VARIABLES
            ],
        }
    bands = {"Low": 0.0, **{t: round(death_rate * m, 4) for t, m in zip(TIERS[1:], TIER_MULTIPLES)}}
    logger.info(f"Research stay view ready: {len(rows)} stays, out-of-fold predictions match the verified run")
    return StayView(death_rate=death_rate, bands=bands, rows=rows, details=details)


class LazyStayView:
    """Builds the stay view on first use (a few seconds of model fitting) and caches it, or its failure."""

    def __init__(self, research_dir: Path, features: List[str], metrics: Dict[str, object]):
        self._args = (research_dir, features, metrics)
        self._lock = threading.Lock()
        self._view: Optional[StayView] = None
        self._error: Optional[StayViewError] = None

    def get(self) -> StayView:
        with self._lock:
            if self._view is None and self._error is None:
                try:
                    self._view = build_stay_view(*self._args)
                except StayViewError as e:
                    logger.warning(f"Research stay view unavailable: {e.code}")
                    self._error = e
                except Exception as e:  # fail closed without exposing details
                    logger.warning(f"Research stay view unavailable: unexpected {type(e).__name__}")
                    self._error = StayViewError("STAY_VIEW_ERROR", "The stay-level view could not be prepared.")
            if self._error is not None:
                raise self._error
            return self._view
