"""Cross-validated evaluation of the Task 1 baselines and the outcome-anomaly sensitivity analyses.

Strategy: subject-grouped, stratified K-fold cross-validation. Each stay is predicted exactly once,
by a model fitted on folds that contain none of that subject's stays; metrics are computed on the
pooled out-of-fold predictions. There is no separate holdout: with the MIMIC-IV Demo's handful of
positive stays, a single holdout split would contain too few deaths to evaluate anything.

Analyses:
- primary: all eligible stays, version 1.2.0 labels.
- sensitivity_evaluation_only: the primary out-of-fold predictions re-scored after removing
  outcome-anomaly stays (same folds and models; isolates the effect on the evaluation population).
- sensitivity_retrained: anomaly stays removed before splitting and training (new folds over the
  smaller population; not fold-for-fold comparable with the primary analysis).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.tasks.build_mimic_t1 import LABEL_COLUMN
from src.tasks.mortality_icu_24h import ANOMALY_COLUMN
from src.training.dataset import SUBJECT_COLUMN
from src.training.features import build_feature_matrix
from src.training.metrics import CI_METRICS, binary_metrics, subject_bootstrap_ci
from src.training.models import MODEL_ORDER, candidate_models
from src.training.splits import InsufficientDataError, assignment_fingerprint, fold_summary, make_grouped_folds

EVALUATION_STRATEGY = "subject-grouped stratified K-fold cross-validation, pooled out-of-fold predictions"
# Heuristic: below this many positive stays every metric is flagged as limited and unstable
LIMITED_EVALUATION_MAX_POSITIVES = 100


def population_summary(frame: pd.DataFrame) -> Dict[str, object]:
    positives = int(frame[LABEL_COLUMN].sum())
    return {
        "stays": int(len(frame)),
        "subjects": int(frame[SUBJECT_COLUMN].nunique()),
        "positives": positives,
        "negatives": int(len(frame) - positives),
        "prevalence": round(positives / len(frame), 4) if len(frame) else None,
        "outcome_anomaly_stays": int(frame[ANOMALY_COLUMN].sum()),
    }


def evaluation_status(positives: int) -> Dict[str, str]:
    if positives < LIMITED_EVALUATION_MAX_POSITIVES:
        return {"status": "limited",
                "reason": f"only {positives} positive stays (< {LIMITED_EVALUATION_MAX_POSITIVES}); metrics are "
                          f"unstable, wide-interval and suitable for pipeline validation only"}
    return {"status": "standard", "reason": "sufficient positives for indicative estimates"}


def _positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def _fold_spread(per_fold: List[Dict[str, object]], metric: str) -> Dict[str, object]:
    values = [f[metric] for f in per_fold if f[metric] is not None]
    if not values:
        return {"mean": None, "sd": None, "folds_defined": 0}
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "folds_defined": len(values)}


def cross_validate(frame: pd.DataFrame, features: List[str], n_folds: int, seed: int, n_boot: int) -> Dict[str, object]:
    """Runs grouped stratified CV for every candidate model. Raises InsufficientDataError if infeasible.

    Returns aggregate results plus "_internal" arrays (pooled predictions) used only in memory for
    the evaluation-only sensitivity analysis; they are never written to disk.
    """
    frame = frame.reset_index(drop=True)
    X = build_feature_matrix(frame, features)
    y = frame[LABEL_COLUMN].to_numpy(dtype=int)
    groups = frame[SUBJECT_COLUMN].to_numpy()
    folds = make_grouped_folds(y, groups, n_folds, seed)

    thresholds = np.full(len(y), np.nan)
    pooled = {name: np.full(len(y), np.nan) for name in MODEL_ORDER}
    per_fold = {name: [] for name in MODEL_ORDER}
    for train_idx, test_idx in folds:
        fold_threshold = float(y[train_idx].mean())
        thresholds[test_idx] = fold_threshold
        for name, prototype in candidate_models(seed).items():
            model = clone(prototype).fit(X.iloc[train_idx], y[train_idx])
            proba = _positive_proba(model, X.iloc[test_idx])
            pooled[name][test_idx] = proba
            per_fold[name].append(binary_metrics(y[test_idx], proba, fold_threshold))

    if np.isnan(thresholds).any() or any(np.isnan(p).any() for p in pooled.values()):
        raise AssertionError("Some stays did not receive an out-of-fold prediction")

    models = {}
    for name in MODEL_ORDER:
        models[name] = {
            "pooled_out_of_fold": binary_metrics(y, pooled[name], thresholds),
            "bootstrap_ci_95": subject_bootstrap_ci(y, pooled[name], groups, n_boot, seed),
            "per_fold": per_fold[name],
            "fold_spread": {m: _fold_spread(per_fold[name], m) for m in CI_METRICS},
        }

    return {
        "population": population_summary(frame),
        "evaluation": evaluation_status(int(y.sum())),
        "split": {
            "strategy": EVALUATION_STRATEGY,
            "n_folds": n_folds,
            "seed": seed,
            "folds": fold_summary(folds, y, groups),
            "assignment_sha256": assignment_fingerprint(folds, groups),
        },
        "models": models,
        "_internal": {"y": y, "groups": groups, "thresholds": thresholds, "pooled": pooled,
                      "anomaly": frame[ANOMALY_COLUMN].to_numpy(dtype=int)},
    }


def restricted_rescore(primary: Dict[str, object], keep: np.ndarray, n_boot: int, seed: int) -> Dict[str, object]:
    """Re-scores the primary pooled predictions on a subset of stays (same folds and models)."""
    internal = primary["_internal"]
    y, groups, thresholds = internal["y"][keep], internal["groups"][keep], internal["thresholds"][keep]
    positives = int(y.sum())
    models = {}
    for name in MODEL_ORDER:
        proba = internal["pooled"][name][keep]
        models[name] = {
            "pooled_out_of_fold": binary_metrics(y, proba, thresholds),
            "bootstrap_ci_95": subject_bootstrap_ci(y, proba, groups, n_boot, seed),
        }
    return {
        "population": {
            "stays": int(keep.sum()), "subjects": int(len(np.unique(groups))), "positives": positives,
            "negatives": int(keep.sum() - positives),
            "prevalence": round(positives / int(keep.sum()), 4) if keep.any() else None,
            "outcome_anomaly_stays": int(internal["anomaly"][keep].sum()),
        },
        "evaluation": evaluation_status(positives),
        "split": {"strategy": "primary folds and models; anomaly stays removed from scoring only"},
        "models": models,
    }


def public(result: Dict[str, object]) -> Dict[str, object]:
    """Drops in-memory arrays so only aggregate results are persisted."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def run_analyses(frame: pd.DataFrame, features: List[str], n_folds: int, seed: int, n_boot: int) -> Dict[str, object]:
    """Primary analysis plus both anomaly sensitivity analyses (aggregate results only)."""
    primary = cross_validate(frame, features, n_folds, seed, n_boot)  # infeasible primary -> run fails

    keep = primary["_internal"]["anomaly"] == 0
    evaluation_only = restricted_rescore(primary, keep, n_boot, seed)

    reduced = frame[frame[ANOMALY_COLUMN] == 0]
    try:
        retrained: Optional[Dict[str, object]] = public(cross_validate(reduced, features, n_folds, seed, n_boot))
    except InsufficientDataError as e:
        retrained = {"population": population_summary(reduced),
                     "evaluation": {"status": "not_evaluable", "reason": str(e)}}

    return {
        "primary": public(primary),
        "sensitivity_evaluation_only": evaluation_only,
        "sensitivity_retrained": retrained,
    }
