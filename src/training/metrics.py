"""Binary classification metrics for an imbalanced outcome, with explicit undefined handling.

A metric that is mathematically undefined for the given data (e.g. AUROC with one class present)
is reported as None together with the reason; it is never replaced by 0.

Average precision is scikit-learn's ``average_precision_score`` (step-wise sum over recall
increments of precision), not a trapezoidal area under the precision-recall curve.

Decision threshold: a stay is predicted positive when its predicted risk is >= the outcome
prevalence of the training data that produced the prediction (per fold in cross-validation).
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

THRESHOLD_RULE = "predicted risk >= outcome prevalence of the training data (per fold)"
AVERAGE_PRECISION_DEFINITION = "sklearn.metrics.average_precision_score (step-wise, not trapezoidal AUPRC)"
# Guards float round-off when a prior-only model predicts exactly the training prevalence
THRESHOLD_TOLERANCE = 1e-12
CI_METRICS = ("auroc", "average_precision", "brier_score")


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator == 0 else numerator / denominator


def binary_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: Union[float, Sequence[float]],
) -> Dict[str, object]:
    """Computes threshold-free and thresholded metrics; undefined values are None with a reason."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    thr = np.broadcast_to(np.asarray(threshold, dtype=float), p.shape)
    n, positives = len(y), int(y.sum())
    negatives = n - positives
    undefined: Dict[str, str] = {}

    def need(condition: bool, metric: str, reason: str) -> bool:
        if not condition:
            undefined[metric] = reason
        return condition

    pred = (p >= thr - THRESHOLD_TOLERANCE).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    auroc = float(roc_auc_score(y, p)) if need(positives > 0 and negatives > 0, "auroc",
                                                 "requires both positive and negative cases") else None
    ap = float(average_precision_score(y, p)) if need(positives > 0, "average_precision",
                                                        "requires at least one positive case") else None
    brier = float(brier_score_loss(y, p, pos_label=1)) if need(n > 0, "brier_score", "no cases") else None

    sensitivity = _ratio(tp, tp + fn)
    need(sensitivity is not None, "sensitivity", "no positive cases")
    specificity = _ratio(tn, tn + fp)
    need(specificity is not None, "specificity", "no negative cases")
    precision = _ratio(tp, tp + fp)
    need(precision is not None, "precision", "no predicted positives at this threshold")
    f1 = None
    if precision is not None and sensitivity is not None:
        if precision + sensitivity > 0:
            f1 = 2 * precision * sensitivity / (precision + sensitivity)
        else:
            undefined["f1"] = "precision and sensitivity are both 0"
    else:
        undefined["f1"] = "precision or sensitivity undefined"

    return {
        "n": n, "positives": positives, "negatives": negatives,
        "prevalence": _ratio(positives, n),
        "auroc": auroc, "average_precision": ap, "brier_score": brier,
        "sensitivity": sensitivity, "specificity": specificity, "precision": precision, "f1": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "undefined": undefined,
    }


def subject_bootstrap_ci(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    groups: Sequence[object],
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, object]]:
    """Percentile CIs from resampling subjects (clusters) with replacement.

    Applied to pooled out-of-fold predictions, so it reflects case-mix sampling variability only,
    not model-refitting variability; with few positives the intervals are wide and indicative only.
    Replicates in which a metric is undefined are skipped and counted.
    """
    y, p, g = np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float), np.asarray(groups)
    subjects = np.unique(g)
    rows_by_subject = {s: np.flatnonzero(g == s) for s in subjects}
    rng = np.random.default_rng(seed)
    draws = {m: [] for m in CI_METRICS}
    for _ in range(n_boot):
        idx = np.concatenate([rows_by_subject[s] for s in rng.choice(subjects, size=len(subjects), replace=True)])
        yb, pb = y[idx], p[idx]
        has_pos, has_neg = yb.any(), (yb == 0).any()
        if has_pos and has_neg:
            draws["auroc"].append(roc_auc_score(yb, pb))
        if has_pos:
            draws["average_precision"].append(average_precision_score(yb, pb))
        draws["brier_score"].append(brier_score_loss(yb, pb, pos_label=1))
    result = {}
    for metric, values in draws.items():
        if len(values) < 0.9 * n_boot:
            result[metric] = {"low": None, "high": None, "valid_replicates": len(values),
                              "reason": "too many bootstrap replicates where the metric is undefined"}
        else:
            low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
            result[metric] = {"low": float(low), "high": float(high), "valid_replicates": len(values)}
    return result
