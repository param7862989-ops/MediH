"""Subject-grouped, stratified cross-validation folds with explicit safety checks.

All stays of a subject fall into the same fold, so no subject is ever in both the training and
the test part of a split. The splitter refuses to run, rather than degrade silently, when there
are too few positive or negative cases for every test fold to contain both classes.
"""

import hashlib
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

Fold = Tuple[np.ndarray, np.ndarray]


class InsufficientDataError(ValueError):
    """Raised when the data cannot support the requested grouped, stratified evaluation."""


def make_grouped_folds(y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int) -> List[Fold]:
    y, groups = np.asarray(y), np.asarray(groups)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_folds < 2:
        raise InsufficientDataError("At least 2 folds are required.")
    if n_pos < n_folds or n_neg < n_folds:
        raise InsufficientDataError(
            f"Too few cases for {n_folds}-fold evaluation: {n_pos} positive and {n_neg} negative stays "
            f"(each test fold needs at least one of each). Use fewer folds or more data; "
            f"a non-grouped or non-stratified split will not be used instead."
        )
    if len(np.unique(groups)) < n_folds:
        raise InsufficientDataError(f"Only {len(np.unique(groups))} subjects for {n_folds} folds.")

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for k, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise AssertionError("Subject leakage between training and test folds")  # should be impossible
        for part, idx in (("test", test_idx), ("training", train_idx)):
            if len(np.unique(y[idx])) < 2:
                raise InsufficientDataError(
                    f"Fold {k + 1} {part} part contains a single class with {n_pos} positive stays overall. "
                    f"Use fewer folds; the evaluation will not continue with single-class folds."
                )
        folds.append((train_idx, test_idx))
    return folds


def fold_summary(folds: List[Fold], y: np.ndarray, groups: np.ndarray) -> List[Dict[str, int]]:
    """Aggregate per-fold counts (no identifiers)."""
    y, groups = np.asarray(y), np.asarray(groups)
    summary = []
    for k, (train_idx, test_idx) in enumerate(folds):
        summary.append({
            "fold": k + 1,
            "train_stays": int(len(train_idx)),
            "train_subjects": int(len(np.unique(groups[train_idx]))),
            "train_positives": int(y[train_idx].sum()),
            "test_stays": int(len(test_idx)),
            "test_subjects": int(len(np.unique(groups[test_idx]))),
            "test_positives": int(y[test_idx].sum()),
            "test_negatives": int(len(test_idx) - y[test_idx].sum()),
        })
    return summary


def assignment_fingerprint(folds: List[Fold], groups: np.ndarray) -> str:
    """SHA-256 over (subject, fold) pairs: verifies split reproducibility without storing identifiers."""
    groups = np.asarray(groups)
    pairs = sorted({(str(groups[i]), k) for k, (_, test_idx) in enumerate(folds) for i in test_idx})
    return hashlib.sha256(";".join(f"{g}:{k}" for g, k in pairs).encode("utf-8")).hexdigest()
