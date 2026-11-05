"""SHAP feature attributions for the synthetic ensemble.

Two explainers are used, each matched to its model:
- Decision Tree: exact TreeSHAP (path-dependent), so the explanation is exact for the tree.
- Neural network (MLP): permutation SHAP against the reference cohort as background data. This
  is a sampling estimate (fixed seed, so it is reproducible), because no exact method exists
  for an MLP.

In both cases base value + sum of all contributions = the model's predicted probability for
the explained tier. The values describe how the model behaves; they are not clinical causes.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.data.feature_engineering import FEATURE_LABELS, REVERSE_TIER_MAP


def format_attribution(
    feature_names: Sequence[str],
    contributions: np.ndarray,
    display_values: np.ndarray,
    base_value: float,
    predicted_probability: float,
    target_code: int,
    method: str,
    model: str,
    top_n: int = 8,
) -> Dict[str, Any]:
    """Turns raw SHAP values into the JSON payload shown in the patient chart."""
    contributions = np.asarray(contributions, dtype=np.float64)
    order = np.argsort(-np.abs(contributions))
    nonzero = [i for i in order if abs(contributions[i]) > 1e-9]
    shown = nonzero[:top_n]

    features: List[Dict[str, Any]] = []
    for i in shown:
        name = feature_names[i]
        features.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, (name, ""))[0],
            "unit": FEATURE_LABELS.get(name, ("", ""))[1],
            "value": round(float(display_values[i]), 2),
            "shap_value": round(float(contributions[i]), 4),
        })
    rest = [i for i in order if i not in shown]
    return {
        "method": method,
        "model": model,
        "target_tier": REVERSE_TIER_MAP.get(int(target_code), "Medium"),
        "base_value": round(float(base_value), 4),
        "predicted_probability": round(float(predicted_probability), 4),
        "features": features,
        "other_features_total": round(float(np.sum(contributions[rest])) if rest else 0.0, 4),
        "n_nonzero_features": len(nonzero),
    }


def permutation_shap(
    predict_proba,
    x_scaled: np.ndarray,
    background_scaled: np.ndarray,
    class_index: int,
    max_evals: Optional[int] = None,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """Permutation SHAP values of one class probability for one sample (None if shap is missing)."""
    try:
        import shap  # imported lazily: only needed when an explanation is requested
    except ImportError:
        return None

    x = np.asarray(x_scaled, dtype=np.float64).reshape(1, -1)
    n_features = x.shape[1]
    masker = shap.maskers.Independent(np.asarray(background_scaled, dtype=np.float64), max_samples=100)
    explainer = shap.PermutationExplainer(lambda data: predict_proba(data)[:, class_index], masker, seed=seed)
    result = explainer(x, max_evals=max_evals or (4 * n_features + 1), silent=True)
    return {
        "values": np.asarray(result.values)[0],
        "base_value": float(np.atleast_1d(result.base_values)[0]),
    }
