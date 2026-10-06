"""Interpretable Decision Tree Clinical Classifier for MediHaven.

Provides fully transparent clinical risk tier classification with step-by-step
human-readable if-then decision paths, satisfying the explainable AI mandate.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.data.feature_engineering import FEATURE_COLUMNS, TIER_MAP, REVERSE_TIER_MAP
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("medihaven.models.decision_tree")

# Clinical display units for human-readable rule formatting
FEATURE_UNITS: Dict[str, str] = {
    "heart_rate_latest": "bpm",
    "heart_rate_mean": "bpm",
    "heart_rate_slope": "bpm/hr",
    "sbp_latest": "mmHg",
    "sbp_mean": "mmHg",
    "sbp_slope": "mmHg/hr",
    "dbp_latest": "mmHg",
    "temperature_latest": "°C",
    "temp_max": "°C",
    "oxygen_saturation_latest": "%",
    "spo2_mean": "%",
    "spo2_min": "%",
    "spo2_slope": "%/hr",
    "respiratory_rate_latest": "breaths/min",
    "mean_arterial_pressure": "mmHg",
    "pulse_pressure": "mmHg",
    "shock_index": "",
    "modified_shock_index": "",
    "glucose_latest": "mg/dL",
    "glucose_delta": "mg/dL",
    "creatinine_latest": "mg/dL",
    "creatinine_delta": "mg/dL",
    "wbc_latest": "x10^3/uL",
    "bun_latest": "mg/dL",
    "bun_creatinine_ratio": "",
    "hemoglobin_latest": "g/dL",
    "platelets_latest": "x10^3/uL",
    "age": "years",
    "active_medications_count": "meds",
}


class DecisionTreeRiskClassifier:
    """Interpretable Decision Tree classifier with human-readable path extraction."""

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
    ):
        """Initializes the decision tree classifier.

        Args:
            max_depth: Maximum tree depth to constrain rule length (default: 5).
            min_samples_leaf: Minimum samples required at leaf node.
            random_state: Random state for deterministic splits.
            feature_names: Ordered list of feature names.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.feature_names = feature_names or list(FEATURE_COLUMNS)

        self.tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion="gini",
            random_state=random_state,
        )
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRiskClassifier":
        """Fits the decision tree classifier on training data.

        Args:
            X: Feature matrix (scaled or unscaled).
            y: Integer-encoded target risk tiers (0: Low, 1: Medium, 2: High, 3: Critical).

        Returns:
            Fitted instance.
        """
        logger.info(f"Fitting Decision Tree classifier (max_depth={self.max_depth})...")
        self.tree.fit(X, y)
        self.is_fitted = True
        logger.info(f"Decision Tree fitted with {self.tree.get_n_leaves()} leaves.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class tier codes for input samples."""
        if not self.is_fitted:
            raise RuntimeError("Decision tree must be fitted before predict()")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.tree.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class tier probability distributions."""
        if not self.is_fitted:
            raise RuntimeError("Decision tree must be fitted before predict_proba()")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.tree.predict_proba(X)

    def extract_rule_path(
        self,
        sample_vector: np.ndarray,
        raw_features: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """Traverses the tree to construct human-readable clinical explanation rules.

        Args:
            sample_vector: 1D feature array evaluated by the tree.
            raw_features: Optional dictionary of unscaled original clinical values for display.

        Returns:
            List of human-readable decision rule strings.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to extract decision path.")

        vec = sample_vector.flatten()
        tree_ = self.tree.tree_

        # Get node trajectory for this sample
        node_indicator = self.tree.decision_path(vec.reshape(1, -1))
        node_indices = node_indicator.indices

        rule_path: List[str] = []

        for node_id in node_indices:
            # If not a leaf node
            if tree_.children_left[node_id] != tree_.children_right[node_id]:
                feature_idx = tree_.feature[node_id]
                feature_name = self.feature_names[feature_idx]
                threshold = tree_.threshold[node_id]

                # Unit and raw value
                unit = FEATURE_UNITS.get(feature_name, "")
                unit_str = f" {unit}".strip() if unit else ""

                if raw_features and feature_name in raw_features:
                    obs_val = raw_features[feature_name]
                    obs_str = f" [Observed: {obs_val:.2f}{unit_str}]"
                else:
                    obs_str = f" [Val: {vec[feature_idx]:.2f}]"

                # Condition operator
                clean_name = feature_name.replace("_", " ").title()
                if vec[feature_idx] <= threshold:
                    rule = f"{clean_name} <= {threshold:.2f}{unit_str}{obs_str}"
                else:
                    rule = f"{clean_name} > {threshold:.2f}{unit_str}{obs_str}"

                rule_path.append(rule)
            else:
                # Leaf node: determine predicted class
                leaf_values = tree_.value[node_id][0]
                pred_class = int(np.argmax(leaf_values))
                tier_name = REVERSE_TIER_MAP.get(pred_class, "Medium")
                rule_path.append(f"Classification Outcome: {tier_name.upper()} RISK TIER")

        return rule_path

    def save(self, path: Optional[Path] = None) -> None:
        """Serializes Decision Tree model and feature metadata."""
        target_path = path or Config.DECISION_TREE_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "tree": self.tree,
                "feature_names": self.feature_names,
                "max_depth": self.max_depth,
            },
            target_path,
        )
        logger.info(f"Saved Decision Tree model to: {target_path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DecisionTreeRiskClassifier":
        """Loads serialized Decision Tree model from disk."""
        target_path = path or Config.DECISION_TREE_MODEL_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at: {target_path}")

        data = joblib.load(target_path)
        instance = cls(
            max_depth=data.get("max_depth", 5),
            feature_names=data.get("feature_names", list(FEATURE_COLUMNS)),
        )
        instance.tree = data["tree"]
        instance.is_fitted = True
        logger.info(f"Loaded Decision Tree model from: {target_path}")
        return instance


if __name__ == "__main__":
    from src.data.scaler import ClinicalFeatureScaler

    train_df = pd.read_parquet(Config.PROCESSED_DATA_DIR / "train.parquet")
    scaler = ClinicalFeatureScaler.load()
    X_train_scaled = scaler.transform(train_df)
    y_train = train_df["risk_tier_encoded"].values

    dt_model = DecisionTreeRiskClassifier(max_depth=5)
    dt_model.fit(X_train_scaled, y_train)
    dt_model.save()
    print("Decision Tree training and serialization completed successfully.")
