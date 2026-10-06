"""Production Clinical Feature Scaler and Metadata Serializer for MediHaven.

Manages feature normalization (StandardScaler / RobustScaler) with strict train/test
isolation, preventing data leakage. Serializes trained scaler weights and feature column
manifests to models_store/ for consistent real-time REST API inference.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.data.feature_engineering import FEATURE_COLUMNS
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("medihaven.data.scaler")


class ClinicalFeatureScaler:
    """Production wrapper for feature scaling and inference serialization."""

    def __init__(
        self,
        scaler_type: str = "standard",
        feature_columns: Optional[List[str]] = None,
    ):
        """Initializes the feature scaler.

        Args:
            scaler_type: 'standard' for StandardScaler, 'robust' for RobustScaler.
            feature_columns: List of feature column names in exact expected order.
        """
        self.scaler_type = scaler_type.lower()
        self.feature_columns = feature_columns or list(FEATURE_COLUMNS)

        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> "ClinicalFeatureScaler":
        """Fits the scaler strictly on the training feature DataFrame.

        Args:
            X: Training DataFrame containing all self.feature_columns.

        Returns:
            Fitted scaler instance.
        """
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing}")

        X_subset = X[self.feature_columns].values
        self.scaler.fit(X_subset)
        self.is_fitted = True
        logger.info(
            f"Fitted {self.scaler_type} scaler on {X_subset.shape[0]} samples with "
            f"{len(self.feature_columns)} features."
        )
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray, Dict[str, float]],
    ) -> np.ndarray:
        """Transforms input features into scaled numerical vectors.

        Args:
            X: Input DataFrame, 2D numpy array, or single feature dictionary.

        Returns:
            Normalized 2D numpy array.
        """
        if not self.is_fitted:
            raise RuntimeError("ClinicalFeatureScaler must be fitted or loaded before calling transform()")

        if isinstance(X, dict):
            # Extract features in strict order
            row = [float(X.get(col, 0.0)) for col in self.feature_columns]
            matrix = np.array([row], dtype=np.float64)
        elif isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_columns if c not in X.columns]
            if missing:
                raise ValueError(f"Input DataFrame is missing required features: {missing}")
            matrix = X[self.feature_columns].values.astype(np.float64)
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                matrix = X.reshape(1, -1).astype(np.float64)
            else:
                matrix = X.astype(np.float64)
        else:
            raise TypeError(f"Unsupported data type for transform: {type(X)}")

        return self.scaler.transform(matrix)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fits scaler on DataFrame and returns transformed matrix."""
        return self.fit(X).transform(X)

    def save(
        self,
        scaler_path: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ) -> None:
        """Serializes scaler instance and feature column manifest.

        Args:
            scaler_path: Target path for .joblib scaler file.
            manifest_path: Target path for feature_columns.json file.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted ClinicalFeatureScaler")

        target_scaler = scaler_path or Config.SCALER_PATH
        target_manifest = manifest_path or (Config.MODELS_STORE_DIR / "feature_columns.json")

        target_scaler.parent.mkdir(parents=True, exist_ok=True)
        target_manifest.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scaler, target_scaler)
        with open(target_manifest, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_columns": self.feature_columns,
                    "scaler_type": self.scaler_type,
                    "num_features": len(self.feature_columns),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved feature scaler to: {target_scaler}")
        logger.info(f"Saved feature manifest to: {target_manifest}")

    @classmethod
    def load(
        cls,
        scaler_path: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ) -> "ClinicalFeatureScaler":
        """Loads a pre-fitted scaler and column manifest from disk.

        Args:
            scaler_path: Path to serialized scaler file.
            manifest_path: Path to feature_columns.json file.

        Returns:
            Fitted ClinicalFeatureScaler instance.
        """
        target_scaler = scaler_path or Config.SCALER_PATH
        target_manifest = manifest_path or (Config.MODELS_STORE_DIR / "feature_columns.json")

        if not target_scaler.exists():
            raise FileNotFoundError(f"Scaler file not found at: {target_scaler}")

        feature_cols = list(FEATURE_COLUMNS)
        scaler_type = "standard"

        if target_manifest.exists():
            with open(target_manifest, "r", encoding="utf-8") as f:
                meta = json.load(f)
                feature_cols = meta.get("feature_columns", feature_cols)
                scaler_type = meta.get("scaler_type", scaler_type)

        instance = cls(scaler_type=scaler_type, feature_columns=feature_cols)
        instance.scaler = joblib.load(target_scaler)
        instance.is_fitted = True
        logger.info(f"Loaded ClinicalFeatureScaler successfully from: {target_scaler}")
        return instance
