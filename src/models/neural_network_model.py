"""Multilayer Perceptron (MLP) Neural Network for Complex Clinical Interactions.

Captures non-linear relationships across multimodal features (temporal vitals slope
interactions and cross-laboratory metabolic markers) for high-capacity risk scoring
and 30-day readmission risk estimation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor

from src.data.feature_engineering import FEATURE_COLUMNS, TIER_MAP, REVERSE_TIER_MAP
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("medihaven.models.neural_net")


class NeuralNetworkRiskModel:
    """Multilayer Perceptron neural network for risk tiering and readmission scoring."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        random_state: int = 42,
        max_iter: int = 500,
    ):
        """Initializes neural network classifiers and regressors.

        Args:
            hidden_layer_sizes: Tuple of layer neuron counts (default: (64, 32)).
            random_state: Random state for deterministic weight initialization.
            max_iter: Maximum epochs for backpropagation convergence.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.max_iter = max_iter

        # Multi-class risk tier classifier (Softmax)
        self.classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=16,
            learning_rate_init=0.005,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.15,
            random_state=random_state,
        )

        # 30-Day readmission risk continuous regressor
        self.readmit_regressor = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=max_iter,
            random_state=random_state,
        )

        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y_tier: np.ndarray,
        y_readmit: Optional[np.ndarray] = None,
    ) -> "NeuralNetworkRiskModel":
        """Fits neural network weights on normalized feature vectors.

        Args:
            X: Normalized feature matrix (N x D).
            y_tier: Integer encoded risk tiers (0 to 3).
            y_readmit: Continuous or binary readmission targets.

        Returns:
            Fitted instance.
        """
        logger.info(f"Training Multilayer Perceptron neural network {self.hidden_layer_sizes}...")
        self.classifier.fit(X, y_tier)

        if y_readmit is not None:
            self.readmit_regressor.fit(X, y_readmit)
        else:
            # Fallback synthetic target if missing
            self.readmit_regressor.fit(X, y_tier * 0.25)

        self.is_fitted = True
        logger.info("Neural network training completed successfully.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class tier codes for input samples."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict()")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts Softmax probability distribution across all 4 risk tiers."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba()")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.classifier.predict_proba(X)

    def predict_readmission_risk(self, X: np.ndarray) -> np.ndarray:
        """Predicts continuous 30-day readmission likelihood [0.0, 1.0]."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_readmission_risk()")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds = self.readmit_regressor.predict(X)
        # Clip to valid probability bounds [0.0, 1.0]
        return np.clip(preds, 0.0, 1.0)

    def save(self, path: Optional[Path] = None) -> None:
        """Serializes neural network models to disk."""
        target_path = path or Config.NEURAL_NETWORK_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": self.classifier,
                "readmit_regressor": self.readmit_regressor,
                "hidden_layer_sizes": self.hidden_layer_sizes,
            },
            target_path,
        )
        logger.info(f"Saved Neural Network model to: {target_path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "NeuralNetworkRiskModel":
        """Loads serialized neural network model from disk."""
        target_path = path or Config.NEURAL_NETWORK_MODEL_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at: {target_path}")

        data = joblib.load(target_path)
        instance = cls(hidden_layer_sizes=data.get("hidden_layer_sizes", (64, 32)))
        instance.classifier = data["classifier"]
        instance.readmit_regressor = data["readmit_regressor"]
        instance.is_fitted = True
        logger.info(f"Loaded Neural Network model from: {target_path}")
        return instance


if __name__ == "__main__":
    from src.data.scaler import ClinicalFeatureScaler

    train_df = pd.read_parquet(Config.PROCESSED_DATA_DIR / "train.parquet")
    scaler = ClinicalFeatureScaler.load()
    X_train_scaled = scaler.transform(train_df)
    y_tier = train_df["risk_tier_encoded"].values
    y_readmit = train_df.get("risk_score", train_df["risk_tier_encoded"] * 0.25).values

    nn_model = NeuralNetworkRiskModel()
    nn_model.fit(X_train_scaled, y_tier, y_readmit)
    nn_model.save()
    print("Neural Network training and serialization completed successfully.")
