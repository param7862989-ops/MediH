"""K-Means Clinical Risk Stratification Model for MediHaven.

Provides unsupervised population risk tiering by clustering patients in normalized
physiological feature space. Maps cluster centroids to clinical severity tiers
(Low, Medium, High, Critical) and computes cluster proximity profiles.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.data.feature_engineering import FEATURE_COLUMNS, TIER_MAP, REVERSE_TIER_MAP
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("medihaven.models.kmeans")


class KMeansRiskClustering:
    """Unsupervised patient cohort clustering into 4 clinical risk tiers."""

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
    ):
        """Initializes K-Means model.

        Args:
            n_clusters: Number of risk clusters (default: 4 for Low/Med/High/Critical).
            random_state: Random state for deterministic centroid initialization.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=random_state,
        )
        self.cluster_to_tier_map: Dict[int, str] = {}
        self.cluster_to_tier_code: Dict[int, int] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        X_scaled: np.ndarray,
        y_tiers: Optional[np.ndarray] = None,
    ) -> "KMeansRiskClustering":
        """Fits KMeans clusters and aligns clusters with clinical severity tiers.

        Args:
            X_scaled: Normalized feature matrix (N x D).
            y_tiers: Optional ground truth risk tier strings or codes for cluster labeling.

        Returns:
            Fitted KMeansRiskClustering instance.
        """
        logger.info(f"Fitting K-Means model with K={self.n_clusters} clusters...")
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # Map each cluster to a clinical tier
        if y_tiers is not None and len(y_tiers) == len(X_scaled):
            # Map based on majority label in each cluster
            for c_idx in range(self.n_clusters):
                mask = cluster_labels == c_idx
                if np.sum(mask) > 0:
                    cohort_labels = y_tiers[mask]
                    # Find mode
                    unique_labels, counts = np.unique(cohort_labels, return_counts=True)
                    dominant = unique_labels[np.argmax(counts)]
                    dominant_str = (
                        dominant if isinstance(dominant, str) else REVERSE_TIER_MAP.get(int(dominant), "Medium")
                    )
                else:
                    dominant_str = REVERSE_TIER_MAP.get(c_idx, "Medium")

                self.cluster_to_tier_map[c_idx] = dominant_str
                self.cluster_to_tier_code[c_idx] = TIER_MAP.get(dominant_str, c_idx)
        else:
            # Sort clusters by distance from origin / mean centroid magnitude
            centroid_norms = np.linalg.norm(self.kmeans.cluster_centers_, axis=1)
            sorted_indices = np.argsort(centroid_norms)
            tier_names = ["Low", "Medium", "High", "Critical"]
            for rank, c_idx in enumerate(sorted_indices):
                t_name = tier_names[min(rank, len(tier_names) - 1)]
                self.cluster_to_tier_map[c_idx] = t_name
                self.cluster_to_tier_code[c_idx] = TIER_MAP[t_name]

        self.is_fitted = True
        logger.info(f"K-Means fitted successfully. Cluster to Tier mapping: {self.cluster_to_tier_map}")
        return self

    def predict_cluster(self, X_scaled: np.ndarray) -> np.ndarray:
        """Assigns samples to cluster IDs.

        Args:
            X_scaled: Normalized feature array.

        Returns:
            1D array of cluster indices.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted or loaded before prediction.")
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)
        return self.kmeans.predict(X_scaled)

    def predict_risk_tier(self, X_scaled: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """Maps samples to risk tier names and numerical codes.

        Args:
            X_scaled: Normalized feature array.

        Returns:
            Tuple of (list_of_tier_strings, array_of_tier_codes).
        """
        clusters = self.predict_cluster(X_scaled)
        tier_names = [self.cluster_to_tier_map.get(c, "Medium") for c in clusters]
        tier_codes = np.array([self.cluster_to_tier_code.get(c, 1) for c in clusters])
        return tier_names, tier_codes

    def get_cluster_distances(self, X_scaled: np.ndarray) -> np.ndarray:
        """Computes Euclidean distance from each sample to all 4 cluster centroids."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted or loaded.")
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)
        return self.kmeans.transform(X_scaled)

    def save(self, path: Optional[Path] = None) -> None:
        """Serializes K-Means model to disk."""
        target_path = path or Config.KMEANS_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kmeans": self.kmeans,
                "cluster_to_tier_map": self.cluster_to_tier_map,
                "cluster_to_tier_code": self.cluster_to_tier_code,
                "n_clusters": self.n_clusters,
            },
            target_path,
        )
        logger.info(f"Saved K-Means model to: {target_path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "KMeansRiskClustering":
        """Loads serialized K-Means model from disk."""
        target_path = path or Config.KMEANS_MODEL_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at: {target_path}")

        data = joblib.load(target_path)
        instance = cls(n_clusters=data.get("n_clusters", 4))
        instance.kmeans = data["kmeans"]
        instance.cluster_to_tier_map = data["cluster_to_tier_map"]
        instance.cluster_to_tier_code = data["cluster_to_tier_code"]
        instance.is_fitted = True
        logger.info(f"Loaded K-Means model from: {target_path}")
        return instance


if __name__ == "__main__":
    from src.data.scaler import ClinicalFeatureScaler

    train_df = pd.read_parquet(Config.PROCESSED_DATA_DIR / "train.parquet")
    scaler = ClinicalFeatureScaler.load()
    X_train_scaled = scaler.transform(train_df)
    y_tiers = train_df["risk_tier"].values

    model = KMeansRiskClustering(n_clusters=4)
    model.fit(X_train_scaled, y_tiers)
    model.save()
    print("K-Means training and serialization completed successfully.")
