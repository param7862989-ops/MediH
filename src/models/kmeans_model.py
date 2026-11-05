"""K-Means Patient Cohort Discovery for MediHaven.

K-Means is used for what it is good at: finding groups of physiologically similar patients
without using the risk labels. It runs in two clearly separated steps:

1. Unsupervised clustering: patients are grouped in the standardized 30-feature space
   (k-means++, K=4). The risk tiers are not used here.
2. Cluster-to-tier mapping: only after the clusters are found, each cluster is labelled with
   the most common risk tier among its members (majority vote). How well a cluster matches its
   tier is reported as its purity, so the mapping can be checked rather than assumed.

For the dashboard, each cluster is also profiled (size, tier mix, mean vital signs and labs,
outcome rates) and the whole reference cohort is projected onto two principal components so the
clusters, their centroids and the current patient can be drawn on one 2-D "cohort map".
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.data.feature_engineering import FEATURE_COLUMNS, FEATURE_LABELS, TIER_MAP, REVERSE_TIER_MAP
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("medihaven.models.kmeans")

TIER_ORDER = ["Low", "Medium", "High", "Critical"]

# Features summarised in each cluster's clinical profile (shown next to the cohort map)
PROFILE_FEATURES: List[str] = [
    "heart_rate_latest",
    "sbp_latest",
    "oxygen_saturation_latest",
    "respiratory_rate_latest",
    "temperature_latest",
    "shock_index",
    "mean_arterial_pressure",
    "creatinine_latest",
    "wbc_latest",
    "glucose_latest",
]


def _to_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(number) else number


class KMeansRiskClustering:
    """Unsupervised patient cohort clustering with a post-hoc, auditable cluster-to-tier mapping."""

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
    ):
        """Initializes K-Means model.

        Args:
            n_clusters: Number of patient clusters (default: 4).
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
        self.pca: Optional[PCA] = None
        self.cohort_map: Optional[Dict[str, Any]] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X_scaled: np.ndarray,
        y_tiers: Optional[np.ndarray] = None,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> "KMeansRiskClustering":
        """Clusters the patients, then maps each cluster to a tier and builds the cohort map.

        Args:
            X_scaled: Normalized feature matrix (N x D).
            y_tiers: Optional risk tiers, used only after clustering to name each cluster.
            reference_df: Optional rows matching X_scaled (raw features, patient_id, mrn, outcomes)
                used to build the cluster profiles and the 2-D cohort map.

        Returns:
            Fitted KMeansRiskClustering instance.
        """
        logger.info(f"Fitting K-Means model with K={self.n_clusters} clusters (labels not used)...")
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # Step 2: name each discovered cluster with its majority tier
        if y_tiers is not None and len(y_tiers) == len(X_scaled):
            for c_idx in range(self.n_clusters):
                mask = cluster_labels == c_idx
                if np.sum(mask) > 0:
                    unique_labels, counts = np.unique(y_tiers[mask], return_counts=True)
                    dominant = unique_labels[np.argmax(counts)]
                    dominant_str = (
                        dominant if isinstance(dominant, str) else REVERSE_TIER_MAP.get(int(dominant), "Medium")
                    )
                else:
                    dominant_str = REVERSE_TIER_MAP.get(c_idx, "Medium")
                self.cluster_to_tier_map[c_idx] = dominant_str
                self.cluster_to_tier_code[c_idx] = TIER_MAP.get(dominant_str, c_idx)
        else:
            # No labels: order clusters by centroid distance from the cohort mean
            centroid_norms = np.linalg.norm(self.kmeans.cluster_centers_, axis=1)
            for rank, c_idx in enumerate(np.argsort(centroid_norms)):
                t_name = TIER_ORDER[min(rank, len(TIER_ORDER) - 1)]
                self.cluster_to_tier_map[int(c_idx)] = t_name
                self.cluster_to_tier_code[int(c_idx)] = TIER_MAP[t_name]

        self.is_fitted = True
        if reference_df is not None and len(reference_df) == len(X_scaled):
            self.build_cohort_map(X_scaled, reference_df)
        logger.info(f"K-Means fitted. Cluster to tier mapping: {self.cluster_to_tier_map}")
        return self

    def build_cohort_map(self, X_scaled: np.ndarray, reference_df: pd.DataFrame) -> Dict[str, Any]:
        """Profiles every cluster and projects the reference cohort to 2-D for visualization."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before building the cohort map.")
        reference_df = reference_df.reset_index(drop=True)
        labels = self.kmeans.predict(X_scaled)

        self.pca = PCA(n_components=2, random_state=self.random_state)
        coords = self.pca.fit_transform(X_scaled)
        centroids_2d = self.pca.transform(self.kmeans.cluster_centers_)

        silhouette = None
        if 1 < len(np.unique(labels)) < len(X_scaled):
            silhouette = round(float(silhouette_score(X_scaled, labels)), 3)

        tiers = reference_df["risk_tier"].astype(str).values if "risk_tier" in reference_df else None

        clusters: List[Dict[str, Any]] = []
        for c_idx in range(self.n_clusters):
            mask = labels == c_idx
            members = reference_df[mask]
            size = int(mask.sum())
            composition = {t: 0 for t in TIER_ORDER}
            if tiers is not None:
                for tier in tiers[mask]:
                    composition[tier] = composition.get(tier, 0) + 1
            mapped = self.cluster_to_tier_map.get(c_idx, "Medium")
            purity = round(composition.get(mapped, 0) / size, 3) if size else 0.0

            profile = {}
            for feat in PROFILE_FEATURES:
                if feat in members and size:
                    profile[feat] = _to_float(round(float(members[feat].mean()), 2))
            outcome_rates = {}
            for outcome in ("readmitted_30d", "icu_transfer"):
                if outcome in members and size:
                    outcome_rates[outcome] = round(float(members[outcome].astype(float).mean()), 3)

            clusters.append({
                "cluster_id": c_idx,
                "mapped_tier": mapped,
                "size": size,
                "share": round(size / len(reference_df), 3) if len(reference_df) else 0.0,
                "tier_composition": composition,
                "purity": purity,
                "centroid_2d": [round(float(centroids_2d[c_idx, 0]), 4), round(float(centroids_2d[c_idx, 1]), 4)],
                "profile": profile,
                "outcome_rates": outcome_rates,
            })

        points = []
        for i in range(len(reference_df)):
            row = reference_df.iloc[i]
            points.append({
                "patient_id": int(row["patient_id"]) if "patient_id" in row else i + 1,
                "mrn": str(row.get("mrn", "")),
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "cluster_id": int(labels[i]),
                "risk_tier": str(tiers[i]) if tiers is not None else None,
            })

        overall_purity = None
        if tiers is not None:
            mapped_labels = np.array([self.cluster_to_tier_map.get(int(c), "Medium") for c in labels])
            overall_purity = round(float(np.mean(mapped_labels == tiers)), 3)

        self.cohort_map = {
            "n_clusters": self.n_clusters,
            "n_reference_patients": int(len(reference_df)),
            "silhouette_score": silhouette,
            "overall_purity": overall_purity,
            "explained_variance": [round(float(v), 3) for v in self.pca.explained_variance_ratio_],
            "profile_features": [
                {"feature": f, "label": FEATURE_LABELS.get(f, (f, ""))[0], "unit": FEATURE_LABELS.get(f, ("", ""))[1]}
                for f in PROFILE_FEATURES
            ],
            "clusters": clusters,
            "points": points,
        }
        return self.cohort_map

    def predict_cluster(self, X_scaled: np.ndarray) -> np.ndarray:
        """Assigns samples to cluster IDs."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted or loaded before prediction.")
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)
        return self.kmeans.predict(X_scaled)

    def predict_risk_tier(self, X_scaled: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """Maps samples to the tier of their cluster (names and numerical codes)."""
        clusters = self.predict_cluster(X_scaled)
        tier_names = [self.cluster_to_tier_map.get(int(c), "Medium") for c in clusters]
        tier_codes = np.array([self.cluster_to_tier_code.get(int(c), 1) for c in clusters])
        return tier_names, tier_codes

    def get_cluster_distances(self, X_scaled: np.ndarray) -> np.ndarray:
        """Computes Euclidean distance from each sample to all cluster centroids."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted or loaded.")
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)
        return self.kmeans.transform(X_scaled)

    def describe_patient(
        self,
        X_scaled: np.ndarray,
        raw_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Places one patient in the cohort: assigned cluster, distances, 2-D position, profile comparison."""
        vec = X_scaled.reshape(1, -1) if X_scaled.ndim == 1 else X_scaled[:1]
        cluster_id = int(self.predict_cluster(vec)[0])
        distances = self.get_cluster_distances(vec)[0]
        order = np.argsort(distances)
        nearest_other = int(order[1]) if len(order) > 1 else cluster_id

        result: Dict[str, Any] = {
            "cluster_id": cluster_id,
            "mapped_tier": self.cluster_to_tier_map.get(cluster_id, "Medium"),
            "distance_to_centroid": round(float(distances[cluster_id]), 3),
            "nearest_other_cluster": nearest_other,
            "nearest_other_tier": self.cluster_to_tier_map.get(nearest_other, "Medium"),
            "distance_to_nearest_other": round(float(distances[nearest_other]), 3),
            "centroid_distances": [
                {
                    "cluster_id": int(c),
                    "mapped_tier": self.cluster_to_tier_map.get(int(c), "Medium"),
                    "distance": round(float(distances[c]), 3),
                }
                for c in range(len(distances))
            ],
        }
        if self.pca is not None:
            xy = self.pca.transform(vec)[0]
            result["coords_2d"] = [round(float(xy[0]), 4), round(float(xy[1]), 4)]

        if self.cohort_map:
            cluster = next((c for c in self.cohort_map["clusters"] if c["cluster_id"] == cluster_id), None)
            if cluster is not None:
                result["cluster_size"] = cluster["size"]
                result["cluster_purity"] = cluster["purity"]
                if raw_features:
                    result["profile_comparison"] = [
                        {
                            "feature": f,
                            "label": FEATURE_LABELS.get(f, (f, ""))[0],
                            "unit": FEATURE_LABELS.get(f, ("", ""))[1],
                            "patient": _to_float(raw_features.get(f)),
                            "cluster_mean": cluster["profile"].get(f),
                        }
                        for f in PROFILE_FEATURES
                    ]
        return result

    def save(self, path: Optional[Path] = None) -> None:
        """Serializes K-Means model, its tier mapping and cohort map to disk."""
        target_path = path or Config.KMEANS_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kmeans": self.kmeans,
                "cluster_to_tier_map": self.cluster_to_tier_map,
                "cluster_to_tier_code": self.cluster_to_tier_code,
                "n_clusters": self.n_clusters,
                "pca": self.pca,
                "cohort_map": self.cohort_map,
            },
            target_path,
        )
        logger.info(f"Saved K-Means model to: {target_path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "KMeansRiskClustering":
        """Loads serialized K-Means model from disk (older artifacts without a cohort map still load)."""
        target_path = path or Config.KMEANS_MODEL_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at: {target_path}")

        data = joblib.load(target_path)
        instance = cls(n_clusters=data.get("n_clusters", 4))
        instance.kmeans = data["kmeans"]
        instance.cluster_to_tier_map = {int(k): v for k, v in data["cluster_to_tier_map"].items()}
        instance.cluster_to_tier_code = {int(k): int(v) for k, v in data["cluster_to_tier_code"].items()}
        instance.pca = data.get("pca")
        instance.cohort_map = data.get("cohort_map")
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
    model.fit(X_train_scaled, y_tiers, reference_df=train_df)
    model.save()
    print("K-Means training and serialization completed successfully.")
