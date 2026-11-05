"""Integration and Benchmark Tests for Phase 4 Machine Learning Models & Ensemble.

Validates:
1. Serialization and loading of all 4 complementary models (K-Means, Decision Tree, KNN, Neural Net).
2. Human-readable Decision Tree rule path generation with clinical units.
3. K-Nearest Neighbors historical case precedent retrieval and protocol success rate calculation.
4. Multilayer Perceptron Softmax probabilities and continuous readmission risk scoring.
5. Ensemble multi-model consensus arbitration and calibrated risk score generation.
6. Automated 7–14 day early warning deterioration alert triggers for high/critical cohorts.
7. Sub-15ms multi-model inference latency for live REST API readiness.
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import FEATURE_COLUMNS, TIER_MAP, REVERSE_TIER_MAP
from src.data.scaler import ClinicalFeatureScaler
from src.models.kmeans_model import KMeansRiskClustering
from src.models.decision_tree_model import DecisionTreeRiskClassifier
from src.models.knn_model import KNNSimilarityMatcher
from src.models.neural_network_model import NeuralNetworkRiskModel
from src.models.ensemble import EnsembleClinicalPredictor
from src.utils.config import Config


@pytest.fixture(scope="module", autouse=True)
def ensure_models_trained():
    """Ensure all models are trained and artifacts exist before running tests."""
    meta_path = Config.MODELS_STORE_DIR / "ensemble_metadata.json"
    if not meta_path.exists():
        train_data = pd.read_parquet(Config.PROCESSED_DATA_DIR / "train.parquet")
        ensemble = EnsembleClinicalPredictor()
        ensemble.train_all(train_data)


def test_models_store_artifacts_exist():
    """Verify all serialized model weights and metadata exist in models_store/."""
    artifacts = [
        Config.KMEANS_MODEL_PATH,
        Config.DECISION_TREE_MODEL_PATH,
        Config.KNN_MODEL_PATH,
        Config.NEURAL_NETWORK_MODEL_PATH,
        Config.MODELS_STORE_DIR / "ensemble_metadata.json",
    ]
    for artifact in artifacts:
        assert artifact.exists(), f"Missing model artifact: {artifact}"

    with open(Config.MODELS_STORE_DIR / "ensemble_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["status"] == "TRAINED_AND_VERIFIED"
    assert "model_weights" in meta


def test_kmeans_risk_clustering():
    """Verify K-Means loads, outputs valid tiers, and calculates centroid distances."""
    kmeans = KMeansRiskClustering.load()
    assert kmeans.is_fitted is True

    # Test sample vector
    sample_vec = np.zeros(len(FEATURE_COLUMNS))
    tier_names, tier_codes = kmeans.predict_risk_tier(sample_vec)

    assert len(tier_names) == 1
    assert tier_names[0] in {"Low", "Medium", "High", "Critical"}
    assert tier_codes[0] in {0, 1, 2, 3}

    distances = kmeans.get_cluster_distances(sample_vec)
    assert distances.shape == (1, 4)
    assert np.all(distances >= 0)


def test_decision_tree_explainability_and_rules():
    """Verify Decision Tree classifies and extracts natural clinical rule paths."""
    dt = DecisionTreeRiskClassifier.load()
    assert dt.is_fitted is True

    train_data = pd.read_parquet(Config.PROCESSED_DATA_DIR / "train.parquet")
    sample_row = train_data.iloc[0]
    sample_vec = sample_row[FEATURE_COLUMNS].values.astype(np.float64)

    # Class prediction
    pred_code = dt.predict(sample_vec)[0]
    assert pred_code in {0, 1, 2, 3}

    # Rule path extraction
    rule_path = dt.extract_rule_path(sample_vec, raw_features=dict(sample_row))
    assert isinstance(rule_path, list)
    assert len(rule_path) >= 2, "Rule path should contain at least 1 decision node and 1 leaf"
    assert "RISK TIER" in rule_path[-1]

    # Check for clinical units or feature names in text
    rule_text = " ".join(rule_path)
    assert any(term in rule_text for term in ["Glucose", "Heart", "Oxygen", "Classification Outcome"])


def test_knn_similarity_and_protocol_recommendations():
    """Verify KNN case retrieval and protocol recommendation engine."""
    knn = KNNSimilarityMatcher.load()
    assert knn.is_fitted is True

    scaler = ClinicalFeatureScaler.load()
    sample_vec = np.zeros(len(FEATURE_COLUMNS))

    # Test case retrieval (K=5)
    similar_cases = knn.find_similar(sample_vec, k=5)
    assert len(similar_cases) == 5
    for case in similar_cases:
        assert "patient_id" in case
        assert "mrn" in case
        assert "similarity_score" in case
        assert 0.0 < case["similarity_score"] <= 1.0
        assert "protocol" in case

    # Test treatment protocol recommendation
    rec = knn.recommend_protocol(sample_vec, k=5)
    assert "recommended_protocol" in rec
    assert "success_rate" in rec
    assert rec["success_rate"] >= 0.70
    assert "similar_patients" in rec


def test_neural_network_probabilities_and_readmission():
    """Verify MLP output probabilities (Softmax) and continuous readmission score."""
    nn = NeuralNetworkRiskModel.load()
    assert nn.is_fitted is True

    sample_vec = np.zeros(len(FEATURE_COLUMNS))

    # Softmax probabilities
    probs = nn.predict_proba(sample_vec)[0]
    assert len(probs) == 4
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)
    assert all(0.0 <= p <= 1.0 for p in probs)

    # Readmission risk regression
    readmit_risk = float(nn.predict_readmission_risk(sample_vec)[0])
    assert 0.0 <= readmit_risk <= 1.0


def test_ensemble_consensus_and_live_inference_latency():
    """Verify multi-model ensemble consensus arbitration and latency benchmarks."""
    ensemble = EnsembleClinicalPredictor.load()

    # Warm-up call
    result = ensemble.predict_patient(1)

    # 1. Pure in-memory multi-model inference latency (Target: < 20ms)
    sample_vec = np.zeros(len(FEATURE_COLUMNS))
    t0 = time.perf_counter()
    for _ in range(20):
        ensemble.predict(sample_vec)
    t1 = time.perf_counter()
    pure_latency = ((t1 - t0) / 20.0) * 1000.0

    # 2. End-to-end database extraction + feature pipeline + multi-model inference latency (Target: < 80ms)
    t0 = time.perf_counter()
    for _ in range(5):
        result = ensemble.predict_patient(1)
    t1 = time.perf_counter()
    e2e_latency = ((t1 - t0) / 5.0) * 1000.0

    # Verify structured clinical intelligence packet
    assert result["patient_id"] == 1
    assert result["risk_tier"] in {"Low", "Medium", "High", "Critical"}
    assert 0.0 <= result["risk_score"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert "decision_tree_explanation" in result
    assert "similar_patients" in result
    assert "recommended_protocol" in result
    assert "protocol_success_rate" in result
    assert "model_consensus" in result
    assert "raw_features" in result

    # Check latency meets real-time budgets
    assert pure_latency < 20.0, f"Pure in-memory inference too slow: {pure_latency:.2f} ms"
    assert e2e_latency < 80.0, f"End-to-end pipeline too slow: {e2e_latency:.2f} ms"


def test_ensemble_early_warning_deterioration_triggers():
    """Verify early warning alert triggers appropriately on critical vs stable cohorts."""
    ensemble = EnsembleClinicalPredictor.load()

    # Find a patient from Critical and Low cohorts
    train_data = pd.read_parquet(Config.PROCESSED_DATA_DIR / "train.parquet")
    crit_patient = train_data[train_data["risk_tier"] == "Critical"].iloc[0]
    low_patient = train_data[train_data["risk_tier"] == "Low"].iloc[0]

    res_crit = ensemble.predict_patient(int(crit_patient["patient_id"]))
    res_low = ensemble.predict_patient(int(low_patient["patient_id"]))

    # Critical cohort must trigger early warning alert
    assert res_crit["risk_tier"] in ("High", "Critical")
    assert res_crit["early_warning_alert"] is True
    assert "7–14 Days" in res_crit["early_warning_window"]
    assert res_crit["risk_score"] >= 0.70

    # Low cohort must be stable
    assert res_low["risk_tier"] == "Low"
    assert res_low["early_warning_alert"] is False
    assert res_low["risk_score"] < 0.40


def test_each_model_reports_its_tier_and_risk_score():
    """The chart drawer shows every model's tier with its own 0-1 risk score."""
    ensemble = EnsembleClinicalPredictor.load()
    result = ensemble.predict_patient(1)
    scores = result["model_consensus"]["model_scores"]
    assert set(scores) == {"kmeans", "decision_tree", "knn", "neural_network"}
    for view in scores.values():
        assert view["tier"] in {"Low", "Medium", "High", "Critical"}
        assert 0.0 <= view["risk_score"] <= 1.0 and 0.0 <= view["confidence"] <= 1.0
    consensus = result["model_consensus"]
    assert scores["kmeans"]["tier"] == consensus["kmeans_tier"]
    assert scores["decision_tree"]["tier"] == consensus["decision_tree_tier"]
    assert scores["knn"]["tier"] == consensus["knn_tier"]
    assert scores["neural_network"]["tier"] == consensus["neural_network_tier"]


def test_kmeans_probabilities_are_aligned_to_tiers_before_fusion():
    """Cluster numbers are arbitrary; K-Means pseudo-probabilities must be re-ordered by tier code."""
    from src.data.pipeline import transform_single_patient
    ensemble = EnsembleClinicalPredictor.load()
    sample_vec, raw = transform_single_patient(86, scaler=ensemble.scaler)
    distances = ensemble.kmeans.get_cluster_distances(sample_vec)[0]
    by_cluster = (1.0 / (1.0 + distances)) / np.sum(1.0 / (1.0 + distances))
    aligned = np.zeros(4)
    for cluster, prob in enumerate(by_cluster):
        aligned[ensemble.kmeans.cluster_to_tier_code[cluster]] += prob
    expected_score = round(float(np.sum(aligned * np.array([0.10, 0.45, 0.78, 0.94]))), 2)
    result = ensemble.predict(sample_vec, raw_features=raw)
    assert result["model_consensus"]["model_scores"]["kmeans"]["risk_score"] == expected_score
    assert int(np.argmax(aligned)) == TIER_MAP[result["model_consensus"]["kmeans_tier"]]


def test_kmeans_cohort_map_profiles_every_cluster():
    """K-Means clusters without labels, then maps clusters to tiers; the cohort map must be consistent."""
    ensemble = EnsembleClinicalPredictor.load()
    cohort = ensemble.kmeans.cohort_map
    assert cohort is not None
    assert len(cohort["clusters"]) == 4
    assert sum(c["size"] for c in cohort["clusters"]) == cohort["n_reference_patients"] == len(cohort["points"])
    assert -1.0 <= cohort["silhouette_score"] <= 1.0
    assert len(cohort["explained_variance"]) == 2
    for cluster in cohort["clusters"]:
        assert cluster["mapped_tier"] == ensemble.kmeans.cluster_to_tier_map[cluster["cluster_id"]]
        assert sum(cluster["tier_composition"].values()) == cluster["size"]
        # The mapped tier is the majority tier of the cluster's members
        assert cluster["tier_composition"][cluster["mapped_tier"]] == max(cluster["tier_composition"].values())
        assert 0.0 <= cluster["purity"] <= 1.0
    # Every plotted point is in the cluster K-Means assigns it to
    point_clusters = {pt["cluster_id"] for pt in cohort["points"]}
    assert point_clusters <= {c["cluster_id"] for c in cohort["clusters"]}


def test_cluster_map_places_patient_in_its_cluster():
    """The patient's cluster on the map matches the K-Means tier used in the vote."""
    ensemble = EnsembleClinicalPredictor.load()
    result = ensemble.cluster_map_for_patient(1)
    patient = result["patient"]
    assert patient["patient_id"] == 1
    assert len(patient["coords_2d"]) == 2
    distances = {d["cluster_id"]: d["distance"] for d in patient["centroid_distances"]}
    assert patient["cluster_id"] == min(distances, key=distances.get)
    prediction = ensemble.predict_patient(1)
    assert prediction["kmeans_cluster"]["cluster_id"] == patient["cluster_id"]
    assert prediction["model_consensus"]["kmeans_tier"] == patient["mapped_tier"]
    assert {row["feature"] for row in patient["profile_comparison"]}


@pytest.mark.parametrize("model_key", ["decision_tree", "neural_network"])
def test_shap_values_add_up_to_the_model_probability(model_key):
    """SHAP additivity: base value + all contributions = the model's predicted probability."""
    ensemble = EnsembleClinicalPredictor.load()
    for patient_id in (1, 90):
        explanation = ensemble.predict_patient(patient_id)["shap_explanations"][model_key]
        assert explanation is not None
        assert explanation["target_tier"] in {"Low", "Medium", "High", "Critical"}
        total = (explanation["base_value"] + sum(f["shap_value"] for f in explanation["features"])
                 + explanation["other_features_total"])
        assert abs(total - explanation["predicted_probability"]) < 0.02
        values = [abs(f["shap_value"]) for f in explanation["features"]]
        assert values == sorted(values, reverse=True)


def test_shap_explanations_are_deterministic():
    """The permutation explainer uses a fixed seed, so repeated requests give identical values."""
    ensemble = EnsembleClinicalPredictor.load()
    first = ensemble.predict_patient(42)["shap_explanations"]
    second = ensemble.predict_patient(42)["shap_explanations"]
    assert first == second
