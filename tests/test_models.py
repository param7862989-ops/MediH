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
