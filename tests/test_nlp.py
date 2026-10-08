"""Unit and Integration Tests for Natural Language Processing & Gemini Generative AI (Phase 7).

Validates:
1. ClinicalNarrativeSummarizer configuration and prompt construction.
2. SBAR physician handover narrative generation.
3. Plain-language patient discharge summary generation.
4. Resilient deterministic fallback synthesis.
5. REST endpoint integration (/api/patients/<id>/clinical-narrative and /api/admin/gemini/synthesize).
"""

import pytest
from flask.testing import FlaskClient

from src.api.app import create_app
from src.nlp.summarizer import ClinicalNarrativeSummarizer


@pytest.fixture(scope="module")
def app():
    """Create Flask application in test mode."""
    return create_app({"TESTING": True})


@pytest.fixture(scope="module")
def client(app):
    """Create test client."""
    return app.test_client()


def test_summarizer_deterministic_fallback():
    """Verifies that summarizer generates high-quality clinical text even without external API."""
    offline_summarizer = ClinicalNarrativeSummarizer(api_key="")
    assert offline_summarizer.is_configured() is False

    patient = {
        "full_name": "Test Patient",
        "age": 62,
        "ward": "Cardiology Stepdown",
    }
    prediction = {
        "risk_tier": "High",
        "risk_score": 0.78,
        "recommended_protocol": "Protocol B: Cardiac Telemetry Escalation",
        "readmission_30d_risk": 0.42,
    }

    # Test Handover fallback
    res_handover = offline_summarizer.generate(patient, prediction, "handover")
    assert "Test Patient" in res_handover["narrative"]
    assert "HIGH" in res_handover["narrative"]
    assert res_handover["is_live"] is False
    assert "Protocol B" in res_handover["narrative"]

    # Test Discharge fallback
    res_discharge = offline_summarizer.generate(patient, prediction, "discharge")
    assert "Test Patient" in res_discharge["narrative"]
    assert res_discharge["is_live"] is False


def test_patient_clinical_narrative_endpoint(client: FlaskClient):
    """Verifies GET and POST to /api/patients/<id>/clinical-narrative."""
    # Test GET for Patient 1
    resp_get = client.get("/api/patients/1/clinical-narrative?type=handover")
    assert resp_get.status_code == 200
    data_get = resp_get.get_json()
    assert data_get["success"] is True
    assert "narrative" in data_get["data"]
    assert len(data_get["data"]["narrative"]) > 20
    assert data_get["data"]["patient_id"] == 1

    # Test POST for Discharge summary
    resp_post = client.post(
        "/api/patients/1/clinical-narrative",
        json={"type": "discharge"},
    )
    assert resp_post.status_code == 200
    data_post = resp_post.get_json()
    assert data_post["success"] is True
    assert data_post["data"]["type"] == "discharge"
    assert "narrative" in data_post["data"]


def test_patient_clinical_narrative_not_found(client: FlaskClient):
    """Verifies 404 handling for non-existent patient ID."""
    resp = client.get("/api/patients/999999/clinical-narrative")
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "PATIENT_NOT_FOUND"


def test_admin_gemini_synthesize_endpoint(client: FlaskClient):
    """Verifies /api/admin/gemini/synthesize test endpoint."""
    resp = client.post(
        "/api/admin/gemini/synthesize",
        json={"patient_id": 2, "type": "handover"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["data"]["patient_id"] == 2
    assert "narrative" in data["data"]
