"""Comprehensive Test Suite for MediHaven RESTful API Layer (Phase 6).

Validates:
1. System telemetry & health probe (/api/health).
2. Patient registry, triage listings, filters, and chart profiles (/api/patients).
3. Patient admission registration (/api/patients POST).
4. Longitudinal vitals time-series & laboratory biomarkers (/api/patients/<id>/vitals, /labs).
5. Live multi-model risk inference & prediction auditing (/api/patients/<id>/predict).
6. Precedent similarity search (/api/patients/<id>/similar).
7. Executive hospital dashboard metrics (/api/dashboard/summary).
8. Early deterioration warning alerts feed & clinical acknowledgement (/api/alerts).
9. Patient Medical Vault record creation, retrieval & filtering (/api/vault/upload, /api/vault/<id>).
10. Cryptographic QR pass generation, optical/string access, and 1-click revocation.
11. Dual-stream audit trail & active token telemetry (/api/vault/<id>/access-log).
12. Web portal views & centralized HTTP error handlers (/dashboard, /vault, /scanner, 404, 405).
"""

import json
import pytest
from flask import Flask
from flask.testing import FlaskClient

from src.api.app import create_app
from src.database.db import get_connection


@pytest.fixture(scope="module")
def app() -> Flask:
    """Creates a configured Flask application for testing."""
    test_app = create_app({"TESTING": True})
    return test_app


@pytest.fixture(scope="module")
def client(app: Flask) -> FlaskClient:
    """Provides a test client for simulating HTTP requests."""
    return app.test_client()


# ==============================================================================
# 1. Telemetry and Health Probe Tests
# ==============================================================================

def test_health_check_endpoint(client: FlaskClient):
    """Verify system health endpoint returns operational status and model telemetry."""
    response = client.get("/api/health")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    assert payload["data"]["status"] == "OPERATIONAL"
    assert "service" in payload["data"]
    assert "version" in payload["data"]
    assert isinstance(payload["data"]["models_online"], bool)
    assert "timestamp" in payload["meta"]


# ==============================================================================
# 2. Patient Roster & Triage Filtering Tests
# ==============================================================================

def test_list_patients_default(client: FlaskClient):
    """Verify listing all patients returns populated triage list."""
    response = client.get("/api/patients")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    assert isinstance(payload["data"], list)
    assert payload["meta"]["total"] >= 100

    patient = payload["data"][0]
    assert "patient_id" in patient
    assert "mrn" in patient
    assert "full_name" in patient
    assert "risk_tier" in patient
    assert "risk_score" in patient
    assert "vitals_snapshot" in patient
    assert "shock_index" in patient["vitals_snapshot"]
    assert "mean_arterial_pressure" in patient["vitals_snapshot"]


def test_list_patients_filtered_by_risk_tier(client: FlaskClient):
    """Verify filtering patients by risk tier (Critical, High, Medium, Low)."""
    response = client.get("/api/patients?risk_tier=High")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    for p in payload["data"]:
        assert p["risk_tier"] == "High"


def test_list_patients_search_by_name_and_mrn(client: FlaskClient):
    """Verify search filter by substring of patient name or MRN."""
    # First get a known patient name
    list_res = client.get("/api/patients")
    sample = list_res.get_json()["data"][0]
    sample_name = sample["full_name"].split()[0]  # First name

    search_res = client.get(f"/api/patients?search={sample_name}")
    assert search_res.status_code == 200
    search_payload = search_res.get_json()
    assert len(search_payload["data"]) >= 1
    assert any(sample_name.lower() in p["full_name"].lower() for p in search_payload["data"])


# ==============================================================================
# 3. Patient Chart View Tests
# ==============================================================================

def test_get_patient_chart_success(client: FlaskClient):
    """Verify full clinical chart retrieval for existing patient."""
    response = client.get("/api/patients/1")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    chart = payload["data"]
    assert chart["patient_id"] == 1
    assert "mrn" in chart
    assert "full_name" in chart
    assert "ward" in chart
    assert "bed_number" in chart
    assert "latest_vitals" in chart
    assert "latest_labs" in chart
    assert "active_medications" in chart
    assert isinstance(chart["active_medications"], list)


def test_get_patient_chart_not_found(client: FlaskClient):
    """Verify 404 response for nonexistent patient chart."""
    response = client.get("/api/patients/999999")
    assert response.status_code == 404

    payload = response.get_json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "PATIENT_NOT_FOUND"


# ==============================================================================
# 4. Patient Registration Tests
# ==============================================================================

def test_register_patient_success(client: FlaskClient):
    """Verify admission registration of a new patient."""
    new_patient_data = {
        "full_name": "Test Admission Subject",
        "age": 54,
        "gender": "M",
        "ward": "Step-Down telemetry",
        "bed_number": "SD-401",
        "status": "Admitted",
    }
    response = client.post(
        "/api/patients",
        data=json.dumps(new_patient_data),
        content_type="application/json",
    )
    assert response.status_code == 201

    payload = response.get_json()
    assert payload["success"] is True
    new_id = payload["data"]["patient_id"]
    assert new_id > 100

    # Clean up test registration so database retains canonical 100-patient cohort
    with get_connection() as conn:
        conn.execute("DELETE FROM patients WHERE patient_id = ?", (new_id,))
        conn.commit()


def test_register_patient_validation_error(client: FlaskClient):
    """Verify 422 error on patient registration with missing required fields."""
    incomplete_data = {
        "full_name": "Incomplete Patient",
        # missing age, gender, ward, bed_number
    }
    response = client.post(
        "/api/patients",
        data=json.dumps(incomplete_data),
        content_type="application/json",
    )
    assert response.status_code == 422

    payload = response.get_json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "VALIDATION_ERROR"


# ==============================================================================
# 5. Longitudinal Vitals & Labs Tests
# ==============================================================================

def test_get_patient_vitals_series(client: FlaskClient):
    """Verify longitudinal vitals time series with derived hemodynamics."""
    response = client.get("/api/patients/1/vitals")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    assert isinstance(payload["data"], list)
    assert len(payload["data"]) >= 1

    entry = payload["data"][0]
    assert "recorded_at" in entry
    assert "heart_rate" in entry
    assert "blood_pressure_sys" in entry
    assert "blood_pressure_dia" in entry
    assert "mean_arterial_pressure" in entry
    assert "shock_index" in entry


def test_get_patient_labs(client: FlaskClient):
    """Verify lab biomarkers chronological history retrieval."""
    response = client.get("/api/patients/1/labs")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    assert isinstance(payload["data"], list)
    if len(payload["data"]) > 0:
        lab = payload["data"][0]
        assert "recorded_at" in lab
        assert "glucose" in lab
        assert "creatinine" in lab


# ==============================================================================
# 6. Live Clinical Risk Assessment & Explainability Tests
# ==============================================================================

def test_predict_patient_risk_live(client: FlaskClient):
    """Verify live multi-model prediction and database auditing."""
    response = client.get("/api/patients/1/predict")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    pred = payload["data"]

    # Verify multi-model outputs
    assert pred["patient_id"] == 1
    assert pred["risk_tier"] in ["Critical", "High", "Medium", "Low"]
    assert 0.0 <= pred["risk_score"] <= 1.0
    assert "model_consensus" in pred
    assert "kmeans_tier" in pred["model_consensus"]
    assert "decision_tree_explanation" in pred
    assert isinstance(pred["decision_tree_explanation"], list)
    assert "similar_patients" in pred
    assert isinstance(pred["similar_patients"], list)
    assert "recommended_protocol" in pred
    assert "model_consensus" in pred
    assert "readmission_30d_risk" in pred

    # Verify that prediction was written into predictions audit table
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE patient_id = 1")
        count = cursor.fetchone()[0]
        assert count >= 1


def test_predict_patient_not_found(client: FlaskClient):
    """Verify 404 response on predict endpoint for invalid patient."""
    response = client.get("/api/patients/999999/predict")
    assert response.status_code == 404
    payload = response.get_json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "PATIENT_NOT_FOUND"


# ==============================================================================
# 7. Case Precedent Similarity Search Tests
# ==============================================================================

def test_get_similar_cases(client: FlaskClient):
    """Verify top-K similar historical precedents endpoint."""
    response = client.get("/api/patients/1/similar?k=3")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    sim_data = payload["data"]
    assert sim_data["patient_id"] == 1
    assert "recommended_protocol" in sim_data
    assert "protocol_success_rate" in sim_data
    assert len(sim_data["similar_patients"]) <= 3

    if len(sim_data["similar_patients"]) > 0:
        precedent = sim_data["similar_patients"][0]
        assert "patient_id" in precedent
        assert "similarity_score" in precedent
        assert "protocol" in precedent


# ==============================================================================
# 8. Executive Dashboard Summary Metrics Tests
# ==============================================================================

def test_dashboard_summary_metrics(client: FlaskClient):
    """Verify hospital-wide dashboard aggregate statistics."""
    response = client.get("/api/dashboard/summary")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    summary = payload["data"]

    assert summary["total_admitted"] >= 1
    assert "risk_distribution" in summary
    assert "Critical" in summary["risk_distribution"]
    assert "High" in summary["risk_distribution"]
    assert "Medium" in summary["risk_distribution"]
    assert "Low" in summary["risk_distribution"]
    assert summary["icu_occupancy"] >= 0
    assert summary["active_deterioration_alerts"] >= 0
    assert 0.0 <= summary["average_30d_readmission_risk"] <= 1.0
    assert isinstance(summary["ward_distribution"], dict)
    assert summary["early_warning_window"] == "7–14 Days"


# ==============================================================================
# 9. Deterioration Alerts Feed & Acknowledgement Tests
# ==============================================================================

def test_get_deterioration_alerts(client: FlaskClient):
    """Verify active early-warning alerts feed for high & critical patients."""
    response = client.get("/api/alerts")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    assert isinstance(payload["data"], list)

    if len(payload["data"]) > 0:
        alert = payload["data"][0]
        assert "alert_id" in alert
        assert "patient_id" in alert
        assert alert["risk_tier"] in ["Critical", "High"]
        assert "urgency_level" in alert
        assert alert["early_warning_window"] == "7–14 Days"
        assert isinstance(alert["physiological_triggers"], list)
        assert len(alert["physiological_triggers"]) >= 1
        assert "latest_telemetry" in alert


def test_acknowledge_alert(client: FlaskClient):
    """Verify physician clinical acknowledgement endpoint."""
    ack_payload = {
        "doctor_name": "Dr. Sarah Chen, MD",
        "notes": "Patient reassessed at bedside; titration of vasopressors ordered.",
    }
    response = client.post(
        "/api/alerts/1/acknowledge",
        data=json.dumps(ack_payload),
        content_type="application/json",
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    assert payload["data"]["patient_id"] == 1
    assert payload["data"]["acknowledged_by"] == "Dr. Sarah Chen, MD"
    assert "acknowledged_at" in payload["data"]


# ==============================================================================
# 10. Patient Medical Vault CRUD Tests
# ==============================================================================

def test_vault_record_upload_and_get(client: FlaskClient):
    """Verify medical record upload to vault and retrieval with category summary."""
    new_record = {
        "patient_id": 2,
        "category": "allergy",
        "title": "API Uploaded Penicillin Allergy",
        "description": "Severe anaphylactic reaction observed during 2021 admission",
        "structured_data": {"severity": "High", "symptoms": ["urticaria", "bronchospasm"]},
        "is_sensitive": False,
    }
    # 1. Upload
    up_res = client.post(
        "/api/vault/upload",
        data=json.dumps(new_record),
        content_type="application/json",
    )
    assert up_res.status_code == 201
    up_payload = up_res.get_json()
    assert up_payload["success"] is True
    assert up_payload["data"]["vault_id"] > 0

    # 2. Retrieve patient vault
    get_res = client.get("/api/vault/2")
    assert get_res.status_code == 200
    get_payload = get_res.get_json()
    assert get_payload["success"] is True
    assert get_payload["data"]["patient_id"] == 2
    assert "category_summary" in get_payload["data"]
    assert get_payload["data"]["category_summary"]["allergy"] >= 1
    assert len(get_payload["data"]["records"]) >= 1


def test_vault_upload_invalid_category(client: FlaskClient):
    """Verify 422 error on invalid medical record category."""
    invalid_record = {
        "patient_id": 2,
        "category": "horoscope",  # Invalid category
        "title": "Non-clinical Entry",
    }
    response = client.post(
        "/api/vault/upload",
        data=json.dumps(invalid_record),
        content_type="application/json",
    )
    assert response.status_code == 422
    payload = response.get_json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "INVALID_CATEGORY"


# ==============================================================================
# 11. Cryptographic QR Sharing, Scoped Access & Revocation Tests
# ==============================================================================

def test_vault_share_scoped_access_and_revocation(client: FlaskClient):
    """Verify end-to-end QR pass generation, scoped access, and 1-click revocation."""
    patient_id = 3

    # Add a lab record and a surgery record so we can test scope isolation
    client.post(
        "/api/vault/upload",
        data=json.dumps({
            "patient_id": patient_id,
            "category": "lab_report",
            "title": "Confidential Arterial Blood Gas",
        }),
        content_type="application/json",
    )
    client.post(
        "/api/vault/upload",
        data=json.dumps({
            "patient_id": patient_id,
            "category": "surgery",
            "title": "Appendectomy Operative Report",
        }),
        content_type="application/json",
    )

    # 1. Generate QR Pass strictly scoped to ['lab_report']
    share_payload = {
        "scope": ["lab_report"],
        "expires_in_minutes": 60,
        "max_uses": 5,
    }
    share_res = client.post(
        f"/api/vault/{patient_id}/share",
        data=json.dumps(share_payload),
        content_type="application/json",
    )
    assert share_res.status_code == 201
    share_data = share_res.get_json()["data"]
    token = share_data["token"]
    token_id = share_data["token_id"]
    assert "qr_code_base64" in share_data
    assert share_data["qr_code_base64"].startswith("data:image/png;base64,")

    # 2. Access vault using token (Authorized scope: lab_report)
    access_req = {
        "token": token,
        "accessed_by": "Dr. Pulmonary Specialist, MD",
    }
    access_res = client.post(
        "/api/vault/access",
        data=json.dumps(access_req),
        content_type="application/json",
    )
    assert access_res.status_code == 200
    access_data = access_res.get_json()["data"]
    assert access_data["access_granted"] is True
    assert access_data["patient_id"] == patient_id
    assert access_data["authorized_scope"] == ["lab_report"]

    # Verify Zero Data Leakage: surgery record MUST NOT be present
    records = access_data["records"]
    categories_returned = {r["category"] for r in records}
    assert "surgery" not in categories_returned

    # 3. Patient revokes pass immediately (1-click revocation)
    revoke_res = client.post(
        f"/api/vault/revoke/{token_id}",
        data=json.dumps({"patient_id": patient_id}),
        content_type="application/json",
    )
    assert revoke_res.status_code == 200
    assert revoke_res.get_json()["data"]["revoked"] is True

    # 4. Attempt to access again with revoked token -> must return 403 Forbidden
    retry_res = client.post(
        "/api/vault/access",
        data=json.dumps(access_req),
        content_type="application/json",
    )
    assert retry_res.status_code == 403
    retry_payload = retry_res.get_json()
    assert retry_payload["success"] is False
    assert retry_payload["error"]["code"] == "REVOKED"


# ==============================================================================
# 12. Vault Audit Log & Active Tokens Telemetry Tests
# ==============================================================================

def test_vault_access_audit_log_and_active_tokens(client: FlaskClient):
    """Verify patient audit trail and active tokens query endpoints."""
    patient_id = 3

    # Check active tokens
    tokens_res = client.get(f"/api/vault/{patient_id}/active-tokens")
    assert tokens_res.status_code == 200
    tokens_data = tokens_res.get_json()
    assert "active_count" in tokens_data["meta"]

    # Check access log
    audit_res = client.get(f"/api/vault/{patient_id}/access-log")
    assert audit_res.status_code == 200
    audit_data = audit_res.get_json()
    assert "entries" in audit_data["meta"]
    assert isinstance(audit_data["data"], list)
    assert audit_data["meta"]["entries"] >= 1  # From previous access attempts


# ==============================================================================
# 13. Web View Routing & Centralized Error Handler Tests
# ==============================================================================

def test_web_portal_view_routes(client: FlaskClient):
    """Verify HTML template render view routes for Physician, Patient, and Scanner."""
    # Root redirects to /dashboard
    root_res = client.get("/")
    assert root_res.status_code == 302
    assert "/dashboard" in root_res.headers["Location"]

    # Dashboard view renders HTML with dashboard.js & triage elements
    dash_res = client.get("/dashboard")
    assert dash_res.status_code == 200
    assert b"MediHaven" in dash_res.data
    assert b"Physician Clinical Triage" in dash_res.data
    assert b"dashboard.js" in dash_res.data
    assert b"chart.umd.min.js" in dash_res.data

    # Vault view renders HTML with vault.js & patient sovereignty components
    vault_res = client.get("/vault")
    assert vault_res.status_code == 200
    assert b"MediHaven" in vault_res.data
    assert b"Sovereign Patient Medical Vault" in vault_res.data
    assert b"vault.js" in vault_res.data

    # Scanner view renders HTML with scanner.js & optical viewfinder
    scanner_res = client.get("/scanner")
    assert scanner_res.status_code == 200
    assert b"MediHaven" in scanner_res.data
    assert b"Provider Optical QR Scanner" in scanner_res.data
    assert b"scanner.js" in scanner_res.data


def test_centralized_error_handlers(client: FlaskClient):
    """Verify JSON envelope response on HTTP 404 and HTTP 405."""
    # 404 on API endpoint
    res_404 = client.get("/api/nonexistent-route-endpoint")
    assert res_404.status_code == 404
    payload_404 = res_404.get_json()
    assert payload_404["success"] is False
    assert payload_404["error"]["code"] == "NOT_FOUND"

    # 405 on POST to GET-only route
    res_405 = client.post("/api/health")
    assert res_405.status_code == 405
    payload_405 = res_405.get_json()
    assert payload_405["success"] is False
    assert payload_405["error"]["code"] == "METHOD_NOT_ALLOWED"
