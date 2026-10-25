"""Phase 3 security tests: Gemini TLS and data egress, credentials, cookies, CORS and request limits."""

import importlib
import io
import json
import logging
import ssl
import urllib.error

import pytest

from src.api.app import create_app
from src.nlp.summarizer import PATIENT_PLACEHOLDER, ClinicalNarrativeSummarizer
from src.utils.config import Config
from tests.isolation_utils import PROJECT_ROOT

# src/nlp/__init__.py re-exports the `summarizer` instance under the submodule's name
summarizer_module = importlib.import_module("src.nlp.summarizer")

FAKE_KEY = "AIzaFAKEKEY0123456789abcdefghijKLMN"
PATIENT = {"full_name": "Aarav Verma", "mrn": "MRN-2026-0001", "age": 67, "gender": "M", "ward": "ICU",
           "symptoms": "Fever 39.4C, rigors, confusion", "primary_diagnosis": "Septic Shock (A41.9)"}
PREDICTION = {"risk_tier": "Critical", "risk_score": 0.93, "readmission_30d_risk": 0.41,
              "recommended_protocol": "Protocol A: Sepsis Bundle (Aggressive IV resuscitation, blood cultures)",
              "decision_tree_explanation": ["Glucose Latest > 152.30 mg/dL [Observed: 178.40 mg/dL]"]}
SENSITIVE = ["Aarav", "Verma", "MRN-2026-0001", "Fever", "rigors", "Septic", "A41.9", "ICU", "67", "178.40",
             "Glucose", "Aggressive IV"]


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def captured_calls(monkeypatch):
    calls = []

    def fake_urlopen(request, context=None, timeout=None):
        calls.append({"request": request, "context": context})
        body = {"candidates": [{"content": {"parts": [{"text": f"{PATIENT_PLACEHOLDER} remains in the Critical tier."}]}}]}
        return _FakeResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(summarizer_module.urllib.request, "urlopen", fake_urlopen)
    return calls


# ------------------------------------------------------------------------------
# Gemini: TLS verification, key handling, de-identified prompts
# ------------------------------------------------------------------------------
def test_gemini_calls_verify_tls_certificates(captured_calls):
    ClinicalNarrativeSummarizer(api_key=FAKE_KEY).generate(PATIENT, PREDICTION, "handover")
    context = captured_calls[0]["context"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True


def test_gemini_key_is_sent_in_header_not_url(captured_calls):
    ClinicalNarrativeSummarizer(api_key=FAKE_KEY).generate(PATIENT, PREDICTION, "handover")
    request = captured_calls[0]["request"]
    assert FAKE_KEY not in request.full_url and "key=" not in request.full_url
    assert request.get_header("X-goog-api-key") == FAKE_KEY
    assert request.full_url.startswith("https://")


@pytest.mark.parametrize("narrative_type", ["handover", "discharge"])
def test_gemini_prompt_contains_no_identifying_or_sensitive_content(captured_calls, narrative_type):
    result = ClinicalNarrativeSummarizer(api_key=FAKE_KEY).generate(PATIENT, PREDICTION, narrative_type)
    sent = captured_calls[0]["request"].data.decode("utf-8")
    for fragment in SENSITIVE:
        assert fragment not in sent, f"'{fragment}' was sent to the external AI service"
    assert PATIENT_PLACEHOLDER in sent and "Critical" in sent
    # the name is restored locally after the response
    assert result["is_live"] is True and result["narrative"].startswith("Aarav Verma")


def test_no_external_ai_call_in_research_mode(monkeypatch):
    monkeypatch.setattr(Config, "DATA_MODE", "mimic")
    monkeypatch.setattr(summarizer_module.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("external call in research mode")))
    result = ClinicalNarrativeSummarizer(api_key=FAKE_KEY).generate(PATIENT, PREDICTION, "handover")
    assert result["is_live"] is False and "disabled in research mode" in result["external_ai"]


def test_gemini_errors_never_log_key_or_response_body(monkeypatch, caplog):
    def failing_urlopen(request, context=None, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {},
                                     io.BytesIO(f"secret-body {FAKE_KEY}".encode("utf-8")))

    monkeypatch.setattr(summarizer_module.urllib.request, "urlopen", failing_urlopen)
    with caplog.at_level(logging.INFO):
        result = ClinicalNarrativeSummarizer(api_key=FAKE_KEY).generate(PATIENT, PREDICTION, "handover")
    assert result["is_live"] is False
    assert FAKE_KEY not in caplog.text and "secret-body" not in caplog.text and "HTTP 403" in caplog.text


def test_tls_verification_is_never_disabled_in_source():
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "CERT_NONE" not in text and "check_hostname = False" not in text, path
        assert "_create_unverified_context" not in text, path


def test_admin_telemetry_exposes_at_most_last_four_key_characters(monkeypatch):
    monkeypatch.setattr(summarizer_module.summarizer, "api_key", FAKE_KEY)
    client = create_app({"TESTING": True}).test_client()
    llm = client.get("/api/admin/models").get_json()["data"]["llm_service"]
    assert llm["configured"] is True and llm["masked_key"] == "…" + FAKE_KEY[-4:]
    assert FAKE_KEY[:6] not in json.dumps(llm)


# ------------------------------------------------------------------------------
# Credentials, cookies, CORS and request limits
# ------------------------------------------------------------------------------
def test_undocumented_weak_passwords_are_rejected():
    client = create_app({"TESTING": True}).test_client()
    assert client.post("/api/auth/login", json={"username": "physician1", "password": "physician123"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "patient1", "password": "patient123"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "physician1", "password": "Physician@123"}).status_code == 200


def test_session_cookie_is_httponly_and_samesite():
    client = create_app({"TESTING": True}).test_client()
    response = client.post("/api/auth/login", json={"username": "physician1", "password": "Physician@123"})
    cookie = response.headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie


def test_cors_is_same_origin_by_default_and_allowlist_only(monkeypatch):
    client = create_app({"TESTING": True}).test_client()
    response = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in response.headers

    monkeypatch.setattr(Config, "CORS_ORIGINS", ["https://trusted.example"])
    client = create_app({"TESTING": True}).test_client()
    trusted = client.get("/api/health", headers={"Origin": "https://trusted.example"})
    assert trusted.headers.get("Access-Control-Allow-Origin") == "https://trusted.example"
    other = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert other.headers.get("Access-Control-Allow-Origin") != "https://evil.example"


def test_oversized_requests_are_rejected(monkeypatch):
    monkeypatch.setattr(Config, "MAX_REQUEST_MB", 0.01)
    client = create_app({"TESTING": True}).test_client()
    response = client.post("/api/vault/access", data=b"x" * 20000, content_type="application/json")
    assert response.status_code == 413 and response.get_json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_synthetic_mode_warns_about_default_secrets(caplog):
    with caplog.at_level(logging.WARNING):
        create_app({"TESTING": True})
    assert "SECURITY: FLASK_SECRET_KEY is the public default" in caplog.text
