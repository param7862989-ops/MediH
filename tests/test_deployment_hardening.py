"""Deployment hardening: production configuration, credentials, headers, rate limits, pass quotas,
audit identities, input validation and the WSGI entrypoint.

Uses the session's temporary synthetic database (tests/conftest.py); no real data or secrets.
"""

import io
import os
import re
import sqlite3
import subprocess
import sys
import threading

import pytest
from PIL import Image
from werkzeug.security import generate_password_hash

from src.api.app import create_app
from src.bootstrap import database_seeded
from src.utils.config import Config
from src.utils.security import ProductionConfigError
from src.vault.access_controller import access_vault_with_token
from src.vault.qr_generator import generate_vault_access_token
from src.vault.qr_scanner import MAX_QR_IMAGE_PIXELS, extract_qr_text_from_image
from tests.isolation_utils import PROJECT_ROOT

ENFORCE = {"X-Enforce-Auth": "1"}
STRONG = "a" * 16 + "B" * 16 + "7" * 16


def _login(app, username, password):
    client = app.test_client()
    response = client.post("/api/auth/login", json={"username": username, "password": password}, headers=ENFORCE)
    assert response.status_code == 200, response.get_json()
    return client


@pytest.fixture
def production(monkeypatch):
    """FLASK_ENV=production with strong (throwaway) signing secrets."""
    monkeypatch.setattr(Config, "IS_PRODUCTION", True)
    monkeypatch.setattr(Config, "FLASK_SECRET_KEY", STRONG)
    monkeypatch.setattr(Config, "JWT_SECRET_KEY", STRONG[::-1])


# ------------------------------------------------------------------------------
# Production configuration
# ------------------------------------------------------------------------------
def test_production_refuses_public_default_secrets(monkeypatch):
    monkeypatch.setattr(Config, "IS_PRODUCTION", True)
    with pytest.raises(ProductionConfigError, match="FLASK_SECRET_KEY is the public default"):
        create_app({"TESTING": True})


def test_weak_secrets_are_refused_on_a_network_bind_address(monkeypatch):
    monkeypatch.setattr(Config, "FLASK_HOST", "0.0.0.0")
    with pytest.raises(ProductionConfigError):
        create_app({"TESTING": True})


def test_production_starts_with_strong_secrets_and_no_debugger(production, monkeypatch):
    monkeypatch.setattr(Config, "FLASK_DEBUG", True)
    assert create_app({"TESTING": True}).debug is False


def test_debugger_follows_config_not_ambient_flask_debug(monkeypatch):
    monkeypatch.setenv("FLASK_DEBUG", "1")
    monkeypatch.setattr(Config, "FLASK_DEBUG", False)
    assert create_app({"TESTING": True}).debug is False


# ------------------------------------------------------------------------------
# Credentials
# ------------------------------------------------------------------------------
def test_source_holds_password_hashes_only():
    source = (PROJECT_ROOT / "src" / "api" / "auth.py").read_text(encoding="utf-8")
    for password in ("Vidhi@123", "Nirupam@123", "Param@123", "Physician@123", "Patient@123"):
        assert password not in source


def test_production_disables_development_admin_passwords(production, monkeypatch):
    monkeypatch.setattr(Config, "DEMO_LOGINS", False)
    client = create_app({"TESTING": True, "RATE_LIMITS_ENABLED": False}).test_client()
    for username, password in (("param", "Param@123"), ("physician1", "Physician@123")):
        response = client.post("/api/auth/login", json={"username": username, "password": password}, headers=ENFORCE)
        assert response.status_code == 401


def test_production_password_hash_override_and_demo_logins(production, monkeypatch):
    monkeypatch.setattr(Config, "DEMO_LOGINS", True)
    monkeypatch.setenv("PASSWORD_HASH_PARAM", generate_password_hash("a-long-deployment-password"))
    app = create_app({"TESTING": True, "RATE_LIMITS_ENABLED": False})
    _login(app, "param", "a-long-deployment-password")
    _login(app, "physician1", "Physician@123")  # public demo account, enabled by DEMO_LOGINS
    client = app.test_client()
    assert client.post("/api/auth/login", json={"username": "param", "password": "Param@123"},
                       headers=ENFORCE).status_code == 401


def test_login_page_shows_demo_credentials_only_when_enabled(monkeypatch):
    client = create_app({"TESTING": True}).test_client()
    assert b"Physician@123" in client.get("/login").data
    monkeypatch.setattr(Config, "DEMO_LOGINS", False)
    assert b"Physician@123" not in create_app({"TESTING": True}).test_client().get("/login").data


def test_login_is_rate_limited():
    client = create_app({"TESTING": True}).test_client()
    statuses = [client.post("/api/auth/login", json={"username": "param", "password": "wrong"}).status_code
                for _ in range(11)]
    assert statuses[:10] == [401] * 10 and statuses[10] == 429
    blocked = client.post("/api/auth/login", json={"username": "param", "password": "wrong"})
    assert blocked.get_json()["error"]["code"] == "RATE_LIMITED" and int(blocked.headers["Retry-After"]) > 0


def test_login_rejects_non_object_bodies():
    client = create_app({"TESTING": True}).test_client()
    assert client.post("/api/auth/login", json=["param", "x"]).status_code == 401
    assert client.post("/api/auth/login", json={"username": None, "password": 5}).status_code == 401


# ------------------------------------------------------------------------------
# Headers and page scripts
# ------------------------------------------------------------------------------
def test_security_headers_are_set():
    client = create_app({"TESTING": True}).test_client()
    response = client.get("/")
    csp = response.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "camera=(self)" in response.headers["Permissions-Policy"]
    assert client.get("/api/health").headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("path", ["/", "/login", "/dashboard", "/vault", "/scanner", "/admin"])
def test_pages_have_no_inline_scripts_blocked_by_csp(path):
    html = create_app({"TESTING": True}).test_client().get(path).get_data(as_text=True)
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "inline <script> would be blocked by the CSP"
    assert not re.search(r"\son[a-z]+\s*=\s*\"", html), "inline event handler would be blocked by the CSP"
    assert "cdn.jsdelivr.net" not in html


def test_login_notice_comes_from_a_fixed_list():
    script = (PROJECT_ROOT / "static" / "js" / "login.js").read_text(encoding="utf-8")
    assert "LOGIN_NOTICES[code] || LOGIN_NOTICES.signin" in script


# ------------------------------------------------------------------------------
# QR passes: quota under concurrency, scope, audit identity, input validation
# ------------------------------------------------------------------------------
def test_single_use_pass_is_granted_once_under_concurrent_scans():
    token = generate_vault_access_token(patient_id=1, scope=["*"], expires_in_minutes=10, max_uses=1)
    results, barrier = [], threading.Barrier(4)

    def scan():
        barrier.wait()
        results.append(access_vault_with_token(token["qr_payload"], accessed_by="Concurrency test"))

    threads = [threading.Thread(target=scan) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(r["access_granted"] for r in results) == 1
    assert {r["access_status"] for r in results if not r["access_granted"]} == {"MAX_USES_EXCEEDED"}


def test_access_scope_comes_from_the_stored_grant():
    token = generate_vault_access_token(patient_id=1, scope=["allergy"], expires_in_minutes=10)
    with sqlite3.connect(Config.DATABASE_PATH) as conn:
        conn.execute("UPDATE vault_access_tokens SET scope_json = ? WHERE token_id = ?",
                     ('["medication"]', token["token_id"]))
    result = access_vault_with_token(token["qr_payload"])
    assert result["authorized_scope"] == ["medication"]
    assert all(r["category"] == "medication" for r in result["records"])


def test_audit_log_records_the_signed_in_clinician_not_the_typed_name():
    app = create_app({"TESTING": True})
    token = generate_vault_access_token(patient_id=1, scope=["*"], expires_in_minutes=10)
    physician = _login(app, "physician1", "Physician@123")
    response = physician.post("/api/vault/access", headers=ENFORCE,
                              json={"token": token["qr_payload"], "accessed_by": "Dr Nobody (forged)"})
    assert response.status_code == 200
    assert response.get_json()["data"]["accessed_by"] == "Dr. Sarah Chen, MD (Attending)"
    ack = physician.post("/api/alerts/1/acknowledge", headers=ENFORCE, json={"doctor_name": "Someone Else"})
    assert ack.get_json()["data"]["acknowledged_by"] == "Dr. Sarah Chen, MD (Attending)"


@pytest.mark.parametrize("method, path, body", [
    ("post", "/api/vault/share", [1, 2, 3]),
    ("post", "/api/vault/upload", ["not", "an", "object"]),
    ("post", "/api/alerts/1/acknowledge", {"doctor_name": None, "notes": None}),
    ("post", "/api/alerts/dispatch", {"patient_id": [1]}),
    ("post", "/api/patients", [1]),
    ("get", "/api/patients/1/vitals?hours=999999999999", None),
])
def test_malformed_input_never_causes_a_server_error(method, path, body):
    client = create_app({"TESTING": True}).test_client()
    response = getattr(client, method)(path, json=body) if body is not None else getattr(client, method)(path)
    assert response.status_code < 500, response.get_data(as_text=True)


@pytest.mark.parametrize("field, value", [("expires_in_minutes", 10 ** 12), ("max_uses", 10 ** 9), ("max_uses", True)])
def test_pass_lifetime_and_quota_are_bounded(field, value):
    client = create_app({"TESTING": True}).test_client()
    response = client.post("/api/vault/1/share", json={"scope": ["*"], field: value})
    assert response.status_code == 422


@pytest.mark.parametrize("payload", [str(PROJECT_ROOT / "README.md"), {"nested": "object"}, 12345])
def test_scanner_accepts_only_tokens_or_image_data(payload):
    client = create_app({"TESTING": True}).test_client()
    response = client.post("/api/vault/access", json={"qr_payload": payload})
    assert response.status_code == 422


def test_oversized_qr_images_are_rejected_before_decoding():
    side = int(MAX_QR_IMAGE_PIXELS ** 0.5) + 1
    buffer = io.BytesIO()
    Image.new("1", (side, side)).save(buffer, format="PNG")  # tiny file, huge pixel count
    assert extract_qr_text_from_image(buffer.getvalue()) is None


def test_errors_do_not_leak_internal_details(monkeypatch):
    from src.models.ensemble import EnsembleClinicalPredictor
    app = create_app({"TESTING": True})
    app.ensemble_predictor = None

    def failing_load(*args, **kwargs):
        raise FileNotFoundError(r"C:\secret\internal\path\model.joblib")

    monkeypatch.setattr(EnsembleClinicalPredictor, "load", failing_load)
    body = app.test_client().get("/api/patients/1/predict").get_data(as_text=True)
    assert "INFERENCE_FAILED" in body and "secret" not in body and "joblib" not in body


# ------------------------------------------------------------------------------
# Bootstrap and WSGI entrypoint
# ------------------------------------------------------------------------------
def test_empty_or_missing_database_counts_as_not_seeded(monkeypatch, tmp_path):
    assert database_seeded()  # the session's seeded test database
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    monkeypatch.setattr(Config, "DATABASE_PATH", empty)
    assert not database_seeded()
    monkeypatch.setattr(Config, "DATABASE_PATH", tmp_path / "missing.db")
    assert not database_seeded()


def test_wsgi_entrypoint_refuses_research_mode(tmp_path):
    env = {**os.environ, "DATA_MODE": "mimic", "LOG_FILE": str(tmp_path / "app.log"),
           "AUDIT_LOG_FILE": str(tmp_path / "audit.log"), "DATABASE_PATH": str(tmp_path / "db.sqlite")}
    result = subprocess.run([sys.executable, "-c", "import wsgi"], cwd=PROJECT_ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "Research mode (DATA_MODE=mimic) must stay on loopback" in result.stderr
    assert not (tmp_path / "db.sqlite").exists()


def test_narrative_limit_applies_only_to_live_gemini_calls(monkeypatch):
    from src.nlp import summarizer as summarizer_instance
    client = create_app({"TESTING": True}).test_client()
    monkeypatch.setattr(summarizer_instance, "is_configured", lambda: False)
    assert all(client.get("/api/patients/1/clinical-narrative").status_code == 200 for _ in range(65))
    monkeypatch.setattr(summarizer_instance, "is_configured", lambda: True)
    monkeypatch.setattr(summarizer_instance, "generate", lambda *a, **k: {"narrative": "stub", "engine": "test"})
    statuses = [client.get("/api/patients/1/clinical-narrative").status_code for _ in range(61)]
    assert statuses[:60] == [200] * 60 and statuses[60] == 429
