"""Phase 3 tests: explicit research mode in the web application.

Covers mode selection, refusal to seed/recreate/load synthetic assets, blocking of synthetic clinical
APIs, preservation of the patient vault and QR sharing, verified-artifact inference, fail-closed
artifact handling and the absence of identifiers in responses and logs.

Research artifacts are produced by the real Phase 2 pipeline from small synthetic fixtures
(tests/training_fixtures.py); no real MIMIC-IV data is used.
"""

import hashlib
import json
import shutil

import joblib
import pytest

from src.api.app import create_app
from src.datasources.mimic.mappings import feature_columns
from src.training.train_mimic_t1 import output_dir_for, train_and_evaluate
from src.utils.config import Config
from src.utils.security import ResearchModeConfigError
from tests.isolation_utils import PROJECT_ROOT
from tests.training_fixtures import SUBJECT_BASE, write_task1_dataset

STRONG_SECRET = "a" * 24 + "research-test-secret-0123456789"
FIXTURE_IDS = [str(SUBJECT_BASE + i) for i in range(60)] + [str(78000000 + i) for i in range(70)] + \
              [str(79000000 + i) for i in range(70)]
ENFORCE = {"X-Enforce-Auth": "1"}


@pytest.fixture(scope="module")
def artifacts_dir(tmp_path_factory):
    """A complete, valid Phase 2 run built from a synthetic fixture dataset."""
    research_dir = tmp_path_factory.mktemp("research")
    write_task1_dataset(research_dir)
    train_and_evaluate(research_dir, n_boot=50)
    return research_dir


@pytest.fixture
def research_env(monkeypatch, artifacts_dir):
    """Research mode with safe secrets pointing at a valid artifact set."""
    monkeypatch.setattr(Config, "DATA_MODE", "mimic")
    monkeypatch.setattr(Config, "FLASK_SECRET_KEY", STRONG_SECRET)
    monkeypatch.setattr(Config, "JWT_SECRET_KEY", STRONG_SECRET[::-1])
    monkeypatch.setattr(Config, "FLASK_HOST", "127.0.0.1")
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", artifacts_dir)
    monkeypatch.setattr(Config, "RESEARCH_ARTIFACT_MANIFEST_SHA256", "")
    return artifacts_dir


def _login(app, username, password):
    client = app.test_client()
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return client


def _physician(app):
    return _login(app, "physician1", "Physician@123")


def _copy_artifacts(src, dst):
    shutil.copytree(src, dst)
    return dst


def _rehash(research_dir, relative):
    run_dir = output_dir_for(research_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_files_sha256"][relative] = hashlib.sha256((run_dir / relative).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


# ------------------------------------------------------------------------------
# Mode selection and synthetic-mode regression
# ------------------------------------------------------------------------------
def test_synthetic_mode_is_default_and_research_surface_disabled():
    app = create_app({"TESTING": True})
    client = app.test_client()
    health = client.get("/api/health").get_json()["data"]
    assert health["data_mode"] == "synthetic" and health["models_online"] is True
    assert app.research_service is None and app.ensemble_predictor is not None
    assert client.get("/api/research/status").status_code == 409
    assert client.get("/research").status_code == 404
    assert client.get("/api/patients").status_code == 200
    assert client.get("/dashboard").status_code == 200


@pytest.mark.parametrize("attribute, value, message", [
    ("FLASK_SECRET_KEY", Config.DEFAULT_FLASK_SECRET_KEY, "public default"),
    ("JWT_SECRET_KEY", "replace-with-a-secure-jwt-signing-key", "placeholder"),
    ("FLASK_SECRET_KEY", "short-secret", "shorter than"),
    ("FLASK_HOST", "0.0.0.0", "loopback"),
])
def test_research_mode_refuses_unsafe_configuration(research_env, monkeypatch, attribute, value, message):
    monkeypatch.setattr(Config, attribute, value)
    with pytest.raises(ResearchModeConfigError, match=message):
        create_app({"TESTING": True})


def test_research_mode_never_loads_synthetic_models_or_seeds(research_env, monkeypatch):
    from src.database.db import execute_query
    from src.models.ensemble import EnsembleClinicalPredictor
    import database.seed_data as seed_module

    def forbidden(*_a, **_k):
        raise AssertionError("synthetic asset touched in research mode")

    monkeypatch.setattr(EnsembleClinicalPredictor, "load", classmethod(lambda cls: forbidden()))
    monkeypatch.setattr(seed_module, "seed_database", forbidden)
    before = execute_query("SELECT COUNT(*) AS n FROM patients")[0]["n"]

    app = create_app({"TESTING": True})
    assert app.ensemble_predictor is None
    health = app.test_client().get("/api/health").get_json()["data"]
    assert health["data_mode"] == "mimic" and health["models_online"] is True
    assert execute_query("SELECT COUNT(*) AS n FROM patients")[0]["n"] == before


def test_research_mode_never_creates_the_demo_database(research_env, monkeypatch, tmp_path):
    from src.database.db import get_db_connection
    missing = tmp_path / "no_such.db"
    monkeypatch.setattr(Config, "DATABASE_PATH", missing)
    app = create_app({"TESTING": True})
    client = _login(app, "patient1", "Patient@123")
    response = client.get("/api/vault/1", headers=ENFORCE)
    assert response.status_code == 503 and response.get_json()["error"]["code"] == "DEMO_DATABASE_MISSING"
    with pytest.raises(Exception):
        get_db_connection()
    assert not missing.exists()


# ------------------------------------------------------------------------------
# Synthetic clinical intelligence is disabled; vault and QR sharing are preserved
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("method, path", [
    ("GET", "/api/patients"), ("GET", "/api/patients/1"), ("GET", "/api/patients/1/vitals"),
    ("GET", "/api/patients/1/predict"), ("GET", "/api/patients/1/similar"),
    ("GET", "/api/patients/1/clinical-narrative"), ("GET", "/api/dashboard/summary"),
    ("GET", "/api/alerts"), ("POST", "/api/alerts/1/acknowledge"), ("GET", "/api/admin/models"),
    ("POST", "/api/admin/gemini/synthesize"),
])
def test_synthetic_clinical_apis_are_blocked_in_research_mode(research_env, method, path):
    client = _physician(create_app({"TESTING": True}))
    response = client.open(path, method=method, json={}, headers=ENFORCE)
    assert response.status_code == 503
    assert response.get_json()["error"]["code"] == "UNAVAILABLE_IN_RESEARCH_MODE"


def test_dashboard_is_research_triage_and_admin_redirects_to_research_view(research_env):
    app = create_app({"TESTING": True})
    response = _physician(app).get("/dashboard", headers=ENFORCE)
    html = response.get_data(as_text=True)
    assert response.status_code == 200 and "js/research_triage.js" in html and "js/dashboard.js" not in html
    assert "NOT FOR PATIENT CARE" in html and 'id="research-triage-table"' in html
    response = _login(app, "param", "Param@123").get("/admin", headers=ENFORCE)
    assert response.status_code == 302 and response.headers["Location"].endswith("/research")


def test_patient_vault_and_qr_sharing_work_in_research_mode(research_env):
    app = create_app({"TESTING": True})
    patient = _login(app, "patient1", "Patient@123")
    assert patient.get("/api/patients/1", headers=ENFORCE).status_code == 200  # vault profile panel
    assert patient.get("/api/patients/2", headers=ENFORCE).status_code in (403, 503)
    vault = patient.get("/api/vault/1", headers=ENFORCE)
    assert vault.status_code == 200 and vault.get_json()["data"]["records_count"] > 0

    share = patient.post("/api/vault/1/share", json={"scope": ["vaccination"], "max_uses": 2}, headers=ENFORCE)
    assert share.status_code == 201
    token, token_id = share.get_json()["data"]["token"], share.get_json()["data"]["token_id"]

    physician = _physician(app)
    access = physician.post("/api/vault/access", json={"token": token}, headers=ENFORCE)
    assert access.status_code == 200
    data = access.get_json()["data"]
    assert data["access_granted"] and {r["category"] for r in data["records"]} <= {"vaccination"}

    assert patient.post(f"/api/vault/revoke/{token_id}", json={}, headers=ENFORCE).status_code == 200
    denied = physician.post("/api/vault/access", json={"token": token}, headers=ENFORCE)
    assert denied.status_code == 403 and denied.get_json()["error"]["code"] == "REVOKED"


# ------------------------------------------------------------------------------
# Research endpoints
# ------------------------------------------------------------------------------
def test_research_status_model_card_and_view(research_env):
    app = create_app({"TESTING": True})
    client = _physician(app)
    status = client.get("/api/research/status", headers=ENFORCE).get_json()["data"]
    assert status["research_inference_available"] is True
    assert status["dataset"]["dataset_version"] == "1.2.0" and status["dataset"]["task_version"] == "3"
    assert "NOT clinically validated" in status["disclaimer"]
    card = client.get("/api/research/model-card", headers=ENFORCE).get_json()["data"]
    assert card["primary"]["population"]["stays"] == 70
    assert set(card["primary"]["models"]) == {"dummy_prior", "logistic_regression", "decision_tree", "random_forest", "neural_network"}
    page = client.get("/research", headers=ENFORCE)
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "RESEARCH MODE" in html and "NOT FOR PATIENT CARE" in html and "Verified artifacts" in html


def test_research_prediction_uses_permitted_features_only(research_env):
    client = _physician(create_app({"TESTING": True}))
    payload = {"heart_rate_count": 6, "heart_rate_min": 95, "heart_rate_max": 128, "heart_rate_mean": 112,
               "heart_rate_last": 120, "lactate_count": 1, "lactate_last": 4.1, "age": 71, "sbp_min": None}
    response = client.post("/api/research/predict", headers=ENFORCE, json={"features": payload})
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["research_only"] is True and "NOT clinically validated" in data["disclaimer"]
    assert set(data["probabilities"]) == {"dummy_prior", "logistic_regression", "decision_tree", "random_forest", "neural_network"}
    assert all(0.0 <= p <= 1.0 for p in data["probabilities"].values())
    assert 0.0 < data["probabilities"]["decision_tree"] < 1.0  # smoothed leaves: never exactly 0% or 100%
    summary = data["input_summary"]
    assert summary["variables_measured"] == ["heart_rate", "lactate"]
    assert len(summary["variables_not_measured"]) == 12 and "sbp" in summary["variables_not_measured"]
    assert summary["demographics_imputed"] == ["sex_male"]
    assert "_missing = 1" in summary["encoding"] and "features_treated_as_missing" not in data
    assert "not clinically validated" in data["limitations"]


@pytest.mark.parametrize("payload, code", [
    ({"features": {"heart_rate_mean": 112}}, "INCOMPLETE_VARIABLE"),
    ({"features": {"heart_rate_count": 4, "heart_rate_last": 90, "heart_rate_missing": 1}}, "CONTRADICTORY_INPUT"),
    ({"features": {"heart_rate_count": -2, "heart_rate_last": 90}}, "INVALID_VALUE"),
])
def test_research_prediction_enforces_input_contract(research_env, payload, code):
    client = _physician(create_app({"TESTING": True}))
    response = client.post("/api/research/predict", json=payload, headers=ENFORCE)
    assert response.status_code == 422 and response.get_json()["error"]["code"] == code


@pytest.mark.parametrize("payload, code", [
    ({"features": {"src_subject_id": 1}}, "FORBIDDEN_FEATURE"),
    ({"features": {"label_inhospital_mortality": 1}}, "FORBIDDEN_FEATURE"),
    ({"features": {"outcome_anomaly_deathtime_after_discharge": 0}}, "FORBIDDEN_FEATURE"),
    ({"features": {"index_time": 0}}, "FORBIDDEN_FEATURE"),
    ({"features": {"risk_tier": 2}}, "FORBIDDEN_FEATURE"),
    ({"features": {"deathtime": 0}}, "FORBIDDEN_FEATURE"),
    ({"features": {"mystery_score": 1.0}}, "UNKNOWN_FEATURE"),
    ({"features": {"heart_rate_mean": "high"}}, "INVALID_VALUE"),
    ({"features": {"heart_rate_mean": True}}, "INVALID_VALUE"),
    ({"features": {}}, "INVALID_INPUT"),
    ({"features": [1, 2]}, "INVALID_INPUT"),
])
def test_research_prediction_rejects_forbidden_and_invalid_inputs(research_env, payload, code):
    client = _physician(create_app({"TESTING": True}))
    response = client.post("/api/research/predict", json=payload, headers=ENFORCE)
    assert response.status_code == 422 and response.get_json()["error"]["code"] == code


def test_research_endpoints_require_clinical_role(research_env):
    app = create_app({"TESTING": True})
    assert app.test_client().get("/api/research/status", headers=ENFORCE).status_code == 401
    patient = _login(app, "patient1", "Patient@123")
    assert patient.get("/api/research/status", headers=ENFORCE).status_code == 403
    assert patient.post("/api/research/predict", json={"features": {"age": 50}}, headers=ENFORCE).status_code == 403


def test_no_identifiers_or_supplied_values_in_responses_or_logs(research_env):
    app = create_app({"TESTING": True})
    client = _physician(app)
    log_before = Config.LOG_FILE.read_text(encoding="utf-8") if Config.LOG_FILE.exists() else ""
    texts = [
        client.get("/api/research/status", headers=ENFORCE).get_data(as_text=True),
        client.get("/api/research/model-card", headers=ENFORCE).get_data(as_text=True),
        client.get("/research", headers=ENFORCE).get_data(as_text=True),
        client.get("/api/research/stays", headers=ENFORCE).get_data(as_text=True),
        client.get("/api/research/stays/1", headers=ENFORCE).get_data(as_text=True),
        client.get("/dashboard", headers=ENFORCE).get_data(as_text=True),
        client.post("/api/research/predict", headers=ENFORCE,
                    json={"features": {"heart_rate_mean": 123.4567, "heart_rate_count": 1}}).get_data(as_text=True),
    ]
    assert '"probabilities"' in texts[-1]
    new_log = (Config.LOG_FILE.read_text(encoding="utf-8") if Config.LOG_FILE.exists() else "")[len(log_before):]
    for text in texts + [new_log]:
        assert not any(identifier in text for identifier in FIXTURE_IDS)
    assert "123.4567" not in new_log and str(research_env) not in "".join(texts)


# ------------------------------------------------------------------------------
# Artifact integrity: fail closed
# ------------------------------------------------------------------------------
def _status(app):
    return app.research_service.status()


def _assert_fails_closed(codes):
    app = create_app({"TESTING": True})
    status = _status(app)
    assert status["research_inference_available"] is False and status["reason_code"] in codes
    client = _physician(app)
    for method, path in (("GET", "/api/research/model-card"), ("POST", "/api/research/predict")):
        response = client.open(path, method=method, json={"features": {"age": 60}}, headers=ENFORCE)
        assert response.status_code == 503
    assert "Research model unavailable" in client.get("/research", headers=ENFORCE).get_data(as_text=True)
    return status


def test_missing_or_unconfigured_artifacts_fail_closed(research_env, monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", tmp_path / "empty")
    _assert_fails_closed({"ARTIFACTS_MISSING"})
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", None)
    _assert_fails_closed({"NOT_CONFIGURED"})
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", PROJECT_ROOT / "research_output_should_not_exist")
    _assert_fails_closed({"UNSAFE_LOCATION"})


def test_tampered_model_is_never_deserialized(research_env, monkeypatch, tmp_path):
    import src.training.artifacts as artifacts_module
    research_dir = _copy_artifacts(research_env, tmp_path / "r")
    with open(output_dir_for(research_dir) / "models" / "random_forest.joblib", "ab") as handle:
        handle.write(b"tampered")
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", research_dir)
    monkeypatch.setattr(artifacts_module.joblib, "load",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unverified file deserialized")))
    _assert_fails_closed({"ARTIFACT_HASH_MISMATCH"})


def test_version_and_dataset_mismatches_fail_closed(research_env, monkeypatch, tmp_path):
    research_dir = _copy_artifacts(research_env, tmp_path / "r")
    manifest_path = output_dir_for(research_dir) / "run_manifest.json"
    original = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(original)
    manifest["dataset"]["dataset_version"] = "1.1.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", research_dir)
    _assert_fails_closed({"VERSION_MISMATCH"})

    manifest_path.write_text(original, encoding="utf-8")
    phase1 = research_dir / "medihaven_t1_inhospital_mortality_icu24h_v1.2.0" / "manifest.json"
    dataset_manifest = json.loads(phase1.read_text(encoding="utf-8"))
    dataset_manifest["content_sha256"] = "0" * 64
    phase1.write_text(json.dumps(dataset_manifest), encoding="utf-8")
    _assert_fails_closed({"DATASET_MISMATCH"})


@pytest.mark.parametrize("change", ["drop_feature", "forbidden_feature"])
def test_feature_schema_mismatch_fails_closed(research_env, monkeypatch, tmp_path, change):
    research_dir = _copy_artifacts(research_env, tmp_path / "r")
    relative = "models/logistic_regression.joblib"
    path = output_dir_for(research_dir) / relative
    bundle = joblib.load(path)
    if change == "drop_feature":
        bundle["features"] = bundle["features"][:-1]
    else:
        bundle["features"] = ["src_subject_id"] + bundle["features"][1:]
    joblib.dump(bundle, path)
    _rehash(research_dir, relative)  # hashes consistent: the schema check itself must catch it
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", research_dir)
    _assert_fails_closed({"FEATURE_SCHEMA_MISMATCH", "FORBIDDEN_FEATURE"})


def test_manifest_pin_and_unsafe_manifest_entries(research_env, monkeypatch, tmp_path):
    manifest_file = output_dir_for(research_env) / "run_manifest.json"
    monkeypatch.setattr(Config, "RESEARCH_ARTIFACT_MANIFEST_SHA256", "f" * 64)
    _assert_fails_closed({"MANIFEST_PIN_MISMATCH"})
    monkeypatch.setattr(Config, "RESEARCH_ARTIFACT_MANIFEST_SHA256", hashlib.sha256(manifest_file.read_bytes()).hexdigest())
    assert _status(create_app({"TESTING": True}))["research_inference_available"] is True

    monkeypatch.setattr(Config, "RESEARCH_ARTIFACT_MANIFEST_SHA256", "")
    research_dir = _copy_artifacts(research_env, tmp_path / "r")
    manifest_path = output_dir_for(research_dir) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_files_sha256"]["../escape.json"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", research_dir)
    _assert_fails_closed({"UNSAFE_MANIFEST_ENTRY"})


# ------------------------------------------------------------------------------
# Launcher
# ------------------------------------------------------------------------------
def test_launcher_starts_research_server_without_self_healing(research_env, monkeypatch, capsys):
    import run_demo
    from flask import Flask
    calls = {}
    monkeypatch.setattr(run_demo, "run_preflight_checks",
                        lambda: (_ for _ in ()).throw(AssertionError("pre-flight must not run")))
    monkeypatch.setattr(Flask, "run", lambda self, **kwargs: calls.update(kwargs))
    monkeypatch.setenv("NO_BROWSER", "1")
    monkeypatch.setenv("FLASK_DEBUG", "1")  # as in the project .env; the launcher still forces the debugger off
    run_demo.main()
    assert calls["debug"] is False and calls["host"] == "127.0.0.1"
    out = capsys.readouterr().out
    assert "RESEARCH MODE" in out and "Research artifacts verified" in out


# ------------------------------------------------------------------------------
# Research dashboard (cohort-level charts)
# ------------------------------------------------------------------------------
def test_research_dashboard_endpoint_serves_aggregate_chart_data(research_env):
    client = _physician(create_app({"TESTING": True}))
    response = client.get("/api/research/dashboard", headers=ENFORCE)
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["kpis"]["stays"] == 70 and data["kpis"]["features"] == len([c for c in feature_columns() if not c.endswith("_count")])
    assert data["kpis"]["top_model"] in {"logistic_regression", "decision_tree", "random_forest", "neural_network"}
    assert set(data["models"]) == {"dummy_prior", "logistic_regression", "decision_tree", "random_forest", "neural_network"}
    assert all(len(m["per_fold_auroc"]) == 5 for m in data["models"].values())
    assert set(data["sensitivity"]) == {"primary", "sensitivity_evaluation_only", "sensitivity_retrained"}
    assert len(data["folds"]) == 5 and "test_positives" in data["folds"][0]
    weights = data["feature_weights"]
    for key in ("random_forest_importance", "logistic_regression_coefficient"):
        assert 0 < len(weights[key]) <= 10
        assert {w["feature"] for w in weights[key]} <= set(feature_columns())
    assert [v["name"] for v in data["whatif_variables"]][:3] == ["heart_rate", "resp_rate", "spo2"]
    assert data["cohort"] is None  # the synthetic fixture manifest records no cohort counts
    assert not any(identifier in response.get_data(as_text=True) for identifier in FIXTURE_IDS)


def test_research_dashboard_cohort_charts_from_phase1_manifest(research_env, monkeypatch, tmp_path):
    research_dir = _copy_artifacts(research_env, tmp_path / "r")
    phase1 = research_dir / "medihaven_t1_inhospital_mortality_icu24h_v1.2.0" / "manifest.json"
    manifest = json.loads(phase1.read_text(encoding="utf-8"))
    manifest["source"] = {"release": "mimic-iv-demo-2.2"}
    manifest["aggregate_counts"] = {
        "exclusions": {"icu_stays_in_source": 90, "not_first_icu_stay_of_admission": 8, "under_18": 0,
                       "left_icu_before_index": 12, "cohort_size": 70, "positive_labels": 14, "negative_labels": 56},
        "ambiguous_outcome_reasons": {"missing_discharge_time": 0},
        "outcome_anomalies": {"positive_deathtime_after_discharge": 3},
        "variable_coverage_fraction_of_stays": {"heart_rate": 1.0, "lactate": 0.64},
        "observation_quality": {"heart_rate": {"kept": 100, "implausible": 2, "stored_after_index": 5},
                                "lactate": {"kept": 20, "implausible": 0, "stored_after_index": 1}},
    }
    phase1.write_text(json.dumps(manifest), encoding="utf-8")  # content_sha256 unchanged, so still verified
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", research_dir)
    cohort = _physician(create_app({"TESTING": True})).get("/api/research/dashboard", headers=ENFORCE).get_json()["data"]["cohort"]
    assert [(s["excluded"], s["remaining"]) for s in cohort["flow"]] == [(0, 90), (8, 82), (0, 82), (12, 70)]
    assert cohort["outcome_anomalies"] == 3 and cohort["coverage"] == {"heart_rate": 1.0, "lactate": 0.64}
    assert cohort["data_quality"] == {"kept": 120, "dropped": {"outside_window": 0, "stored_after_index": 6,
                                                               "non_numeric": 0, "unit_mismatch": 0, "implausible": 2}}
    assert cohort["source_release"] == "mimic-iv-demo-2.2"


def test_research_dashboard_access_and_availability(research_env, monkeypatch, tmp_path):
    app = create_app({"TESTING": True})
    assert app.test_client().get("/api/research/dashboard", headers=ENFORCE).status_code == 401
    assert _login(app, "patient1", "Patient@123").get("/api/research/dashboard", headers=ENFORCE).status_code == 403
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", tmp_path / "empty")
    unavailable = _physician(create_app({"TESTING": True})).get("/api/research/dashboard", headers=ENFORCE)
    assert unavailable.status_code == 503
    monkeypatch.setattr(Config, "DATA_MODE", "synthetic")
    assert create_app({"TESTING": True}).test_client().get("/api/research/dashboard").status_code == 409


def test_research_page_is_a_chart_dashboard(research_env):
    page = _physician(create_app({"TESTING": True})).get("/research", headers=ENFORCE).get_data(as_text=True)
    for canvas in ("researchAurocChart", "researchApChart", "researchFlowChart", "researchCoverageChart",
                   "researchSensitivityChart", "researchFoldChart", "researchWeightsChart", "researchQualityChart",
                   "researchWhatIfChart"):
        assert f'id="{canvas}"' in page
    assert 'class="kpi-grid"' in page and "js/research.js" in page and "NOT FOR PATIENT CARE" in page


def test_research_script_never_injects_html_or_uses_browser_storage():
    script = (PROJECT_ROOT / "static" / "js" / "research.js").read_text(encoding="utf-8")
    import re
    assert not re.search(r"\.innerHTML\s*[+]?=", script)
    assert not re.search(r"localStorage|sessionStorage|indexedDB", script)
    assert "/api/research/dashboard" in script and "/api/research/predict" in script


# ------------------------------------------------------------------------------
# Alternate launchers
# ------------------------------------------------------------------------------
def test_research_mode_refuses_flask_cli_launches(research_env, monkeypatch):
    monkeypatch.setenv("FLASK_RUN_FROM_CLI", "true")  # set by the `flask` CLI before it loads the app
    monkeypatch.setenv("FLASK_DEBUG", "1")
    with pytest.raises(ResearchModeConfigError) as err:
        create_app({"TESTING": True})
    assert "'flask' CLI" in str(err.value) and "FLASK_DEBUG" in str(err.value)


def test_research_mode_refuses_non_loopback_flask_run_host(research_env, monkeypatch):
    monkeypatch.setenv("FLASK_RUN_HOST", "0.0.0.0")
    with pytest.raises(ResearchModeConfigError, match="FLASK_RUN_HOST"):
        create_app({"TESTING": True})
    monkeypatch.setenv("FLASK_RUN_HOST", "127.0.0.1")
    assert create_app({"TESTING": True}).research_service.available


def test_synthetic_mode_is_unaffected_by_flask_cli_settings(monkeypatch):
    monkeypatch.setenv("FLASK_RUN_FROM_CLI", "true")
    monkeypatch.setenv("FLASK_DEBUG", "1")
    monkeypatch.setenv("FLASK_RUN_HOST", "0.0.0.0")
    app = create_app({"TESTING": True})
    assert app.data_mode == "synthetic" and app.test_client().get("/api/health").status_code == 200


# ------------------------------------------------------------------------------
# Vault never shows synthetic predictions in research mode
# ------------------------------------------------------------------------------
def test_vault_profile_omits_synthetic_prediction_in_research_mode(research_env, monkeypatch):
    patient = _login(create_app({"TESTING": True}), "patient1", "Patient@123")
    chart = patient.get("/api/patients/1", headers=ENFORCE).get_json()["data"]
    assert chart["latest_prediction"] is None
    assert chart["full_name"] and "active_medications" in chart  # the rest of the vault profile is intact

    monkeypatch.setattr(Config, "DATA_MODE", "synthetic")
    synthetic = _login(create_app({"TESTING": True}), "patient1", "Patient@123")
    assert synthetic.get("/api/patients/1", headers=ENFORCE).get_json()["data"]["latest_prediction"]["risk_tier"]


def test_vault_frontend_renders_risk_only_when_a_prediction_exists():
    vault_js = (PROJECT_ROOT / "static" / "js" / "vault.js").read_text(encoding="utf-8")
    template = (PROJECT_ROOT / "templates" / "patient_vault.html").read_text(encoding="utf-8")
    assert "const pred = p.latest_prediction || {};" in vault_js
    assert "if (riskEl && pred.risk_tier) {" in vault_js  # no risk badge without a prediction
    assert '<span class="risk-badge" id="patient-dossier-risk">--</span>' in template


# ------------------------------------------------------------------------------
# Artifact robustness: unexpected problems disable inference without crashing startup
# ------------------------------------------------------------------------------
def _write_manifest(research_dir, mutate):
    path = output_dir_for(research_dir) / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(mutate(manifest) if callable(mutate) else mutate, encoding="utf-8")


@pytest.mark.parametrize("content, code", [
    ("not json at all", "MANIFEST_INVALID"),
    ("[1, 2, 3]", "MANIFEST_INVALID"),
    (lambda m: json.dumps({**m, "output_files_sha256": ["metrics.json"]}), "MANIFEST_INVALID"),
    (lambda m: json.dumps({**m, "dataset": "1.2.0"}), "MANIFEST_INVALID"),
    (lambda m: json.dumps({**m, "software": {**m["software"], "scikit-learn": "0.24.2"}}), "SKLEARN_VERSION_MISMATCH"),
])
def test_malformed_manifests_fail_closed(research_env, monkeypatch, tmp_path, content, code):
    research_dir = _copy_artifacts(research_env, tmp_path / "r")
    _write_manifest(research_dir, content)
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", research_dir)
    _assert_fails_closed({code})


def test_missing_and_corrupt_model_files_fail_closed(research_env, monkeypatch, tmp_path):
    research_dir = _copy_artifacts(research_env, tmp_path / "missing")
    (output_dir_for(research_dir) / "models" / "decision_tree.joblib").unlink()
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", research_dir)
    _assert_fails_closed({"ARTIFACT_MISSING"})

    research_dir = _copy_artifacts(research_env, tmp_path / "corrupt")
    relative = "models/decision_tree.joblib"
    (output_dir_for(research_dir) / relative).write_bytes(b"not a pickle")
    _rehash(research_dir, relative)  # hash consistent, so deserialization itself must fail safely
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", research_dir)
    _assert_fails_closed({"BUNDLE_UNLOADABLE"})


def test_version_warning_during_deserialization_fails_closed(research_env, monkeypatch):
    import warnings
    import src.training.artifacts as artifacts_module
    InconsistentVersionWarning = type("InconsistentVersionWarning", (UserWarning,), {})
    real_load = artifacts_module.joblib.load

    def load_with_warning(fileobj):
        warnings.warn("pickled with another version", InconsistentVersionWarning)
        return real_load(fileobj)

    monkeypatch.setattr(artifacts_module.joblib, "load", load_with_warning)
    _assert_fails_closed({"SKLEARN_VERSION_MISMATCH"})


def test_malformed_metrics_fail_closed(research_env, monkeypatch, tmp_path):
    research_dir = _copy_artifacts(research_env, tmp_path / "r")
    (output_dir_for(research_dir) / "metrics.json").write_text(json.dumps({"analyses": {}}), encoding="utf-8")
    _rehash(research_dir, "metrics.json")
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", research_dir)
    _assert_fails_closed({"METRICS_INVALID"})


def test_unexpected_verification_error_keeps_app_available(research_env, monkeypatch):
    import src.training.serving as serving_module
    monkeypatch.setattr(serving_module, "verify_and_load",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    status = _assert_fails_closed({"VERIFICATION_ERROR"})
    assert "boom" not in status["reason"]


def test_deserializer_receives_exactly_the_verified_bytes(research_env, monkeypatch):
    """The model file is swapped on disk after verification; loading must use the verified bytes only."""
    import io
    import src.training.artifacts as artifacts_module
    run_dir = output_dir_for(research_env)
    hashes = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["output_files_sha256"]
    real_load = artifacts_module.joblib.load
    seen = []
    target = run_dir / "models" / "random_forest.joblib"
    original = target.read_bytes()

    def spying_load(fileobj):
        assert isinstance(fileobj, io.BytesIO), "deserializer must receive in-memory verified bytes"
        data = fileobj.getvalue()
        seen.append(hashlib.sha256(data).hexdigest())
        target.write_bytes(b"swapped after verification")  # an attacker replacing the file mid-load
        return real_load(io.BytesIO(data))

    monkeypatch.setattr(artifacts_module.joblib, "load", spying_load)
    try:
        app = create_app({"TESTING": True})
    finally:
        target.write_bytes(original)
    assert app.research_service.available
    assert seen == [hashes[f"models/{m}.joblib"] for m in
                    ("dummy_prior", "logistic_regression", "decision_tree", "random_forest", "neural_network")]


def test_missing_manifest_pin_is_reported(research_env, monkeypatch, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        status = create_app({"TESTING": True}).research_service.status()
    assert status["manifest_pinned"] is False and "RESEARCH_ARTIFACT_MANIFEST_SHA256 is not set" in caplog.text
    pin = hashlib.sha256((output_dir_for(research_env) / "run_manifest.json").read_bytes()).hexdigest()
    monkeypatch.setattr(Config, "RESEARCH_ARTIFACT_MANIFEST_SHA256", pin)
    assert create_app({"TESTING": True}).research_service.status()["manifest_pinned"] is True


# ------------------------------------------------------------------------------
# Research triage: stay-level view with out-of-fold scores
# ------------------------------------------------------------------------------
def test_research_stays_have_out_of_fold_scores_and_no_identifiers(research_env):
    client = _physician(create_app({"TESTING": True}))
    response = client.get("/api/research/stays", headers=ENFORCE)
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["kpis"]["stays"] == 70 and data["kpis"]["deaths"] == 14
    assert sum(data["kpis"]["tiers"].values()) == 70
    assert [s["stay"] for s in sorted(data["stays"], key=lambda s: s["stay"])] == list(range(1, 71))
    for stay in data["stays"]:
        assert set(stay["models"]) == {"logistic_regression", "decision_tree", "random_forest", "neural_network"}
        probabilities = [m["probability"] for m in stay["models"].values()]
        assert all(0.0 < p < 1.0 for p in probabilities)
        assert stay["risk"] == pytest.approx(sum(probabilities) / 4, abs=1e-3)
        assert stay["tier"] in ("Low", "Medium", "High", "Critical")
    text = response.get_data(as_text=True)
    assert not any(identifier in text for identifier in FIXTURE_IDS)
    assert "index_time" not in text and "src_" not in text and "2150-" not in text


def test_research_stay_detail_and_access_rules(research_env, monkeypatch):
    app = create_app({"TESTING": True})
    client = _physician(app)
    detail = client.get("/api/research/stays/1", headers=ENFORCE).get_json()["data"]
    assert detail["stay"] == 1 and 1 <= detail["fold"] <= 5 and len(detail["variables"]) == 14
    assert {"min", "max", "mean", "last", "count", "measured", "unit"} <= set(detail["variables"][0])
    assert client.get("/api/research/stays/999", headers=ENFORCE).status_code == 404
    patient = _login(app, "patient1", "Patient@123")
    assert patient.get("/api/research/stays", headers=ENFORCE).status_code == 403
    assert patient.get("/api/research/stays/1", headers=ENFORCE).status_code == 403
    monkeypatch.setattr(Config, "DATA_MODE", "synthetic")
    assert create_app({"TESTING": True}).test_client().get("/api/research/stays").status_code == 409


@pytest.mark.parametrize("tamper, code", [
    (lambda m: m["analyses"]["primary"]["split"].update(assignment_sha256="0" * 64), "SPLIT_MISMATCH"),
    (lambda m: m["analyses"]["primary"]["models"]["random_forest"]["pooled_out_of_fold"].update(auroc=0.123),
     "PREDICTION_MISMATCH"),
])
def test_research_stays_fail_closed_when_scores_do_not_reproduce_the_verified_run(research_env, tamper, code):
    import copy
    from src.training.stays import LazyStayView
    app = create_app({"TESTING": True})
    service = app.research_service
    metrics = copy.deepcopy(service.artifacts.metrics)
    tamper(metrics)
    service.stays = LazyStayView(research_env, service.artifacts.features, metrics)
    response = _physician(app).get("/api/research/stays", headers=ENFORCE)
    assert response.status_code == 503 and response.get_json()["error"]["code"] == code
    assert client_text_has_no_stays(response.get_data(as_text=True))


def client_text_has_no_stays(text):
    return '"stays"' not in text and '"variables"' not in text


def test_research_triage_script_never_injects_html_or_uses_browser_storage():
    import re
    script = (PROJECT_ROOT / "static" / "js" / "research_triage.js").read_text(encoding="utf-8")
    assert not re.search(r"\.innerHTML\s*[+]?=|insertAdjacentHTML", script)
    assert not re.search(r"localStorage|sessionStorage|indexedDB", script)
    assert "/api/research/stays" in script
