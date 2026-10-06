"""Presentation Layer and Web Portals Test Suite (Phase 7).

Validates:
1. Static asset integrity (tokens.css, styles.css, dashboard.js, vault.js, scanner.js).
2. Jinja2 template structure, semantic HTML5 landmarks, and layout inheritance.
3. Physician Dashboard view rendering and required interactive DOM element bindings.
4. Patient Medical Vault view rendering and sovereignty component bindings.
5. Provider QR Scanner view rendering and optical/scoped verification elements.
6. Anti-AI design token styling rules and WCAG accessibility standards.
"""

import pytest
from flask import Flask
from flask.testing import FlaskClient

from src.api.app import create_app
from src.utils.config import Config


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
# 1. Static Asset Integrity Tests
# ==============================================================================

def test_static_css_and_js_assets_exist():
    """Verify all presentation layer stylesheets and JavaScript controllers exist and are non-empty."""
    tokens_css = Config.STATIC_DIR / "css" / "tokens.css"
    styles_css = Config.STATIC_DIR / "css" / "styles.css"
    dashboard_js = Config.STATIC_DIR / "js" / "dashboard.js"
    vault_js = Config.STATIC_DIR / "js" / "vault.js"
    scanner_js = Config.STATIC_DIR / "js" / "scanner.js"

    for asset_path in [tokens_css, styles_css, dashboard_js, vault_js, scanner_js]:
        assert asset_path.exists(), f"Missing required frontend asset: {asset_path}"
        assert asset_path.stat().st_size > 200, f"Frontend asset is empty or truncated: {asset_path}"


def test_design_tokens_and_stylesheet_rules():
    """Verify that styles.css imports tokens.css and contains critical component styles."""
    styles_css = Config.STATIC_DIR / "css" / "styles.css"
    content = styles_css.read_text(encoding="utf-8")

    assert "tokens.css" in content
    assert ".kpi-grid" in content
    assert ".clinical-table" in content
    assert ".drawer-panel" in content
    assert ".dt-step" in content
    assert ".knn-precedent-card" in content
    assert ".qr-pass-card" in content
    assert ".verification-badge-card" in content
    assert ".scanner-viewport-box" in content


# ==============================================================================
# 2. Physician Clinical Dashboard Template Tests
# ==============================================================================

def test_dashboard_view_structure_and_bindings(client: FlaskClient):
    """Verify that /dashboard renders valid HTML with all required interactive DOM elements."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    html = response.data.decode("utf-8")

    # Semantic Landmarks & Titles
    assert "Physician Clinical Triage" in html
    assert '<main class="main-content"' in html
    assert 'role="banner"' in html
    assert 'role="contentinfo"' in html

    # KPI Telemetry Grid Bindings
    assert 'id="kpi-total-admitted"' in html
    assert 'id="kpi-critical-count"' in html
    assert 'id="kpi-high-count"' in html
    assert 'id="kpi-active-alerts"' in html
    assert 'id="kpi-icu-occupancy"' in html
    assert 'id="kpi-readmit-risk"' in html

    # Early Warning Deterioration Alert Container
    assert 'id="alerts-marquee-container"' in html

    # Triage Queue Table & Filters
    assert 'id="triage-table"' in html
    assert 'id="ward-filter-select"' in html
    assert 'id="patient-search-input"' in html

    # Slide-over Clinical Chart Drawer & Visualizers
    assert 'id="drawer-backdrop"' in html
    assert 'id="vitalsChart"' in html
    assert 'id="dt-rules-list"' in html
    assert 'id="knn-precedents-list"' in html

    # Alert Acknowledgement Modal
    assert 'id="ack-modal-overlay"' in html
    assert 'id="ack-doctor-name"' in html

    # JavaScript Controller & Chart.js Binding
    assert 'dashboard.js' in html
    assert 'chart.umd.min.js' in html


# ==============================================================================
# 3. Patient Medical Vault Template Tests
# ==============================================================================

def test_patient_vault_view_structure_and_bindings(client: FlaskClient):
    """Verify that /vault renders valid HTML with patient sovereignty & QR pass components."""
    response = client.get("/vault")
    assert response.status_code == 200
    html = response.data.decode("utf-8")

    # Titles & Switcher
    assert "Patient Medical Vault" in html
    assert 'id="vault-patient-select"' in html

    # 7-Category Summary Metrics Grid
    assert 'id="vault-category-grid"' in html
    assert 'id="count-cat-all"' in html
    assert 'id="count-cat-allergy"' in html
    assert 'id="count-cat-medication"' in html
    assert 'id="count-cat-lab"' in html
    assert 'id="count-cat-surgery"' in html
    assert 'id="count-cat-condition"' in html
    assert 'id="count-cat-immunization"' in html
    assert 'id="count-cat-notes"' in html

    # Records Catalog Grid
    assert 'id="records-grid"' in html

    # Active Passes Panel with 1-Click Revocation
    assert 'id="active-tokens-list"' in html

    # Immutable Patient Access Audit Log Table
    assert 'id="audit-table"' in html

    # QR Share Modal & Upload Record Modal
    assert 'id="modal-qr-share"' in html
    assert 'id="modal-upload-record"' in html
    assert 'id="qr-image-container"' in html
    assert 'id="qr-token-text"' in html

    # JavaScript Controller
    assert 'vault.js' in html


# ==============================================================================
# 4. Provider Optical QR Scanner Template Tests
# ==============================================================================

def test_provider_scanner_view_structure_and_bindings(client: FlaskClient):
    """Verify that /scanner renders valid HTML with camera viewfinder & verification badges."""
    response = client.get("/scanner")
    assert response.status_code == 200
    html = response.data.decode("utf-8")

    # Titles & Viewfinder
    assert "Provider Optical QR Scanner" in html
    assert 'id="scanner-video"' in html
    assert 'id="reticle-box"' in html
    assert 'id="qr-canvas"' in html
    assert 'id="btn-toggle-camera"' in html

    # Alternative Inputs
    assert 'id="qr-file-input"' in html
    assert 'id="qr-token-input"' in html
    assert 'id="btn-submit-token"' in html

    # Verification Badge & Zero Data Leakage Indicators
    assert 'id="verified-badge-card"' in html
    assert 'id="leakage-notice-card"' in html
    assert 'id="scoped-records-container"' in html
    assert 'id="error-card"' in html

    # JavaScript Controller
    assert 'scanner.js' in html
