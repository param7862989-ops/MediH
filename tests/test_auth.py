"""Role-Based Authentication and Access Control Test Suite.

Verifies:
1. Authentication of all three strict Administrator accounts:
   - vidhi   | Vidhi@123
   - nirupam | Nirupam@123
   - param   | Param@123
2. Rejection of any other unauthorized administrative accounts.
3. Authentication of clinical demo accounts:
   - physician1 | Physician@123
   - patient1   | Patient@123
4. Rejection of bad credentials.
5. Role-based routing and protection:
   - Admin accounts -> /admin only (blocked from physician/patient dashboards)
   - Physician account -> /dashboard only (blocked from admin/patient dashboards)
   - Patient account -> /vault only (blocked from admin/physician dashboards)
6. Redirection of unauthenticated requests to /login.
7. Verification that admin usernames (vidhi, nirupam, param) are NOT exposed in any public UI templates.
"""

import pytest
from flask import Flask
from flask.testing import FlaskClient

from src.api.app import create_app


@pytest.fixture(scope="module")
def app() -> Flask:
    """Creates a configured Flask application for testing."""
    test_app = create_app({"TESTING": True})
    return test_app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Provides a fresh test client for each test."""
    return app.test_client()


# ==============================================================================
# 1. Admin Account Authentication Tests
# ==============================================================================

def test_admin_accounts_login_success(client: FlaskClient):
    """Verify that exactly the three authorized admin accounts can log in."""
    admin_accounts = [
        ("vidhi", "Vidhi@123"),
        ("nirupam", "Nirupam@123"),
        ("param", "Param@123"),
    ]

    for username, password in admin_accounts:
        response = client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
        )
        assert response.status_code == 200, f"Failed login for {username}"
        data = response.get_json()
        assert data["success"] is True
        assert data["data"]["role"] == "admin"
        assert data["data"]["redirect_url"] == "/admin"
        assert data["data"]["user"]["username"] == username


def test_other_admin_usernames_rejected(client: FlaskClient):
    """Verify that no other user can gain admin access."""
    unauthorized_admins = [
        ("admin", "Admin@123"),
        ("root", "Root@123"),
        ("superuser", "Super@123"),
        ("hospital_admin", "Hospital@123"),
    ]

    for username, password in unauthorized_admins:
        response = client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
        )
        assert response.status_code == 401
        data = response.get_json()
        assert data["success"] is False
        assert data["error"]["code"] == "INVALID_CREDENTIALS"


# ==============================================================================
# 2. Physician and Patient Authentication Tests
# ==============================================================================

def test_physician_login_success(client: FlaskClient):
    """Verify physician demo credentials."""
    response = client.post(
        "/api/auth/login",
        json={"username": "physician1", "password": "Physician@123"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["data"]["role"] == "physician"
    assert data["data"]["redirect_url"] == "/dashboard"


def test_patient_login_success(client: FlaskClient):
    """Verify patient demo credentials."""
    response = client.post(
        "/api/auth/login",
        json={"username": "patient1", "password": "Patient@123"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["data"]["role"] == "patient"
    assert data["data"]["redirect_url"] == "/vault"
    assert data["data"]["user"]["patient_id"] == 1


def test_bad_password_fails(client: FlaskClient):
    """Verify invalid password produces HTTP 401."""
    response = client.post(
        "/api/auth/login",
        json={"username": "vidhi", "password": "WrongPassword!"},
    )
    assert response.status_code == 401
    data = response.get_json()
    assert data["success"] is False


# ==============================================================================
# 3. Role-Based URL Access & Protection Tests
# ==============================================================================

def test_admin_cannot_access_other_dashboards(client: FlaskClient):
    """Verify Admin is redirected away from Physician and Patient dashboards to /admin."""
    # Login as admin
    client.post("/api/auth/login", json={"username": "vidhi", "password": "Vidhi@123"})

    headers = {"X-Enforce-Auth": "1"}

    # Admin accessing /admin -> Allowed (200)
    admin_res = client.get("/admin", headers=headers)
    assert admin_res.status_code == 200

    # Admin accessing /dashboard -> Blocked and redirected to /admin
    dash_res = client.get("/dashboard", headers=headers, follow_redirects=False)
    assert dash_res.status_code == 302
    assert dash_res.headers["Location"].endswith("/admin")

    # Admin accessing /vault -> Blocked and redirected to /admin
    vault_res = client.get("/vault", headers=headers, follow_redirects=False)
    assert vault_res.status_code == 302
    assert vault_res.headers["Location"].endswith("/admin")


def test_physician_cannot_access_other_dashboards(client: FlaskClient):
    """Verify Physician is redirected away from Admin and Patient dashboards to /dashboard."""
    # Login as physician
    client.post("/api/auth/login", json={"username": "physician1", "password": "Physician@123"})

    headers = {"X-Enforce-Auth": "1"}

    # Physician accessing /dashboard -> Allowed (200)
    dash_res = client.get("/dashboard", headers=headers)
    assert dash_res.status_code == 200

    # Physician accessing /admin -> Blocked and redirected to /dashboard
    admin_res = client.get("/admin", headers=headers, follow_redirects=False)
    assert admin_res.status_code == 302
    assert admin_res.headers["Location"].endswith("/dashboard")

    # Physician accessing /vault -> Blocked and redirected to /dashboard
    vault_res = client.get("/vault", headers=headers, follow_redirects=False)
    assert vault_res.status_code == 302
    assert vault_res.headers["Location"].endswith("/dashboard")


def test_patient_cannot_access_other_dashboards(client: FlaskClient):
    """Verify Patient is redirected away from Admin and Physician dashboards to /vault."""
    # Login as patient
    client.post("/api/auth/login", json={"username": "patient1", "password": "Patient@123"})

    headers = {"X-Enforce-Auth": "1"}

    # Patient accessing /vault -> Allowed (200)
    vault_res = client.get("/vault", headers=headers)
    assert vault_res.status_code == 200

    # Patient accessing /admin -> Blocked and redirected to /vault
    admin_res = client.get("/admin", headers=headers, follow_redirects=False)
    assert admin_res.status_code == 302
    assert admin_res.headers["Location"].endswith("/vault")

    # Patient accessing /dashboard -> Blocked and redirected to /vault
    dash_res = client.get("/dashboard", headers=headers, follow_redirects=False)
    assert dash_res.status_code == 302
    assert dash_res.headers["Location"].endswith("/vault")


def test_unauthenticated_user_redirected_to_login(client: FlaskClient):
    """Verify that unauthenticated user trying to access any dashboard is redirected to /login."""
    headers = {"X-Enforce-Auth": "1"}

    for path in ["/dashboard", "/vault", "/admin"]:
        res = client.get(path, headers=headers, follow_redirects=False)
        assert res.status_code == 302
        assert "/login" in res.headers["Location"]


# ==============================================================================
# 4. Public UI Anonymity Test (Admin account privacy)
# ==============================================================================

def test_admin_usernames_never_exposed_in_public_ui(client: FlaskClient):
    """Verify that vidhi, nirupam, and param are not rendered in public HTML templates."""
    import re
    pages = ["/", "/login", "/dashboard", "/vault", "/admin"]

    for page in pages:
        res = client.get(page)
        assert res.status_code == 200
        html = res.data.decode("utf-8").lower()
        # Ensure none of the admin usernames appear as words in the public UI
        assert not re.search(r"\bvidhi\b", html), f"Found 'vidhi' in {page} HTML!"
        assert not re.search(r"\bnirupam\b", html), f"Found 'nirupam' in {page} HTML!"
        assert not re.search(r"\bparam\b", html), f"Found 'param' in {page} HTML!"

