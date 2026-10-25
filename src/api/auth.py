"""Role-Based Authentication Engine for MediHaven Clinical Intelligence.

Enforces role-based access control (RBAC) across three roles:
1. Administrator: exactly three staff accounts (vidhi, nirupam, param).
2. Physician: the demo account physician1.
3. Patient: the demo account patient1, linked to Patient #1.

Only password hashes are stored. The built-in hashes are development defaults: in production
(FLASK_ENV=production) an account logs in only with a hash supplied as PASSWORD_HASH_<USERNAME>,
except the two public demo accounts when DEMO_LOGINS=1. Generate a hash with:
    python -m src.api.auth
"""

import getpass
import os
from functools import wraps
from typing import Optional, Dict, Any, List
from flask import Blueprint, current_app, request, session, jsonify, redirect, url_for, g
from werkzeug.security import check_password_hash, generate_password_hash

from src.api.rate_limit import rate_limited
from src.utils.config import Config
from src.utils.logger import get_logger
from src.api.schemas import success_response, error_response

logger = get_logger("medihaven.auth")

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

# ------------------------------------------------------------------------------
# User Credential Store (hashes of the development passwords; see the module docstring)
# Exactly three admin accounts. No other accounts possess administrator access.
# ------------------------------------------------------------------------------
CREDENTIALS: Dict[str, Dict[str, Any]] = {
    # Authorized Hospital Administrators (Restricted strictly to these three)
    "vidhi": {
        "password_hash": "scrypt:32768:8:1$pZM6FnmpJbRukZR5$a9c0dca89ec5ee637ba51dfbe02067b879a646f2f7cba413a04b71390e462396fdb9bbc6d812fbfc63db173b041ea60c9e7e79ce899dacbaea1494c19cf7863e",
        "role": "admin",
        "display_name": "Hospital Administrator",
        "email": "admin.vidhi@medihaven.internal",
    },
    "nirupam": {
        "password_hash": "scrypt:32768:8:1$PViPIHYCQMYezO1W$24af297eb800fdb9efe227867bbeedea6b5bfbe672c476be139e364aa380a29b75ff8224f7b46e83c0275c3ef6a7be3bcfe024c64e70cc1731378f900924a1a2",
        "role": "admin",
        "display_name": "Hospital Administrator",
        "email": "admin.nirupam@medihaven.internal",
    },
    "param": {
        "password_hash": "scrypt:32768:8:1$TfZP6qANurzdU3eh$6ac3932d1e95f9961e4483fbdd2a3d36147f8a034827aee78d83e6d4cf2eaa392d9bf20039bc172b9ebdd3b7b7883f433f38cbfd1e72bacecacb5f10974586b0",
        "role": "admin",
        "display_name": "Hospital Administrator",
        "email": "admin.param@medihaven.internal",
    },
    # Clinical Physician Demo Account
    "physician1": {
        "password_hash": "scrypt:32768:8:1$AJG0tADxjsm5NhQo$5c894cb5e4ed262be002d2e1e8d69eafaedb500f8db1efc2334d7f98591a09e15c9de71419f9199192d1eb85de384a6358a980dbb318d9b63642da99fcc2f52b",
        "role": "physician",
        "display_name": "Dr. Sarah Chen, MD (Attending)",
        "department": "Cardiology & Intensive Care",
        "email": "sarah.chen@medihaven.org",
    },
    # Inpatient Medical Vault Demo Account (Linked to Patient 1)
    "patient1": {
        "password_hash": "scrypt:32768:8:1$ScLkms9EYzO33h0s$c6932a9284bb5ce43c8486048b953e4d3563a6c2cdbde664d3fcd6effcfff03dcb55c4b247f84fc3f65e9224976c83579c6f48a33a3549a8fa7566013c1e585a",
        "role": "patient",
        "display_name": "Aarav Verma",
        "patient_id": 1,
        "mrn": "MRN-2026-0001",
        "email": "aarav.verma@patient.medihaven.org",
    },
}

CLINICAL_ROLES = ["physician", "admin"]

# Public demo accounts whose development passwords are shown on the login page when DEMO_LOGINS is on
DEMO_ACCOUNTS = ("physician1", "patient1")

ROLE_DASHBOARD_MAP = {
    "admin": "/admin",
    "physician": "/dashboard",
    "patient": "/vault",
}

# Checked when the username is unknown, so response time does not reveal which usernames exist
_UNKNOWN_USER_HASH = generate_password_hash("medihaven-unknown-user")


def password_hash_for(username: str) -> Optional[str]:
    """The hash an account may log in with, or None if the account is disabled in this configuration."""
    override = os.getenv(f"PASSWORD_HASH_{username.upper()}", "").strip()
    if override:
        return override
    if not Config.IS_PRODUCTION or (Config.DEMO_LOGINS and username in DEMO_ACCOUNTS):
        return CREDENTIALS[username]["password_hash"]
    return None  # production never accepts the development password of an admin account


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Validates submitted username and password against the credential store."""
    if not username or not password:
        return None

    clean_user = username.strip().lower()
    user_record = CREDENTIALS.get(clean_user)
    expected = password_hash_for(clean_user) if user_record else None

    if not check_password_hash(expected or _UNKNOWN_USER_HASH, password.strip()) or not expected:
        return None

    return {
        "username": clean_user,
        "role": user_record["role"],
        "display_name": user_record["display_name"],
        "patient_id": user_record.get("patient_id"),
        "mrn": user_record.get("mrn"),
        "department": user_record.get("department"),
    }


def get_current_user() -> Optional[Dict[str, Any]]:
    """Returns the currently authenticated user from session or None."""
    return session.get("user")


def acting_user_name(supplied: Any, default: str) -> str:
    """Who performed an action, for audit trails: the signed-in user, never a name typed by the client."""
    user = get_current_user()
    if user:
        return user.get("display_name") or user.get("username") or default
    # Only reachable when unit tests bypass authentication (role_required always demands a session)
    return supplied.strip()[:200] if isinstance(supplied, str) and supplied.strip() else default


def _auth_bypassed() -> bool:
    """Mirrors the portal guard: unit tests skip RBAC unless they opt in via X-Enforce-Auth."""
    return bool(current_app.config.get("TESTING")) and not request.headers.get("X-Enforce-Auth")


def patient_access_denied(patient_id: Any):
    """Returns a 403 response if a patient-role user targets another patient's records, else None."""
    if _auth_bypassed():
        return None
    user = get_current_user() or {}
    if user.get("role") != "patient":
        return None
    try:
        requested = int(patient_id)
    except (TypeError, ValueError):
        requested = None
    if requested != user.get("patient_id"):
        return error_response(
            message="Access denied: patients may only access their own medical vault.",
            code="FORBIDDEN",
            status_code=403,
        )
    return None


def role_required(allowed_roles: List[str]):
    """Decorator for route handlers that restricts access to specified roles."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if _auth_bypassed():
                return fn(*args, **kwargs)
            user = get_current_user()
            if not user:
                if request.is_json or request.path.startswith("/api/"):
                    return error_response(
                        message="Authentication required.",
                        code="UNAUTHORIZED",
                        status_code=401,
                    )
                return redirect(url_for("view_login", next=request.path, error="signin"))

            if user.get("role") not in allowed_roles:
                if request.is_json or request.path.startswith("/api/"):
                    return error_response(
                        message="Access denied: insufficient role privileges.",
                        code="FORBIDDEN",
                        status_code=403,
                    )
                # Redirect user to their own role dashboard with notice
                own_dashboard = ROLE_DASHBOARD_MAP.get(user.get("role"), "/")
                return redirect(url_for("view_login", error="role"))

            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ------------------------------------------------------------------------------
# REST API Authentication Endpoints
# ------------------------------------------------------------------------------
@auth_bp.route("/login", methods=["POST"])
@rate_limited("login", limit=10, window_seconds=300)
def api_login():
    """Handles JSON and form-encoded authentication requests."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = request.form
    username = str(data.get("username") or "").strip()
    password = str(data.get("password") or "").strip()

    user = authenticate_user(username, password)
    if not user:
        return error_response(
            message="Invalid credentials. Please verify your username and password.",
            code="INVALID_CREDENTIALS",
            status_code=401,
        )

    # Set session
    session["user"] = user
    session.permanent = True

    redirect_url = ROLE_DASHBOARD_MAP.get(user["role"], "/")
    logger.info(f"User '{user['username']}' successfully logged in with role '{user['role']}'")

    return success_response(
        data={
            "user": user,
            "role": user["role"],
            "redirect_url": redirect_url,
        },
        message="Authentication successful.",
    )


@auth_bp.route("/logout", methods=["POST", "GET"])
def api_logout():
    """Clears the active session and logs out user."""
    username = session.get("user", {}).get("username", "anonymous")
    session.clear()
    logger.info(f"User '{username}' logged out.")
    if request.is_json:
        return success_response(data={"redirect_url": "/login"}, message="Logged out successfully.")
    return redirect(url_for("view_login"))


@auth_bp.route("/me", methods=["GET"])
def api_me():
    """Returns the active user profile from session."""
    user = get_current_user()
    if not user:
        return error_response(message="Not authenticated.", code="UNAUTHORIZED", status_code=401)
    return success_response(data=user)


if __name__ == "__main__":
    # Prints a hash for PASSWORD_HASH_<USERNAME>; the password itself is never echoed or stored
    first = getpass.getpass("New password: ")
    if len(first) < 12 or first != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords must match and be at least 12 characters long.")
    print(generate_password_hash(first))
