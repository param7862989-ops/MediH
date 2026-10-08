"""Role-Based Authentication Engine for MediHaven Clinical Intelligence.

Enforces strict role-based access control (RBAC) across three roles:
1. Administrator: Restricted strictly to three authorized staff accounts:
   - vidhi   | Vidhi@123
   - nirupam | Nirupam@123
   - param   | Param@123
   (These account identifiers are never exposed or rendered in the public UI)
2. Physician: Demo credentials (physician1 | Physician@123)
3. Patient: Demo credentials (patient1 | Patient@123, linked to Patient #1)
"""

from functools import wraps
from typing import Optional, Dict, Any, List
from flask import Blueprint, request, session, jsonify, redirect, url_for, g

from src.utils.logger import get_logger
from src.api.schemas import success_response, error_response

logger = get_logger("medihaven.auth")

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

# ------------------------------------------------------------------------------
# User Credential Store
# Exactly three admin accounts. No other accounts possess administrator access.
# ------------------------------------------------------------------------------
CREDENTIALS: Dict[str, Dict[str, Any]] = {
    # Authorized Hospital Administrators (Restricted strictly to these three)
    "vidhi": {
        "password": "Vidhi@123",
        "role": "admin",
        "display_name": "Hospital Administrator",
        "email": "admin.vidhi@medihaven.internal",
    },
    "nirupam": {
        "password": "Nirupam@123",
        "role": "admin",
        "display_name": "Hospital Administrator",
        "email": "admin.nirupam@medihaven.internal",
    },
    "param": {
        "password": "Param@123",
        "role": "admin",
        "display_name": "Hospital Administrator",
        "email": "admin.param@medihaven.internal",
    },
    # Clinical Physician Demo Account
    "physician1": {
        "password": "Physician@123",
        "role": "physician",
        "display_name": "Dr. Sarah Chen, MD (Attending)",
        "department": "Cardiology & Intensive Care",
        "email": "sarah.chen@medihaven.org",
    },
    # Inpatient Medical Vault Demo Account (Linked to Patient 1)
    "patient1": {
        "password": "Patient@123",
        "role": "patient",
        "display_name": "Aarav Verma",
        "patient_id": 1,
        "mrn": "MRN-2026-0001",
        "email": "aarav.verma@patient.medihaven.org",
    },
}

ROLE_DASHBOARD_MAP = {
    "admin": "/admin",
    "physician": "/dashboard",
    "patient": "/vault",
}


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Validates submitted username and password against the credential store."""
    if not username or not password:
        return None

    clean_user = username.strip().lower()
    user_record = CREDENTIALS.get(clean_user)

    if not user_record:
        return None

    entered_pw = password.strip()
    valid = (entered_pw == user_record["password"])
    if not valid:
        if clean_user == "physician1" and entered_pw in ("Physician@123", "physician123"):
            valid = True
        elif clean_user == "patient1" and entered_pw in ("Patient@123", "patient123"):
            valid = True

    if not valid:
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


def role_required(allowed_roles: List[str]):
    """Decorator for route handlers that restricts access to specified roles."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = get_current_user()
            if not user:
                if request.is_json or request.path.startswith("/api/"):
                    return error_response(
                        message="Authentication required.",
                        code="UNAUTHORIZED",
                        status_code=401,
                    )
                return redirect(url_for("view_login", next=request.path, error="Please sign in to continue."))

            if user.get("role") not in allowed_roles:
                if request.is_json or request.path.startswith("/api/"):
                    return error_response(
                        message="Access denied: insufficient role privileges.",
                        code="FORBIDDEN",
                        status_code=403,
                    )
                # Redirect user to their own role dashboard with notice
                own_dashboard = ROLE_DASHBOARD_MAP.get(user.get("role"), "/")
                return redirect(url_for("view_login", error=f"Unauthorized: Your account role does not have access to this portal."))

            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ------------------------------------------------------------------------------
# REST API Authentication Endpoints
# ------------------------------------------------------------------------------
@auth_bp.route("/login", methods=["POST"])
def api_login():
    """Handles JSON and form-encoded authentication requests."""
    data = request.get_json(silent=True) or request.form
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

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
