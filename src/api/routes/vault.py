"""Patient Medical Vault and Cryptographic QR Access Routes for MediHaven REST API.

Provides endpoints for patient record uploads, dynamic QR pass issuance,
1-click pass revocation, optical/string QR scanning, and patient access audit logs.
"""

from typing import Dict, List, Optional, Any
from flask import Blueprint, request, jsonify

from src.api.schemas import success_response, error_response, validate_required_fields
from src.vault.vault_service import (
    add_vault_record,
    get_patient_vault_records,
    get_vault_category_summary,
    delete_vault_record,
    VALID_VAULT_CATEGORIES,
)
from src.vault.qr_generator import (
    generate_vault_access_token,
    revoke_vault_access_token,
    get_patient_active_tokens,
)
from src.vault.access_controller import access_vault_with_token
from src.vault.access_logger import get_patient_access_audit_log
from src.utils.logger import get_logger

logger = get_logger("medihaven.api.vault")

vault_bp = Blueprint("vault", __name__, url_prefix="/api/vault")


@vault_bp.route("/upload", methods=["POST"])
def upload_vault_record():
    """Adds a new medical record to a patient's Medical Vault."""
    payload = request.get_json(silent=True) or {}
    err = validate_required_fields(payload, ["patient_id", "category", "title"])
    if err:
        return error_response(message=err, code="VALIDATION_ERROR", status_code=422)

    cat = payload["category"].lower().strip()
    if cat not in VALID_VAULT_CATEGORIES:
        return error_response(
            message=f"Invalid category '{cat}'. Allowed: {sorted(VALID_VAULT_CATEGORIES)}",
            code="INVALID_CATEGORY",
            status_code=422,
        )

    try:
        vault_id = add_vault_record(
            patient_id=int(payload["patient_id"]),
            category=cat,
            title=payload["title"],
            description=payload.get("description"),
            structured_data=payload.get("structured_data"),
            file_path=payload.get("file_path"),
            is_sensitive=bool(payload.get("is_sensitive", False)),
        )
        return success_response(
            data={"vault_id": vault_id, "category": cat, "title": payload["title"]},
            message="Medical record added to vault successfully.",
            status_code=201,
        )
    except Exception as e:
        logger.error(f"Error adding vault record: {e}", exc_info=True)
        return error_response(message=f"Database error: {e}", code="VAULT_ERROR", status_code=500)


@vault_bp.route("/<int:patient_id>", methods=["GET"])
def get_vault(patient_id: int):
    """Retrieves all medical records and category summary metrics for a patient."""
    category = request.args.get("category", "").strip() or None

    try:
        records = get_patient_vault_records(patient_id=patient_id, category=category)
        summary = get_vault_category_summary(patient_id=patient_id)

        return success_response(
            data={
                "patient_id": patient_id,
                "category_summary": summary,
                "records_count": len(records),
                "records": records,
            }
        )
    except Exception as e:
        return error_response(message=str(e), code="VAULT_FETCH_ERROR", status_code=500)


@vault_bp.route("/<int:patient_id>/share", methods=["POST"])
def generate_share_token(patient_id: int):
    """Generates a dynamic cryptographic QR access pass with scoped permissions."""
    payload = request.get_json(silent=True) or {}
    scope = payload.get("scope", ["*"])
    expires_in_minutes = payload.get("expires_in_minutes", 1440)
    max_uses = payload.get("max_uses")

    try:
        token_data = generate_vault_access_token(
            patient_id=patient_id,
            scope=scope,
            expires_in_minutes=expires_in_minutes,
            max_uses=max_uses,
        )
        # Convenience aliases for frontend clients
        token_data["token"] = token_data["qr_payload"]
        token_data["qr_code_base64"] = token_data["qr_image_url"]

        return success_response(
            data=token_data,
            message="Cryptographic QR access pass generated successfully.",
            status_code=201,
        )
    except ValueError as e:
        return error_response(message=str(e), code="PATIENT_NOT_FOUND", status_code=404)
    except Exception as e:
        logger.error(f"QR generation error: {e}", exc_info=True)
        return error_response(message=f"Token error: {e}", code="QR_GEN_ERROR", status_code=500)


@vault_bp.route("/revoke/<int:token_id>", methods=["POST"])
def revoke_token(token_id: int):
    """Revokes an issued QR pass immediately (1-click patient privacy action)."""
    payload = request.get_json(silent=True) or {}
    patient_id = payload.get("patient_id")

    revoked = revoke_vault_access_token(token_id=token_id, patient_id=patient_id)
    if not revoked:
        return error_response(
            message=f"Token #{token_id} not found or could not be revoked.",
            code="TOKEN_NOT_FOUND",
            status_code=404,
        )

    return success_response(
        data={"token_id": token_id, "revoked": True},
        message=f"Access pass #{token_id} has been revoked immediately.",
    )


@vault_bp.route("/access", methods=["POST"])
def access_vault():
    """Provider QR scanner endpoint: validates token and returns strictly authorized data."""
    payload = request.get_json(silent=True) or {}
    qr_payload = payload.get("qr_payload") or payload.get("token")
    if not qr_payload:
        return error_response(
            message="Missing 'qr_payload' or 'token' in request body.",
            code="VALIDATION_ERROR",
            status_code=422,
        )

    accessed_by = payload.get("accessed_by", "Consulting Clinician").strip()
    client_ip = request.remote_addr or "127.0.0.1"
    user_agent = request.headers.get("User-Agent", "MediHaven Provider Scanner")

    res = access_vault_with_token(
        raw_qr_input=qr_payload,
        accessed_by=accessed_by,
        ip_address=client_ip,
        user_agent=user_agent,
    )

    if not res.get("access_granted"):
        return error_response(
            message=res.get("error_message", "QR Access Denied"),
            code=res.get("access_status", "ACCESS_DENIED"),
            status_code=403,
        )

    return success_response(
        data=res,
        message="Cryptographic verification successful. Scoped medical history retrieved.",
    )


@vault_bp.route("/<int:patient_id>/access-log", methods=["GET"])
def get_access_log(patient_id: int):
    """Retrieves chronological access audit log for patient inspection."""
    limit = request.args.get("limit", default=50, type=int)
    audit_trail = get_patient_access_audit_log(patient_id=patient_id, limit=limit)

    return success_response(
        data=audit_trail,
        meta={"patient_id": patient_id, "entries": len(audit_trail)},
    )


@vault_bp.route("/<int:patient_id>/active-tokens", methods=["GET"])
def get_active_tokens(patient_id: int):
    """Retrieves all active, non-expired, unrevoked tokens for a patient."""
    tokens = get_patient_active_tokens(patient_id=patient_id)
    return success_response(
        data=tokens,
        meta={"patient_id": patient_id, "active_count": len(tokens)},
    )
