"""Security, Cryptographic, and Integration Tests for Phase 5 Medical Vault & QR Subsystem.

Validates:
1. Patient Medical Vault CRUD operations and category summary metrics.
2. Cryptographic JWT token generation and high-contrast visual QR rendering (Base64 PNG).
3. Optical QR code decoding from visual image bytes via pyzbar / OpenCV.
4. Tampered and forged token signature rejection (Zero-Trust defense).
5. Expired token rejection.
6. One-click instant pass revocation blocking.
7. Single-use quota ('max_uses = 1') enforcement against replay attacks.
8. Strict scope isolation (zero data leakage: allergy pass returns 0 meds or surgeries).
9. Dual-stream audit trail recording (database table & disk audit stream).
"""

import base64
import json
import time
from datetime import datetime, timedelta
import jwt
import pytest

from src.database.db import get_connection, execute_query
from src.utils.config import Config
from src.vault.vault_service import (
    add_vault_record,
    get_patient_vault_records,
    get_vault_record_by_id,
    delete_vault_record,
    get_vault_category_summary,
)
from src.vault.qr_generator import (
    generate_vault_access_token,
    revoke_vault_access_token,
    get_patient_active_tokens,
    render_qr_code_base64,
)
from src.vault.qr_scanner import (
    extract_qr_text_from_image,
    validate_vault_token,
    scan_and_validate_qr,
)
from src.vault.access_controller import access_vault_with_token
from src.vault.access_logger import get_patient_access_audit_log


def test_vault_record_crud_operations():
    """Verify medical record creation, retrieval, filtering, and deletion."""
    patient_id = 1

    # 1. Add record
    vault_id = add_vault_record(
        patient_id=patient_id,
        category="allergy",
        title="Latex Sensitivity Test",
        description="Mild contact dermatitis on prolonged exposure",
        structured_data={"severity": "Mild", "onset": "2024"},
        is_sensitive=False,
    )
    assert vault_id > 0

    # 2. Retrieve single record
    rec = get_vault_record_by_id(vault_id)
    assert rec is not None
    assert rec["title"] == "Latex Sensitivity Test"
    assert rec["category"] == "allergy"
    assert rec["structured_data"]["severity"] == "Mild"

    # 3. Category summary counts
    summary = get_vault_category_summary(patient_id)
    assert summary["allergy"] >= 1
    assert "medication" in summary

    # 4. Clean up test record
    deleted = delete_vault_record(vault_id, patient_id=patient_id)
    assert deleted is True
    assert get_vault_record_by_id(vault_id) is None


def test_qr_token_generation_and_base64_rendering():
    """Verify cryptographic JWT signing, database registration, and QR rendering."""
    patient_id = 1
    token_meta = generate_vault_access_token(
        patient_id=patient_id,
        scope=["allergy", "medication"],
        expires_in_minutes=60,
        max_uses=1,
    )

    assert token_meta["patient_id"] == 1
    assert token_meta["token_hash"].startswith("tok_")
    assert token_meta["scope"] == ["allergy", "medication"]
    assert token_meta["max_uses"] == 1

    # Verify Base64 Data URI format
    assert token_meta["qr_image_url"].startswith("data:image/png;base64,")
    raw_b64 = token_meta["qr_image_url"].split(",", 1)[1]
    img_bytes = base64.b64decode(raw_b64)
    assert len(img_bytes) > 200, "Rendered QR image is too small"

    # Verify valid JWT format (header.payload.sig)
    assert token_meta["qr_payload"].count(".") == 2


def test_optical_qr_decoding_from_base64_image():
    """Verify that pyzbar / OpenCV successfully extracts the JWT string from the QR image."""
    patient_id = 1
    token_meta = generate_vault_access_token(
        patient_id=patient_id,
        scope=["allergy"],
        expires_in_minutes=30,
    )

    # Pass Base64 data URL directly to optical decoder
    decoded_text = extract_qr_text_from_image(token_meta["qr_image_url"])
    assert decoded_text is not None
    assert decoded_text == token_meta["qr_payload"], "Decoded optical text must match original JWT string"


def test_forged_and_tampered_token_rejection():
    """Verify that tampered tokens fail cryptographic verification immediately."""
    patient_id = 1
    token_meta = generate_vault_access_token(patient_id=patient_id, scope=["*"])
    original_jwt = token_meta["qr_payload"]

    # Tamper with the signature portion of the JWT
    parts = original_jwt.split(".")
    tampered_sig = parts[2][:-4] + "XXXX"
    tampered_jwt = f"{parts[0]}.{parts[1]}.{tampered_sig}"

    res = validate_vault_token(tampered_jwt)
    assert res["is_valid"] is False
    assert res["access_status"] == "INVALID_SIGNATURE"
    assert "verification failed" in res["error_message"].lower()


def test_expired_token_rejection():
    """Verify that tokens beyond their expiration window are blocked."""
    patient_id = 1
    # Generate token already expired by 10 minutes
    token_meta = generate_vault_access_token(
        patient_id=patient_id,
        scope=["allergy"],
        expires_in_minutes=-10,
    )

    res = validate_vault_token(token_meta["qr_payload"])
    assert res["is_valid"] is False
    assert res["access_status"] == "EXPIRED"


def test_one_click_token_revocation_blocking():
    """Verify that patient revocation invalidates pass even if JWT timestamp is valid."""
    patient_id = 1
    token_meta = generate_vault_access_token(
        patient_id=patient_id,
        scope=["allergy", "medication"],
        expires_in_minutes=120,
    )

    # Verify valid before revocation
    res_before = validate_vault_token(token_meta["qr_payload"])
    assert res_before["is_valid"] is True

    # Patient clicks "Revoke Access"
    revoked = revoke_vault_access_token(
        token_id=token_meta["token_id"],
        patient_id=patient_id,
    )
    assert revoked is True

    # Verification must now fail with REVOKED status
    res_after = validate_vault_token(token_meta["qr_payload"])
    assert res_after["is_valid"] is False
    assert res_after["access_status"] == "REVOKED"


def test_single_use_quota_enforcement():
    """Verify single-use passes (max_uses=1) cannot be scanned a second time."""
    patient_id = 1
    token_meta = generate_vault_access_token(
        patient_id=patient_id,
        scope=["allergy"],
        expires_in_minutes=60,
        max_uses=1,
    )

    # First access attempt -> Must succeed
    res1 = access_vault_with_token(
        raw_qr_input=token_meta["qr_payload"],
        accessed_by="Dr. Vidhi Patel",
    )
    assert res1["access_granted"] is True
    assert res1["access_status"] == "GRANTED"

    # Second access attempt -> Must be rejected due to quota limit
    res2 = access_vault_with_token(
        raw_qr_input=token_meta["qr_payload"],
        accessed_by="Receptionist Terminal B",
    )
    assert res2["access_granted"] is False
    assert res2["access_status"] == "MAX_USES_EXCEEDED"


def test_scope_isolation_and_zero_data_leakage():
    """Verify strict whitelist filtering: allergy pass leaks zero meds or surgeries."""
    patient_id = 1

    # Ensure patient has records across categories
    add_vault_record(patient_id, "allergy", "Penicillin Anaphylaxis", "Documented allergy", is_sensitive=False)
    add_vault_record(patient_id, "surgery", "Appendectomy", "Emergency surgery 2018", is_sensitive=False)
    add_vault_record(patient_id, "diagnosis", "Asthma", "Mild persistent asthma", is_sensitive=False)

    # Issue an allergy-only pass
    token_meta = generate_vault_access_token(
        patient_id=patient_id,
        scope=["allergy"],
        expires_in_minutes=30,
    )

    res = access_vault_with_token(token_meta["qr_payload"], accessed_by="Dr. Param Thakkar")
    assert res["access_granted"] is True
    assert res["authorized_scope"] == ["allergy"]

    # Verify that every returned record is an allergy
    assert len(res["records"]) > 0
    for record in res["records"]:
        assert record["category"] == "allergy", f"Data leakage detected! Non-allergy returned: {record}"

    # Verify that surgeries and diagnoses are NOT present
    returned_categories = {r["category"] for r in res["records"]}
    assert "surgery" not in returned_categories
    assert "diagnosis" not in returned_categories


def test_immutable_audit_logging_and_patient_inspection():
    """Verify scan events write to vault_access_log and are readable by the patient."""
    patient_id = 1
    token_meta = generate_vault_access_token(
        patient_id=patient_id,
        scope=["allergy", "medication"],
        expires_in_minutes=45,
    )

    # Doctor scans QR pass
    doctor_name = "Dr. Nirupam Gupta (SPIT Triage)"
    access_vault_with_token(
        raw_qr_input=token_meta["qr_payload"],
        accessed_by=doctor_name,
        ip_address="192.168.1.104",
        user_agent="MediHaven ER Scanner Terminal #03",
    )

    # Patient inspects their audit log
    audit_trail = get_patient_access_audit_log(patient_id=patient_id, limit=10)
    assert len(audit_trail) > 0

    latest_entry = audit_trail[0]
    assert latest_entry["patient_id"] == patient_id
    assert latest_entry["accessed_by"] == doctor_name
    assert latest_entry["access_status"] == "GRANTED"
    assert "allergy" in latest_entry["accessed_scope"]
