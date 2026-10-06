"""Cryptographic QR Code Generator and Vault Access Token Service for MediHaven.

Generates HMAC-SHA256 signed JWT tokens embedded with scoped clinical claims
and renders high-contrast, visually scannable QR code images as Base64 Data URIs.
"""

import base64
import io
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import jwt
import qrcode
from PIL import Image
import sqlite3

from src.database.db import get_connection, get_db_connection
from src.utils.config import Config
from src.utils.logger import get_logger, get_audit_logger

logger = get_logger("medihaven.vault.qr_generator")
audit_logger = get_audit_logger("medihaven.vault.audit")


def render_qr_code_base64(payload_text: str) -> str:
    """Renders a text payload into a Base64-encoded PNG Data URI.

    Args:
        payload_text: Text payload (signed JWT string) to encode into QR matrix.

    Returns:
        Data URI string formatted as 'data:image/png;base64,...'.
    """
    qr = qrcode.QRCode(
        version=None,  # Auto-size matrix
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=2,
    )
    qr.add_data(payload_text)
    qr.make(fit=True)

    # Render image with high clinical contrast
    img = qr.make_image(fill_color="#0F172A", back_color="#FFFFFF")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def generate_vault_access_token(
    patient_id: int,
    scope: Optional[List[str]] = None,
    expires_in_minutes: Optional[int] = None,
    max_uses: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """Generates a cryptographically signed QR access token and records it in the database.

    Args:
        patient_id: Patient database ID granting access.
        scope: List of authorized categories (e.g., ['allergy', 'medication'] or ['*']).
        expires_in_minutes: Token validity window (defaults to Config.DEFAULT_TOKEN_TTL_MINUTES).
        max_uses: Optional scan quota (1 for single-use appointment, None for unlimited).
        conn: Optional active database connection.

    Returns:
        Dictionary containing token metadata, signed JWT payload, and Base64 QR code image.
    """
    ttl_minutes = expires_in_minutes or Config.DEFAULT_TOKEN_TTL_MINUTES
    now = datetime.now()
    issued_at = now
    expires_at = now + timedelta(minutes=ttl_minutes)

    # Clean and normalize scope
    if not scope or "*" in scope:
        normalized_scope = ["*"]
    else:
        normalized_scope = [s.lower().strip() for s in scope if s.strip()]

    # Generate unique token identifier
    token_hash = f"tok_{uuid.uuid4().hex[:16]}"

    # Query patient MRN for unambiguous record binding
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT mrn FROM patients WHERE patient_id = ?", (patient_id,))
        patient_row = cursor.fetchone()
        if not patient_row:
            raise ValueError(f"Patient with ID {patient_id} does not exist.")
        mrn = patient_row["mrn"]

        # Build cryptographic JWT payload (sub must be string per RFC 7519)
        jwt_claims = {
            "jti": token_hash,
            "sub": str(patient_id),
            "mrn": mrn,
            "scope": normalized_scope,
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
            "max_uses": max_uses,
        }

        signed_jwt = jwt.encode(
            payload=jwt_claims,
            key=Config.JWT_SECRET_KEY,
            algorithm=Config.JWT_ALGORITHM,
        )

        # Insert state record into relational database
        insert_query = """
            INSERT INTO vault_access_tokens (
                patient_id, token_hash, scope_json, issued_at, 
                expires_at, max_uses, use_count, revoked, qr_payload
            )
            VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?)
        """
        cursor.execute(
            insert_query,
            (
                patient_id,
                token_hash,
                json.dumps(normalized_scope),
                issued_at.isoformat(),
                expires_at.isoformat(),
                max_uses,
                signed_jwt,
            ),
        )
        conn.commit()
        token_id = cursor.lastrowid

        # Render visual QR code image
        qr_data_url = render_qr_code_base64(signed_jwt)

        audit_logger.info(
            f"TOKEN_ISSUED | patient_id={patient_id} | token_id={token_id} | "
            f"token_hash={token_hash} | scope={normalized_scope} | "
            f"expires_at={expires_at.isoformat()} | max_uses={max_uses}"
        )

        return {
            "token_id": token_id,
            "token_hash": token_hash,
            "patient_id": patient_id,
            "mrn": mrn,
            "scope": normalized_scope,
            "issued_at": issued_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "max_uses": max_uses,
            "qr_payload": signed_jwt,
            "qr_image_url": qr_data_url,
        }
    finally:
        if should_close and conn:
            conn.close()


def revoke_vault_access_token(
    token_id: Optional[int] = None,
    token_hash: Optional[str] = None,
    patient_id: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """Revokes an issued QR access token immediately.

    Args:
        token_id: Optional token ID to revoke.
        token_hash: Optional token hash to revoke.
        patient_id: Optional patient ID verification for authorization.
        conn: Optional active database connection.

    Returns:
        True if token was successfully revoked, False otherwise.
    """
    if token_id is None and token_hash is None:
        raise ValueError("Either token_id or token_hash must be specified to revoke a token.")

    now_iso = datetime.now().isoformat()
    query = "UPDATE vault_access_tokens SET revoked = 1, revoked_at = ? WHERE "
    params: List[Any] = [now_iso]

    if token_id is not None:
        query += "token_id = ?"
        params.append(token_id)
    else:
        query += "token_hash = ?"
        params.append(token_hash)

    if patient_id is not None:
        query += " AND patient_id = ?"
        params.append(patient_id)

    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        success = cursor.rowcount > 0
        if success:
            audit_logger.info(
                f"TOKEN_REVOKED | token_id={token_id} | token_hash={token_hash} | "
                f"patient_id={patient_id} | revoked_at={now_iso}"
            )
        return success
    finally:
        if should_close and conn:
            conn.close()


def get_patient_active_tokens(
    patient_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    """Retrieves all active, non-expired, unrevoked tokens for a patient.

    Args:
        patient_id: Target patient database ID.
        conn: Optional active database connection.

    Returns:
        List of active token dictionaries.
    """
    now_iso = datetime.now().isoformat()
    query = """
        SELECT token_id, patient_id, token_hash, scope_json, issued_at, 
               expires_at, max_uses, use_count, revoked, qr_payload
        FROM vault_access_tokens
        WHERE patient_id = ? AND revoked = 0 AND expires_at > ?
        ORDER BY issued_at DESC
    """
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, (patient_id, now_iso))
        rows = cursor.fetchall()
        tokens = []
        for r in rows:
            scope_list = ["*"]
            try:
                scope_list = json.loads(r["scope_json"])
            except Exception:
                pass

            tokens.append({
                "token_id": r["token_id"],
                "patient_id": r["patient_id"],
                "token_hash": r["token_hash"],
                "scope": scope_list,
                "issued_at": str(r["issued_at"]),
                "expires_at": str(r["expires_at"]),
                "max_uses": r["max_uses"],
                "use_count": r["use_count"],
                "revoked": bool(r["revoked"]),
                "qr_payload": r["qr_payload"],
            })
        return tokens
    finally:
        if should_close and conn:
            conn.close()
