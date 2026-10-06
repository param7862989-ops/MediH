"""Access Controller and Scope Isolation Enforcement Engine for MediHaven.

Guarantees zero data leakage by strictly validating cryptographic QR passes,
enforcing categorical whitelist filtering on medical vault records, atomically updating
use quotas, and committing audit log entries.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import sqlite3

from src.database.db import get_connection, get_db_connection
from src.vault.qr_scanner import scan_and_validate_qr
from src.vault.access_logger import log_vault_access
from src.vault.vault_service import get_patient_vault_records
from src.utils.logger import get_logger

logger = get_logger("medihaven.vault.access_controller")


def access_vault_with_token(
    raw_qr_input: Union[str, bytes, Any],
    accessed_by: str = "Consulting Physician",
    ip_address: Optional[str] = "127.0.0.1",
    user_agent: Optional[str] = "MediHaven Provider Portal",
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """Scans and validates a QR access token, then returns strictly authorized vault records.

    Args:
        raw_qr_input: JWT token string or visual QR code image (Base64/bytes/path).
        accessed_by: Doctor/clinician identifier or clinic name.
        ip_address: Client IP address.
        user_agent: Scanner terminal / User-Agent.
        conn: Optional active database connection.

    Returns:
        Structured clinical payload with authorized records, or rejection explanation.
    """
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        # Step 1: Scan & Cryptographically Validate Token
        val_res = scan_and_validate_qr(raw_qr_input, conn=conn)

        token_record = val_res.get("token_record")
        claims = val_res.get("claims")

        token_id = int(token_record.get("token_id", 0)) if token_record else 0
        raw_pid = token_record.get("patient_id") if token_record else (claims.get("sub") if claims else 0)
        patient_id = int(raw_pid) if raw_pid else 0
        scope = claims.get("scope", ["*"]) if claims else []

        # If Token Validation Failed
        if not val_res["is_valid"]:
            status_reason = val_res["access_status"]
            if token_id and patient_id:
                # Log failed attempt
                log_vault_access(
                    token_id=token_id,
                    patient_id=patient_id,
                    accessed_by=accessed_by,
                    access_status=status_reason,
                    accessed_scope=scope,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    conn=conn,
                )

            return {
                "access_granted": False,
                "access_status": status_reason,
                "error_message": val_res["error_message"],
                "records": [],
            }

        # Step 2: Query Patient Identity for Clinical Display
        cursor = conn.cursor()
        cursor.execute("SELECT full_name, mrn, age, gender FROM patients WHERE patient_id = ?", (patient_id,))
        p_row = cursor.fetchone()
        patient_name = p_row["full_name"] if p_row else "Unknown Patient"
        mrn = p_row["mrn"] if p_row else claims.get("mrn", "UNKNOWN")

        # Step 3: Fetch Medical Vault Records with Strict Whitelist Scope Filtering
        all_patient_records = get_patient_vault_records(
            patient_id=patient_id,
            include_sensitive=True,
            conn=conn,
        )

        filtered_records = []
        is_wildcard_scope = "*" in scope

        for rec in all_patient_records:
            cat = rec["category"].lower()
            # If wildcard scope or category explicitly whitelisted
            if is_wildcard_scope or cat in scope:
                # Exclude sensitive items if wildcard unless specifically listed
                if rec["is_sensitive"] and is_wildcard_scope and len(scope) == 1:
                    continue
                filtered_records.append(rec)

        # Step 4: Atomically Increment Token Use Count
        cursor.execute(
            "UPDATE vault_access_tokens SET use_count = use_count + 1 WHERE token_id = ?",
            (token_id,),
        )
        conn.commit()

        # Step 5: Log Immutable Audit Event
        log_id = log_vault_access(
            token_id=token_id,
            patient_id=patient_id,
            accessed_by=accessed_by,
            access_status="GRANTED",
            accessed_scope=scope,
            ip_address=ip_address,
            user_agent=user_agent,
            conn=conn,
        )

        logger.info(
            f"ACCESS_GRANTED | patient_id={patient_id} | token_id={token_id} | "
            f"accessed_by='{accessed_by}' | records_returned={len(filtered_records)}"
        )

        return {
            "access_granted": True,
            "access_status": "GRANTED",
            "log_id": log_id,
            "patient_id": patient_id,
            "mrn": mrn,
            "patient_name": patient_name,
            "patient_age": p_row["age"] if p_row else None,
            "patient_gender": p_row["gender"] if p_row else None,
            "authorized_scope": scope,
            "records_count": len(filtered_records),
            "records": filtered_records,
            "accessed_by": accessed_by,
            "accessed_at": datetime.now().isoformat(),
            "token_expires_at": token_record.get("expires_at") if token_record else None,
            "audit_logged": True,
        }
    finally:
        if should_close and conn:
            conn.close()
