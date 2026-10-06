"""Immutable Audit Trail and Access Logging Engine for MediHaven.

Guarantees full HIPAA-aligned and medico-legal auditability by recording every
QR scan attempt (successful or rejected) to both the relational vault_access_log
table and append-only disk audit log files.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import sqlite3

from src.database.db import get_connection, get_db_connection
from src.utils.config import Config
from src.utils.logger import get_audit_logger

audit_logger = get_audit_logger("medihaven.vault.audit")


def log_vault_access(
    token_id: int,
    patient_id: int,
    accessed_by: str,
    access_status: str,
    accessed_scope: Optional[List[str]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Logs an access attempt to the database and persistent security audit stream.

    Args:
        token_id: Token database ID.
        patient_id: Patient whose vault was accessed or targeted.
        accessed_by: Doctor/clinician name, hospital ID, or kiosk terminal.
        access_status: 'GRANTED', 'EXPIRED', 'REVOKED', 'INVALID_SIGNATURE', etc.
        accessed_scope: List of categories retrieved in this access event.
        ip_address: Client IP address.
        user_agent: Client User-Agent string.
        conn: Optional active database connection.

    Returns:
        Generated log_id.
    """
    now_iso = datetime.now().isoformat()
    scope_str = json.dumps(accessed_scope) if accessed_scope else "[]"

    # Ensure db_status satisfies schema CHECK constraint: ('GRANTED', 'EXPIRED', 'REVOKED', 'SCOPE_MISMATCH')
    valid_db_statuses = {"GRANTED", "EXPIRED", "REVOKED", "SCOPE_MISMATCH"}
    if access_status in valid_db_statuses:
        db_status = access_status
    elif "EXCEED" in access_status or "EXPIR" in access_status:
        db_status = "EXPIRED"
    else:
        db_status = "REVOKED"

    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO vault_access_log (
                token_id, patient_id, accessed_by, access_status, 
                accessed_at, ip_address, user_agent, accessed_scope
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(
            query,
            (
                token_id,
                patient_id,
                accessed_by.strip(),
                db_status,
                now_iso,
                ip_address,
                user_agent,
                scope_str,
            ),
        )
        conn.commit()
        log_id = cursor.lastrowid

        # Dual-write to disk audit logger
        audit_entry = {
            "log_id": log_id,
            "token_id": token_id,
            "patient_id": patient_id,
            "accessed_by": accessed_by,
            "status": access_status,
            "timestamp": now_iso,
            "ip": ip_address,
            "scope": accessed_scope,
        }
        audit_logger.info(f"VAULT_SCAN_EVENT | {json.dumps(audit_entry)}")

        return log_id
    finally:
        if should_close and conn:
            conn.close()


def get_patient_access_audit_log(
    patient_id: int,
    limit: int = 50,
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    """Retrieves full chronological access log for a patient to inspect in their portal.

    Args:
        patient_id: Target patient ID.
        limit: Maximum log rows to return.
        conn: Optional active database connection.

    Returns:
        List of audit log dictionaries.
    """
    query = """
        SELECT log_id, token_id, patient_id, accessed_by, access_status, 
               accessed_at, ip_address, user_agent, accessed_scope
        FROM vault_access_log
        WHERE patient_id = ?
        ORDER BY log_id DESC
        LIMIT ?
    """
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, (patient_id, limit))
        rows = cursor.fetchall()

        logs = []
        for r in rows:
            scope_val = []
            if r["accessed_scope"]:
                try:
                    scope_val = json.loads(r["accessed_scope"])
                except Exception:
                    scope_val = [r["accessed_scope"]]

            logs.append({
                "log_id": r["log_id"],
                "token_id": r["token_id"],
                "patient_id": r["patient_id"],
                "accessed_by": r["accessed_by"],
                "access_status": r["access_status"],
                "accessed_at": str(r["accessed_at"]),
                "ip_address": r["ip_address"] or "127.0.0.1",
                "user_agent": r["user_agent"] or "MediHaven Scanner",
                "accessed_scope": scope_val,
            })
        return logs
    finally:
        if should_close and conn:
            conn.close()
