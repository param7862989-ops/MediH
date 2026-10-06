"""Patient Medical Vault CRUD and Record Management Engine for MediHaven.

Provides secure storage, categorization, and retrieval of patient-owned medical
history (diagnoses, allergies, medications, lab reports, imaging, surgeries, vaccinations).
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import sqlite3

from src.database.db import get_connection, get_db_connection, transaction
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("medihaven.vault.service")

VALID_VAULT_CATEGORIES = {
    "diagnosis",
    "allergy",
    "medication",
    "lab_report",
    "imaging",
    "surgery",
    "vaccination",
}


def add_vault_record(
    patient_id: int,
    category: str,
    title: str,
    description: Optional[str] = None,
    structured_data: Optional[Dict[str, Any]] = None,
    file_path: Optional[str] = None,
    is_sensitive: bool = False,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Adds a new medical record to a patient's Medical Vault.

    Args:
        patient_id: Patient database ID.
        category: Record category (must be in VALID_VAULT_CATEGORIES).
        title: Short title or condition name.
        description: Detailed clinical description or notes.
        structured_data: Optional dictionary of category-specific attributes.
        file_path: Optional path to uploaded document/image/PDF.
        is_sensitive: Whether this record is sensitive (requires explicit scope).
        conn: Optional active database connection.

    Returns:
        Generated vault_id.
    """
    cat_lower = category.lower().strip()
    if cat_lower not in VALID_VAULT_CATEGORIES:
        raise ValueError(
            f"Invalid vault category '{category}'. Must be one of: {sorted(VALID_VAULT_CATEGORIES)}"
        )

    s_data_json = json.dumps(structured_data) if structured_data else None
    uploaded_at = datetime.now().isoformat()

    query = """
        INSERT INTO medical_vault (
            patient_id, category, title, description, 
            file_path, structured_data, uploaded_at, is_sensitive
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        patient_id,
        cat_lower,
        title.strip(),
        description.strip() if description else "",
        file_path,
        s_data_json,
        uploaded_at,
        1 if is_sensitive else 0,
    )

    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        vault_id = cursor.lastrowid
        logger.info(
            f"Added vault record #{vault_id} for patient_id={patient_id} [category={cat_lower}, title='{title}']"
        )
        return vault_id
    finally:
        if should_close and conn:
            conn.close()


def get_patient_vault_records(
    patient_id: int,
    category: Optional[str] = None,
    include_sensitive: bool = True,
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    """Retrieves all medical history records for a patient.

    Args:
        patient_id: Target patient ID.
        category: Optional category filter.
        include_sensitive: If False, omits records flagged is_sensitive=1.
        conn: Optional active database connection.

    Returns:
        List of vault record dictionaries.
    """
    query = """
        SELECT vault_id, patient_id, category, title, description, 
               file_path, structured_data, uploaded_at, is_sensitive
        FROM medical_vault
        WHERE patient_id = ?
    """
    params: List[Any] = [patient_id]

    if category:
        query += " AND category = ?"
        params.append(category.lower().strip())

    if not include_sensitive:
        query += " AND is_sensitive = 0"

    query += " ORDER BY uploaded_at DESC, vault_id DESC"

    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        records = []
        for r in rows:
            s_data = None
            if r["structured_data"]:
                try:
                    s_data = json.loads(r["structured_data"])
                except Exception:
                    s_data = r["structured_data"]

            records.append({
                "vault_id": r["vault_id"],
                "patient_id": r["patient_id"],
                "category": r["category"],
                "title": r["title"],
                "description": r["description"],
                "file_path": r["file_path"],
                "structured_data": s_data,
                "uploaded_at": str(r["uploaded_at"]),
                "is_sensitive": bool(r["is_sensitive"]),
            })
        return records
    finally:
        if should_close and conn:
            conn.close()


def get_vault_record_by_id(
    vault_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieves a single vault record by ID."""
    query = """
        SELECT vault_id, patient_id, category, title, description, 
               file_path, structured_data, uploaded_at, is_sensitive
        FROM medical_vault
        WHERE vault_id = ?
    """
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, (vault_id,))
        row = cursor.fetchone()
        if not row:
            return None

        s_data = None
        if row["structured_data"]:
            try:
                s_data = json.loads(row["structured_data"])
            except Exception:
                s_data = row["structured_data"]

        return {
            "vault_id": row["vault_id"],
            "patient_id": row["patient_id"],
            "category": row["category"],
            "title": row["title"],
            "description": row["description"],
            "file_path": row["file_path"],
            "structured_data": s_data,
            "uploaded_at": str(row["uploaded_at"]),
            "is_sensitive": bool(row["is_sensitive"]),
        }
    finally:
        if should_close and conn:
            conn.close()


def delete_vault_record(
    vault_id: int,
    patient_id: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """Deletes a vault record."""
    query = "DELETE FROM medical_vault WHERE vault_id = ?"
    params: List[Any] = [vault_id]
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
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted vault record #{vault_id}")
        return deleted
    finally:
        if should_close and conn:
            conn.close()


def get_vault_category_summary(
    patient_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, int]:
    """Returns record count per category for dashboard summary metrics."""
    query = """
        SELECT category, COUNT(*) as count
        FROM medical_vault
        WHERE patient_id = ?
        GROUP BY category
    """
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute(query, (patient_id,))
        rows = cursor.fetchall()
        summary = {cat: 0 for cat in VALID_VAULT_CATEGORIES}
        for r in rows:
            summary[r["category"]] = r["count"]
        return summary
    finally:
        if should_close and conn:
            conn.close()
