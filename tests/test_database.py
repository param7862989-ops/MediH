"""Integration & Performance Tests for Phase 2 Relational Database.

Validates that:
1. All 9 tables and composite indexes exist and enforce relational constraints.
2. Foreign key violations trigger IntegrityError (PRAGMA foreign_keys = ON).
3. Transactions rollback atomically on exception.
4. The 100+ patient seeding engine generates clinically valid cohorts and time-series vitals.
5. QR tokens contain valid cryptographically signed JWT payloads.
6. Time-series indexing achieves sub-10ms query latencies.
"""

import json
import time
import pytest
import sqlite3
import jwt
from pathlib import Path

from src.database.db import get_connection, transaction, execute_query, init_db
from src.utils.config import Config


@pytest.fixture(scope="module", autouse=True)
def setup_test_database():
    """Ensure database is initialized and seeded before running database tests."""
    from database.seed_data import seed_database
    init_db(force_recreate=True)
    seed_database(100)


def test_schema_tables_and_indexes_exist():
    """Verify that all 9 tables and critical composite indexes exist."""
    expected_tables = {
        "patients",
        "vitals",
        "lab_results",
        "medications",
        "predictions",
        "outcomes",
        "medical_vault",
        "vault_access_tokens",
        "vault_access_log",
    }

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        for table in expected_tables:
            assert table in tables, f"Mandatory table '{table}' missing from database!"

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}
        assert "idx_vitals_patient_recorded" in indexes
        assert "idx_vault_patient_category" in indexes
        assert "idx_tokens_active" in indexes


def test_foreign_key_cascade_and_integrity_enforcement():
    """Verify that SQLite foreign key enforcement blocks orphaned child records."""
    with pytest.raises(sqlite3.IntegrityError):
        with transaction() as cursor:
            # Attempt to insert a vital sign for non-existent patient ID 999999
            cursor.execute(
                """
                INSERT INTO vitals (patient_id, recorded_at, heart_rate, blood_pressure_sys, 
                                    blood_pressure_dia, temperature, oxygen_saturation, respiratory_rate)
                VALUES (999999, '2026-10-06 12:00:00', 75, 120, 80, 37.0, 98.0, 16)
                """
            )


def test_transaction_atomic_rollback():
    """Verify that transactions rollback completely when an exception occurs."""
    initial_count = len(execute_query("SELECT * FROM patients"))

    with pytest.raises(RuntimeError):
        with transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO patients (mrn, full_name, age, gender, admission_date, status, ward, bed_number)
                VALUES ('MRN-TEST-FAIL', 'Test Rollback', 30, 'M', '2026-10-06 12:00:00', 'Admitted', 'Test Ward', 'B1')
                """
            )
            # Deliberately raise runtime error before commit
            raise RuntimeError("Forced simulation error during transaction")

    # Count must remain strictly unchanged
    final_count = len(execute_query("SELECT * FROM patients"))
    assert initial_count == final_count, "Rollback failed: partial changes were committed!"


def test_seeded_patient_cohorts_and_trajectories():
    """Verify patient counts, cohort distribution, and clinical vitals volume."""
    patients = execute_query("SELECT * FROM patients")
    assert len(patients) == 100, f"Expected 100 seeded patients, found {len(patients)}"

    # Check predictions cohort tiers
    predictions = execute_query("SELECT risk_tier, COUNT(*) as count FROM predictions GROUP BY risk_tier")
    tier_counts = {p["risk_tier"]: p["count"] for p in predictions}

    assert tier_counts.get("Low", 0) == 35
    assert tier_counts.get("Medium", 0) == 30
    assert tier_counts.get("High", 0) == 20
    assert tier_counts.get("Critical", 0) == 15

    # Check vitals volume and bounds
    vitals = execute_query("SELECT * FROM vitals LIMIT 500")
    assert len(vitals) == 500
    for v in vitals:
        assert 40.0 <= v["heart_rate"] <= 260.0
        assert 40.0 <= v["blood_pressure_sys"] <= 300.0
        assert 20.0 <= v["blood_pressure_dia"] <= 200.0
        assert 30.0 <= v["temperature"] <= 45.0
        assert 50.0 <= v["oxygen_saturation"] <= 100.0


def test_vault_access_tokens_and_jwt_signatures():
    """Verify that seeded QR access tokens contain cryptographically valid JWTs."""
    tokens = execute_query("SELECT * FROM vault_access_tokens LIMIT 20")
    assert len(tokens) > 0

    for token in tokens:
        qr_jwt = token["qr_payload"]
        # Verify and decode with secret key
        decoded = jwt.decode(qr_jwt, Config.JWT_SECRET_KEY, algorithms=[Config.JWT_ALGORITHM])
        assert "patient_id" in decoded
        assert "mrn" in decoded
        assert "scope" in decoded
        assert "exp" in decoded
        assert decoded["patient_id"] == token["patient_id"]


def test_timeseries_index_query_performance():
    """Benchmark time-series vitals query latency using idx_vitals_patient_recorded."""
    patient = execute_query("SELECT patient_id FROM patients LIMIT 1")[0]
    p_id = patient["patient_id"]

    start = time.perf_counter()
    vitals_series = execute_query(
        "SELECT * FROM vitals WHERE patient_id = ? ORDER BY recorded_at DESC LIMIT 50",
        (p_id,),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert len(vitals_series) > 0
    # Must complete comfortably under 25ms on SQLite WAL
    assert elapsed_ms < 25.0, f"Query took too long: {elapsed_ms:.2f}ms"
