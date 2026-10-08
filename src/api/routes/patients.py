"""Patient Clinical Telemetry and Chart Routes for MediHaven REST API.

Provides patient triage queues, master chart profiles, admission registration,
longitudinal vitals time-series, and laboratory biomarker history.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Blueprint, request, jsonify
import sqlite3

from src.database.db import get_connection
from src.api.schemas import success_response, error_response, validate_required_fields
from src.utils.logger import get_logger

logger = get_logger("medihaven.api.patients")

patients_bp = Blueprint("patients", __name__, url_prefix="/api/patients")


@patients_bp.route("", methods=["GET"])
def list_patients():
    """Returns a filtered list of admitted patients for physician triage."""
    risk_tier = request.args.get("risk_tier", "").strip()
    ward = request.args.get("ward", "").strip()
    status = request.args.get("status", "").strip()
    search = request.args.get("search", "").strip()

    query = """
        SELECT p.patient_id, p.mrn, p.full_name, p.age, p.gender, 
               p.admission_date, p.discharge_date, p.status, p.ward, p.bed_number,
               p.assigned_physician, p.symptoms, p.primary_diagnosis, p.blood_group, p.emergency_contact,
               pr.risk_tier, pr.risk_score, pr.recommended_protocol,
               v.heart_rate as latest_hr, v.blood_pressure_sys as latest_sbp, 
               v.blood_pressure_dia as latest_dbp, v.oxygen_saturation as latest_spo2,
               v.temperature as latest_temp
        FROM patients p
        LEFT JOIN (
            SELECT patient_id, risk_tier, risk_score, recommended_protocol,
                   MAX(predicted_at) as max_pred
            FROM predictions
            GROUP BY patient_id
        ) pr ON p.patient_id = pr.patient_id
        LEFT JOIN (
            SELECT patient_id, heart_rate, blood_pressure_sys, blood_pressure_dia,
                   oxygen_saturation, temperature, MAX(recorded_at) as max_rec
            FROM vitals
            GROUP BY patient_id
        ) v ON p.patient_id = v.patient_id
        WHERE 1=1
    """
    params: List[Any] = []

    if status:
        query += " AND p.status = ?"
        params.append(status)

    if ward:
        query += " AND p.ward LIKE ?"
        params.append(f"%{ward}%")

    if risk_tier:
        query += " AND pr.risk_tier = ?"
        params.append(risk_tier)

    if search:
        query += " AND (p.full_name LIKE ? OR p.mrn LIKE ? OR p.assigned_physician LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

    query += " ORDER BY CASE pr.risk_tier WHEN 'Critical' THEN 1 WHEN 'High' THEN 2 WHEN 'Medium' THEN 3 ELSE 4 END, p.patient_id ASC"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        patients = []
        for r in rows:
            sbp = r["latest_sbp"] or 120.0
            dbp = r["latest_dbp"] or 80.0
            hr = r["latest_hr"] or 75.0
            si = round(hr / max(sbp, 1.0), 2)
            map_val = round(dbp + (1.0 / 3.0) * (sbp - dbp), 1)

            patients.append({
                "patient_id": r["patient_id"],
                "mrn": r["mrn"],
                "full_name": r["full_name"],
                "age": r["age"],
                "gender": r["gender"],
                "status": r["status"],
                "ward": r["ward"],
                "bed_number": r["bed_number"],
                "assigned_physician": r["assigned_physician"] or "Dr. Sarah Chen, MD (Attending)",
                "symptoms": r["symptoms"] or "Stable observation, baseline vitals",
                "primary_diagnosis": r["primary_diagnosis"] or "Observational Recovery",
                "blood_group": r["blood_group"] or "O+",
                "emergency_contact": r["emergency_contact"] or "+1 (555) 019-2834",
                "admission_date": str(r["admission_date"]),
                "risk_tier": r["risk_tier"] or "Low",
                "risk_score": round(float(r["risk_score"] or 0.1), 2),
                "recommended_protocol": r["recommended_protocol"] or "Protocol S: Standard Observational Recovery",
                "vitals_snapshot": {
                    "heart_rate": r["latest_hr"],
                    "sbp": r["latest_sbp"],
                    "dbp": r["latest_dbp"],
                    "spo2": r["latest_spo2"],
                    "temperature": r["latest_temp"],
                    "shock_index": si,
                    "mean_arterial_pressure": map_val,
                },
            })

    return success_response(
        data=patients,
        meta={"total": len(patients)},
    )


@patients_bp.route("/<int:patient_id>", methods=["GET"])
def get_patient(patient_id: int):
    """Returns full clinical chart details for a specific patient."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Demographics
        cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
        p_row = cursor.fetchone()
        if not p_row:
            return error_response(
                message=f"Patient with ID {patient_id} not found.",
                code="PATIENT_NOT_FOUND",
                status_code=404,
            )

        # Latest Vitals
        cursor.execute(
            """
            SELECT * FROM vitals 
            WHERE patient_id = ? 
            ORDER BY recorded_at DESC LIMIT 1
            """,
            (patient_id,),
        )
        v_row = cursor.fetchone()

        # Latest Lab Results
        cursor.execute(
            """
            SELECT * FROM lab_results 
            WHERE patient_id = ? 
            ORDER BY recorded_at DESC LIMIT 1
            """,
            (patient_id,),
        )
        l_row = cursor.fetchone()

        # Active Medications
        cursor.execute(
            """
            SELECT * FROM medications 
            WHERE patient_id = ? AND is_active = 1
            ORDER BY start_date DESC
            """,
            (patient_id,),
        )
        med_rows = cursor.fetchall()

        # Latest Prediction
        cursor.execute(
            """
            SELECT * FROM predictions 
            WHERE patient_id = ? 
            ORDER BY predicted_at DESC LIMIT 1
            """,
            (patient_id,),
        )
        pred_row = cursor.fetchone()

        chart = {
            "patient_id": p_row["patient_id"],
            "mrn": p_row["mrn"],
            "full_name": p_row["full_name"],
            "age": p_row["age"],
            "gender": p_row["gender"],
            "status": p_row["status"],
            "ward": p_row["ward"],
            "bed_number": p_row["bed_number"],
            "assigned_physician": p_row["assigned_physician"] if "assigned_physician" in p_row.keys() else "Dr. Sarah Chen, MD (Attending)",
            "symptoms": p_row["symptoms"] if "symptoms" in p_row.keys() else "Stable observation, baseline vitals",
            "primary_diagnosis": p_row["primary_diagnosis"] if "primary_diagnosis" in p_row.keys() else "Observational Recovery",
            "blood_group": p_row["blood_group"] if "blood_group" in p_row.keys() else "O+",
            "emergency_contact": p_row["emergency_contact"] if "emergency_contact" in p_row.keys() else "+1 (555) 019-2834",
            "admission_date": str(p_row["admission_date"]),
            "discharge_date": str(p_row["discharge_date"]) if p_row["discharge_date"] else None,
            "latest_vitals": dict(v_row) if v_row else None,
            "latest_labs": dict(l_row) if l_row else None,
            "active_medications": [dict(m) for m in med_rows],
            "latest_prediction": dict(pred_row) if pred_row else None,
        }

    return success_response(data=chart)


@patients_bp.route("", methods=["POST"])
def register_patient():
    """Registers a new patient admission."""
    payload = request.get_json(silent=True) or {}
    err = validate_required_fields(payload, ["full_name", "age", "gender", "ward", "bed_number"])
    if err:
        return error_response(message=err, code="VALIDATION_ERROR", status_code=422)

    now = datetime.now()
    mrn = payload.get("mrn")
    if not mrn:
        import time
        mrn = f"MRN-{now.year}-{(int(time.time() * 10000) % 90000 + 10000):05d}"

    query = """
        INSERT INTO patients (mrn, full_name, age, gender, admission_date, status, ward, bed_number)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        mrn,
        payload["full_name"].strip(),
        int(payload["age"]),
        payload["gender"].strip().upper(),
        now.isoformat(),
        payload.get("status", "Admitted"),
        payload["ward"].strip(),
        payload["bed_number"].strip(),
    )

    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            conn.commit()
            new_id = cursor.lastrowid
        except sqlite3.IntegrityError as e:
            return error_response(
                message=f"Database integrity error: {e}",
                code="DUPLICATE_MRN",
                status_code=409,
            )

    logger.info(f"Registered new patient #{new_id} [MRN: {mrn}, Name: '{payload['full_name']}']")
    return success_response(
        data={"patient_id": new_id, "mrn": mrn},
        message="Patient registered successfully.",
        status_code=201,
    )


@patients_bp.route("/<int:patient_id>/vitals", methods=["GET"])
def get_patient_vitals_series(patient_id: int):
    """Returns chronological vitals time series formatted for Chart.js telemetry charts."""
    hours_filter = request.args.get("hours", type=int)

    query = """
        SELECT recorded_at, heart_rate, blood_pressure_sys, blood_pressure_dia, 
               temperature, oxygen_saturation, respiratory_rate
        FROM vitals
        WHERE patient_id = ?
    """
    params: List[Any] = [patient_id]

    if hours_filter:
        cutoff = (datetime.now() - timedelta(hours=hours_filter)).isoformat()
        query += " AND recorded_at >= ?"
        params.append(cutoff)

    query += " ORDER BY recorded_at ASC"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        series = []
        for r in rows:
            sbp = r["blood_pressure_sys"]
            dbp = r["blood_pressure_dia"]
            hr = r["heart_rate"]
            map_val = round(dbp + (1.0 / 3.0) * (sbp - dbp), 1)
            si = round(hr / max(sbp, 1.0), 2)

            series.append({
                "recorded_at": str(r["recorded_at"]),
                "heart_rate": r["heart_rate"],
                "blood_pressure_sys": sbp,
                "blood_pressure_dia": dbp,
                "temperature": r["temperature"],
                "oxygen_saturation": r["oxygen_saturation"],
                "respiratory_rate": r["respiratory_rate"],
                "mean_arterial_pressure": map_val,
                "shock_index": si,
            })

    return success_response(
        data=series,
        meta={"patient_id": patient_id, "data_points": len(series)},
    )


@patients_bp.route("/<int:patient_id>/labs", methods=["GET"])
def get_patient_labs(patient_id: int):
    """Returns chronological laboratory biomarker test results."""
    query = """
        SELECT recorded_at, glucose, creatinine, cholesterol, 
               wbc_count, bun, hemoglobin, platelets
        FROM lab_results
        WHERE patient_id = ?
        ORDER BY recorded_at ASC
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (patient_id,))
        rows = cursor.fetchall()
        labs = [dict(r) for r in rows]

    return success_response(
        data=labs,
        meta={"patient_id": patient_id, "tests_count": len(labs)},
    )


@patients_bp.route("/<int:patient_id>/reassign", methods=["POST"])
def reassign_patient(patient_id: int):
    """Reassigns a patient to a different physician."""
    data = request.get_json(silent=True) or request.form
    new_physician = data.get("assigned_physician", "").strip()
    if not new_physician:
        return error_response("Physician name is required.", code="MISSING_FIELD", status_code=400)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT patient_id, full_name, assigned_physician FROM patients WHERE patient_id = ?", (patient_id,))
        patient = cursor.fetchone()
        if not patient:
            return error_response(f"Patient #{patient_id} not found.", code="NOT_FOUND", status_code=404)

        old_physician = patient["assigned_physician"] or "Unassigned"
        cursor.execute(
            "UPDATE patients SET assigned_physician = ? WHERE patient_id = ?",
            (new_physician, patient_id)
        )
        conn.commit()

    logger.info(f"PATIENT_REASSIGNED | patient_id={patient_id} | from='{old_physician}' | to='{new_physician}'")
    return success_response(
        data={
            "patient_id": patient_id,
            "full_name": patient["full_name"],
            "previous_physician": old_physician,
            "assigned_physician": new_physician,
        },
        message=f"Patient {patient['full_name']} successfully reassigned to {new_physician}."
    )


@patients_bp.route("/physicians", methods=["GET"])
def list_physicians():
    """Returns directory of hospital physicians, their specialties, and active caseload."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT assigned_physician, COUNT(*) as patient_count,
                   SUM(CASE WHEN pr.risk_tier IN ('Critical', 'High') THEN 1 ELSE 0 END) as high_risk_count
            FROM patients p
            LEFT JOIN (
                SELECT patient_id, risk_tier, MAX(predicted_at) FROM predictions GROUP BY patient_id
            ) pr ON p.patient_id = pr.patient_id
            WHERE assigned_physician IS NOT NULL AND assigned_physician != ''
            GROUP BY assigned_physician
            ORDER BY patient_count DESC
        """)
        rows = cursor.fetchall()
        physicians = [
            {
                "name": r["assigned_physician"],
                "active_patients": r["patient_count"],
                "high_risk_patients": r["high_risk_count"] or 0,
            }
            for r in rows
        ]

    return success_response(data=physicians, meta={"total_physicians": len(physicians)})

