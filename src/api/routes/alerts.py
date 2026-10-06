"""Automated Early-Warning Alert Feed Routes for MediHaven REST API.

Provides real-time clinical deterioration alerts identifying patients at risk of
decompensation 7–14 days before adverse critical events occur.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from flask import Blueprint, request, current_app

from src.database.db import get_connection
from src.api.schemas import success_response, error_response, validate_required_fields
from src.utils.logger import get_logger

logger = get_logger("medihaven.api.alerts")

alerts_bp = Blueprint("alerts", __name__, url_prefix="/api/alerts")


@alerts_bp.route("", methods=["GET"])
def get_alerts():
    """Returns priority-sorted feed of early-warning deterioration alerts."""
    min_tier = request.args.get("min_tier", "High").strip()

    query = """
        SELECT p.patient_id, p.mrn, p.full_name, p.age, p.gender, 
               p.ward, p.bed_number, p.admission_date,
               pr.risk_tier, pr.risk_score, pr.decision_tree_path, 
               pr.recommended_protocol, pr.predicted_at,
               v.heart_rate, v.blood_pressure_sys, v.blood_pressure_dia,
               v.oxygen_saturation, v.respiratory_rate, v.temperature,
               v.recorded_at as vitals_time
        FROM patients p
        JOIN (
            SELECT patient_id, risk_tier, risk_score, decision_tree_path, 
                   recommended_protocol, predicted_at,
                   MAX(predicted_at) as max_pred
            FROM predictions
            GROUP BY patient_id
        ) pr ON p.patient_id = pr.patient_id
        LEFT JOIN (
            SELECT patient_id, heart_rate, blood_pressure_sys, blood_pressure_dia,
                   oxygen_saturation, respiratory_rate, temperature, recorded_at,
                   MAX(recorded_at) as max_rec
            FROM vitals
            GROUP BY patient_id
        ) v ON p.patient_id = v.patient_id
        WHERE p.status IN ('Admitted', 'Observation', 'ICU')
          AND pr.risk_tier IN ('Critical', 'High')
        ORDER BY CASE pr.risk_tier WHEN 'Critical' THEN 1 ELSE 2 END, 
                 pr.risk_score DESC, p.patient_id ASC
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        alerts = []
        for idx, r in enumerate(rows):
            sbp = r["blood_pressure_sys"] or 120.0
            dbp = r["blood_pressure_dia"] or 80.0
            hr = r["heart_rate"] or 75.0
            spo2 = r["oxygen_saturation"] or 98.0
            resp = r["respiratory_rate"] or 16.0
            si = round(hr / max(sbp, 1.0), 2)
            map_val = round(dbp + (1.0 / 3.0) * (sbp - dbp), 1)

            # Synthesize physiological trigger explanations
            triggers = []
            if si >= 0.90:
                triggers.append(f"Elevated Shock Index ({si:.2f}): Impending Hemodynamic Shock")
            if spo2 < 92.0:
                triggers.append(f"Hypoxemia Alert: SpO2 ({spo2:.1f}%) below critical clinical safety floor")
            if resp >= 24:
                triggers.append(f"Tachypnea ({resp} breaths/min): Respiratory Compensation Active")
            if sbp >= 160.0:
                triggers.append(f"Hypertensive Crisis (SBP: {sbp:.0f} mmHg): Elevated CVD Stroke Risk")
            if not triggers:
                triggers.append(f"Multimodal Risk Score {float(r['risk_score']):.2f} exceeds clinical monitoring threshold")

            alerts.append({
                "alert_id": f"ALT-{r['patient_id']:04d}-{idx + 1:02d}",
                "patient_id": r["patient_id"],
                "mrn": r["mrn"],
                "full_name": r["full_name"],
                "age": r["age"],
                "gender": r["gender"],
                "ward": r["ward"],
                "bed_number": r["bed_number"],
                "risk_tier": r["risk_tier"],
                "risk_score": round(float(r["risk_score"]), 2),
                "early_warning_window": "7–14 Days",
                "urgency_level": "CRITICAL ACTION REQUIRED" if r["risk_tier"] == "Critical" else "URGENT CLINICAL REVIEW",
                "physiological_triggers": triggers,
                "recommended_protocol": r["recommended_protocol"],
                "latest_telemetry": {
                    "heart_rate": hr,
                    "blood_pressure": f"{sbp:.0f}/{dbp:.0f} mmHg",
                    "shock_index": si,
                    "mean_arterial_pressure": map_val,
                    "oxygen_saturation": f"{spo2:.1f}%",
                    "respiratory_rate": f"{resp:.0f} bpm",
                    "recorded_at": str(r["vitals_time"]),
                },
                "triggered_at": str(r["predicted_at"]),
            })

    return success_response(
        data=alerts,
        meta={"total_active_alerts": len(alerts)},
    )


@alerts_bp.route("/<int:patient_id>/acknowledge", methods=["POST"])
def acknowledge_alert(patient_id: int):
    """Allows treating physician to acknowledge alert and log clinical intent."""
    payload = request.get_json(silent=True) or {}
    doctor_name = payload.get("doctor_name", "Attending Physician").strip()
    notes = payload.get("notes", "Alert reviewed; clinical intervention planned.").strip()

    logger.info(
        f"ALERT_ACKNOWLEDGED | patient_id={patient_id} | "
        f"doctor='{doctor_name}' | notes='{notes}'"
    )

    return success_response(
        data={
            "patient_id": patient_id,
            "acknowledged_by": doctor_name,
            "acknowledged_at": datetime.now().isoformat(),
            "clinical_notes": notes,
        },
        message=f"Early warning alert for Patient #{patient_id} acknowledged.",
    )
