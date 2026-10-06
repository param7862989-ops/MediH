"""Clinical Intelligence, Machine Learning Inference, and Dashboard Metrics Routes for MediHaven.

Provides live multi-model prediction, explainable rule path extraction, historical case
precedent search (KNN), and aggregated hospital dashboard analytics.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from flask import Blueprint, request, current_app

from src.database.db import get_connection
from src.api.schemas import success_response, error_response
from src.models.ensemble import EnsembleClinicalPredictor
from src.utils.logger import get_logger

logger = get_logger("medihaven.api.predictions")

predictions_bp = Blueprint("predictions", __name__, url_prefix="/api")


def get_ensemble() -> EnsembleClinicalPredictor:
    """Helper to retrieve pre-loaded ensemble instance from Flask current_app context."""
    ensemble = getattr(current_app, "ensemble_predictor", None)
    if ensemble is None:
        logger.warning("Ensemble predictor not cached on app context. Loading from disk...")
        ensemble = EnsembleClinicalPredictor.load()
        current_app.ensemble_predictor = ensemble
    return ensemble


@predictions_bp.route("/patients/<int:patient_id>/predict", methods=["GET"])
def predict_patient_risk(patient_id: int):
    """Executes live multi-model inference and returns full clinical intelligence packet."""
    try:
        ensemble = get_ensemble()
        prediction = ensemble.predict_patient(patient_id)

        # Record prediction into predictions table for audit history
        with get_connection() as conn:
            cursor = conn.cursor()
            rule_path_json = json.dumps(prediction.get("decision_tree_explanation", []))
            similar_ids_json = json.dumps([p["patient_id"] for p in prediction.get("similar_patients", [])])

            cursor.execute(
                """
                INSERT INTO predictions (
                    patient_id, predicted_at, risk_tier, risk_score, model_used,
                    decision_tree_path, similar_patient_ids, recommended_protocol,
                    readmission_30d_risk
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_id,
                    datetime.now().isoformat(),
                    prediction["risk_tier"],
                    prediction["risk_score"],
                    "Ensemble (4-Model)",
                    rule_path_json,
                    similar_ids_json,
                    prediction["recommended_protocol"],
                    prediction.get("readmission_30d_risk", 0.1),
                ),
            )
            conn.commit()

        return success_response(
            data=prediction,
            message="Live clinical risk assessment generated successfully.",
        )
    except ValueError as e:
        return error_response(
            message=str(e),
            code="PATIENT_NOT_FOUND",
            status_code=404,
        )
    except Exception as e:
        logger.error(f"Inference error for patient {patient_id}: {e}", exc_info=True)
        return error_response(
            message=f"Clinical inference error: {e}",
            code="INFERENCE_FAILED",
            status_code=500,
        )


@predictions_bp.route("/patients/<int:patient_id>/similar", methods=["GET"])
def get_similar_cases(patient_id: int):
    """Retrieves top-K similar historical patient precedents and their outcomes."""
    k = request.args.get("k", default=5, type=int)

    try:
        ensemble = get_ensemble()
        prediction = ensemble.predict_patient(patient_id)
        similar_patients = prediction.get("similar_patients", [])[:k]

        return success_response(
            data={
                "patient_id": patient_id,
                "matched_count": len(similar_patients),
                "recommended_protocol": prediction.get("recommended_protocol"),
                "protocol_success_rate": prediction.get("protocol_success_rate"),
                "similar_patients": similar_patients,
            }
        )
    except ValueError as e:
        return error_response(message=str(e), code="PATIENT_NOT_FOUND", status_code=404)
    except Exception as e:
        return error_response(message=str(e), code="SIMILARITY_SEARCH_ERROR", status_code=500)


@predictions_bp.route("/dashboard/summary", methods=["GET"])
def get_dashboard_summary():
    """Returns high-level hospital risk distribution and resource utilization metrics."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Total admitted count
        cursor.execute("SELECT COUNT(*) FROM patients WHERE status IN ('Admitted', 'Observation', 'ICU')")
        total_admitted = cursor.fetchone()[0]

        # 2. Risk tier distribution from latest predictions
        cursor.execute(
            """
            SELECT pr.risk_tier, COUNT(*) as count
            FROM (
                SELECT patient_id, risk_tier, MAX(predicted_at)
                FROM predictions
                GROUP BY patient_id
            ) pr
            JOIN patients p ON pr.patient_id = p.patient_id
            WHERE p.status IN ('Admitted', 'Observation', 'ICU')
            GROUP BY pr.risk_tier
            """
        )
        tier_counts_raw = dict(cursor.fetchall())
        tier_distribution = {
            "Critical": tier_counts_raw.get("Critical", 0),
            "High": tier_counts_raw.get("High", 0),
            "Medium": tier_counts_raw.get("Medium", 0),
            "Low": tier_counts_raw.get("Low", 0),
        }

        # 3. ICU and CCU bed occupancy
        cursor.execute("SELECT COUNT(*) FROM patients WHERE status = 'ICU' OR ward LIKE '%ICU%'")
        icu_occupancy = cursor.fetchone()[0]

        # 4. Active high deterioration risk alerts count (High + Critical cohorts)
        active_alerts_count = tier_distribution["Critical"] + tier_distribution["High"]

        # 5. Average readmission risk
        cursor.execute("SELECT AVG(readmission_30d_risk) FROM predictions")
        avg_readmit = cursor.fetchone()[0] or 0.22

        # 6. Ward distribution
        cursor.execute(
            """
            SELECT ward, COUNT(*) as count 
            FROM patients 
            WHERE status IN ('Admitted', 'Observation', 'ICU')
            GROUP BY ward
            """
        )
        ward_distribution = dict(cursor.fetchall())

    summary = {
        "total_admitted": total_admitted,
        "risk_distribution": tier_distribution,
        "icu_occupancy": icu_occupancy,
        "active_deterioration_alerts": active_alerts_count,
        "average_30d_readmission_risk": round(float(avg_readmit), 2),
        "ward_distribution": ward_distribution,
        "early_warning_window": "7–14 Days",
    }

    return success_response(data=summary)
