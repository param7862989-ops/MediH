"""API route blueprints for patients, predictions, alerts, and vault."""

from src.api.routes.patients import patients_bp
from src.api.routes.predictions import predictions_bp
from src.api.routes.alerts import alerts_bp
from src.api.routes.vault import vault_bp

__all__ = [
    "patients_bp",
    "predictions_bp",
    "alerts_bp",
    "vault_bp",
]
