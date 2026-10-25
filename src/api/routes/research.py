"""Research-mode API (DATA_MODE=mimic only): verified Phase 2 model status, model card and
what-if estimates. Clinical roles only. Responses never contain patient records or identifiers."""

from flask import Blueprint, current_app, request

from src.api.auth import CLINICAL_ROLES, role_required
from src.api.schemas import error_response, success_response
from src.training.serving import PredictionInputError
from src.utils.data_mode import is_research_mode

research_bp = Blueprint("research", __name__, url_prefix="/api/research")


@research_bp.before_request
def require_research_mode():
    if not is_research_mode():
        return error_response(message="Research mode is not enabled (set DATA_MODE=mimic).",
                              code="RESEARCH_MODE_DISABLED", status_code=409)
    return None


def _service():
    return current_app.research_service


def _unavailable():
    status = _service().status()
    return error_response(message=status["reason"], code=status["reason_code"], status_code=503)


@research_bp.route("/status", methods=["GET"])
@role_required(CLINICAL_ROLES)
def research_status():
    return success_response(data=_service().status())


@research_bp.route("/model-card", methods=["GET"])
@role_required(CLINICAL_ROLES)
def research_model_card():
    if not _service().available:
        return _unavailable()
    return success_response(data=_service().model_card())


@research_bp.route("/predict", methods=["POST"])
@role_required(CLINICAL_ROLES)
def research_predict():
    if not _service().available:
        return _unavailable()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return error_response(message="Request body must be a JSON object.", code="INVALID_INPUT", status_code=422)
    try:
        result = _service().predict(payload.get("features"))
    except PredictionInputError as e:
        return error_response(message=str(e), code=e.code, status_code=422)
    return success_response(data=result, message="Research-only estimate. Not for patient-care decisions.")
