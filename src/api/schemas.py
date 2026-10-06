"""JSON Envelope Schemas and Request Validation Helpers for MediHaven REST API.

Enforces consistent response formats, error standardization, and payload validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from flask import jsonify, Response


def success_response(
    data: Any = None,
    message: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    status_code: int = 200,
) -> Tuple[Response, int]:
    """Formats a standardized success JSON response envelope.

    Args:
        data: Primary response payload.
        message: Optional user-facing success message.
        meta: Optional metadata dictionary (timestamps, pagination, counts).
        status_code: HTTP status code (default: 200).

    Returns:
        Flask JSON response tuple.
    """
    envelope = {
        "success": True,
        "data": data,
        "message": message,
        "meta": meta or {},
    }
    envelope["meta"].setdefault("timestamp", datetime.now().isoformat())
    return jsonify(envelope), status_code


def error_response(
    message: str = "An unexpected error occurred",
    code: str = "BAD_REQUEST",
    details: Optional[Any] = None,
    status_code: int = 400,
) -> Tuple[Response, int]:
    """Formats a standardized error JSON response envelope.

    Args:
        message: Descriptive error message.
        code: Machine-readable error code string.
        details: Optional detailed error context.
        status_code: HTTP status code (default: 400).

    Returns:
        Flask JSON response tuple.
    """
    envelope = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "details": details,
        },
        "meta": {
            "timestamp": datetime.now().isoformat(),
        },
    }
    return jsonify(envelope), status_code


def validate_required_fields(payload: Dict[str, Any], required_fields: List[str]) -> Optional[str]:
    """Validates that all required fields are present in a JSON dictionary.

    Args:
        payload: Input dictionary.
        required_fields: List of expected field names.

    Returns:
        Error message string if validation fails, or None if valid.
    """
    if not isinstance(payload, dict):
        return "Request body must be a valid JSON object."

    missing = [f for f in required_fields if f not in payload or payload[f] is None or payload[f] == ""]
    if missing:
        return f"Missing required parameter(s): {', '.join(missing)}"

    return None
