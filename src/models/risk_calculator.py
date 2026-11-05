"""What-if Risk Calculator for the synthetic demo ensemble.

Turns one set of manually entered vital signs and labs into the same 30 engineered features the
dashboard models use, so a clinician can see how the four-model ensemble (Decision Tree, MLP, KNN,
K-Means) responds to hypothetical values. A single entered reading has no history, so:
- the 72-hour means, minimum and maximum equal the entered reading,
- the trend slopes and the first-to-latest lab changes are 0,
- an omitted optional value uses the same default the feature pipeline uses for missing data,
and every such assumption is returned with the result.

The models are trained on synthetic patients; the output is a demonstration, not clinical advice.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.data.feature_engineering import FEATURE_COLUMNS

# key -> (label, unit, min, max, default or None when required). Ranges match the database CHECKs.
INPUT_FIELDS: Dict[str, Tuple[str, str, float, float, Optional[float]]] = {
    "age": ("Age", "years", 18, 120, None),
    "heart_rate": ("Heart rate", "bpm", 20, 300, None),
    "sbp": ("Systolic BP", "mmHg", 40, 300, None),
    "dbp": ("Diastolic BP", "mmHg", 20, 200, None),
    "spo2": ("SpO2", "%", 50, 100, None),
    "respiratory_rate": ("Respiratory rate", "/min", 4, 80, 16.0),
    "temperature": ("Temperature", "°C", 30, 45, 36.8),
    "glucose": ("Glucose", "mg/dL", 10, 1000, 95.0),
    "creatinine": ("Creatinine", "mg/dL", 0.1, 25, 0.95),
    "bun": ("Blood urea nitrogen", "mg/dL", 1, 200, 15.0),
    "wbc": ("White cell count", "K/uL", 0.5, 150, 6.8),
    "hemoglobin": ("Hemoglobin", "g/dL", 2, 25, 14.0),
    "platelets": ("Platelets", "K/uL", 10, 2000, 250.0),
    "active_medications": ("Active medications", "count", 0, 50, 0.0),
}


class RiskInputError(ValueError):
    """Invalid calculator input; `field` names the offending input."""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


def describe_fields() -> List[Dict[str, Any]]:
    """Field metadata for the calculator form."""
    return [
        {"key": key, "label": label, "unit": unit, "min": lo, "max": hi,
         "required": default is None, "default": default}
        for key, (label, unit, lo, hi, default) in INPUT_FIELDS.items()
    ]


def _number(key: str, raw: Any) -> float:
    label, unit, lo, hi, _ = INPUT_FIELDS[key]
    if isinstance(raw, bool):
        raise RiskInputError(key, f"{label} must be a number.")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise RiskInputError(key, f"{label} must be a number.")
    if value != value or value in (float("inf"), float("-inf")):
        raise RiskInputError(key, f"{label} must be a finite number.")
    if not lo <= value <= hi:
        raise RiskInputError(key, f"{label} must be between {lo:g} and {hi:g} {unit}.")
    return value


def build_features(payload: Any) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Validates the entered values and returns (raw 30-feature dict, inputs and assumptions)."""
    if not isinstance(payload, dict):
        raise RiskInputError("body", "Send a JSON object with the measured values.")
    unknown = sorted(set(payload) - set(INPUT_FIELDS) - {"sex"})
    if unknown:
        raise RiskInputError(unknown[0], f"Unknown field: {unknown[0]}.")

    sex = str(payload.get("sex", "")).strip().lower()
    if sex not in ("male", "female"):
        raise RiskInputError("sex", "Sex must be 'male' or 'female'.")

    values: Dict[str, float] = {}
    defaulted: List[str] = []
    for key, (label, _unit, _lo, _hi, default) in INPUT_FIELDS.items():
        raw = payload.get(key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            if default is None:
                raise RiskInputError(key, f"{label} is required.")
            values[key] = default
            defaulted.append(label)
        else:
            values[key] = _number(key, raw)
    if values["sbp"] <= values["dbp"]:
        raise RiskInputError("dbp", "Diastolic BP must be lower than systolic BP.")

    hr, sbp, dbp, spo2 = values["heart_rate"], values["sbp"], values["dbp"], values["spo2"]
    temp, creat, bun = values["temperature"], values["creatinine"], values["bun"]
    mean_arterial_pressure = dbp + (sbp - dbp) / 3.0

    features: Dict[str, float] = {
        "age": values["age"],
        "gender_encoded": 1.0 if sex == "male" else 0.0,
        "heart_rate_latest": hr,
        "sbp_latest": sbp,
        "dbp_latest": dbp,
        "temperature_latest": temp,
        "oxygen_saturation_latest": spo2,
        "respiratory_rate_latest": values["respiratory_rate"],
        "heart_rate_mean": hr,
        "heart_rate_slope": 0.0,
        "sbp_mean": sbp,
        "sbp_slope": 0.0,
        "spo2_mean": spo2,
        "spo2_min": spo2,
        "spo2_slope": 0.0,
        "temp_max": temp,
        "mean_arterial_pressure": mean_arterial_pressure,
        "pulse_pressure": sbp - dbp,
        "shock_index": hr / max(sbp, 1.0),
        "modified_shock_index": hr / max(mean_arterial_pressure, 1.0),
        "glucose_latest": values["glucose"],
        "glucose_delta": 0.0,
        "creatinine_latest": creat,
        "creatinine_delta": 0.0,
        "wbc_latest": values["wbc"],
        "bun_latest": bun,
        "bun_creatinine_ratio": bun / max(creat, 0.1),
        "hemoglobin_latest": values["hemoglobin"],
        "platelets_latest": values["platelets"],
        "active_medications_count": values["active_medications"],
    }
    assert list(features) == FEATURE_COLUMNS, "feature order must match the trained models"

    assumptions = [
        "A single reading is treated as the whole 72-hour window, so means, minimum and maximum "
        "equal the entered value and all trend slopes and lab changes are 0.",
    ]
    if defaulted:
        assumptions.append("Not entered, so a typical value was used: " + ", ".join(defaulted) + ".")
    return features, {
        "inputs": {"sex": sex, **values},
        "derived": {
            "mean_arterial_pressure": round(mean_arterial_pressure, 1),
            "shock_index": round(features["shock_index"], 2),
            "pulse_pressure": round(sbp - dbp, 1),
        },
        "assumptions": assumptions,
        "defaulted": defaulted,
    }
