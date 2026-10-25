"""Input contract for research "what-if" estimates: builds one model row exactly as Phase 1 would.

For each of the 14 base variables (heart_rate, resp_rate, spo2, temperature, sbp, dbp, map,
creatinine, bun, wbc, hemoglobin, platelets, lactate, glucose):

- Not measured: supply nothing (or only ``<var>_count = 0`` and/or ``<var>_missing = 1``). The row
  gets ``<var>_count = 0`` and ``<var>_missing = 1`` with empty summaries — the same encoding Phase 1
  uses for unmeasured stays (``_missing = (count == 0)``), whose summaries the model's training-median
  imputer fills, exactly as during training.
- Measured: ``<var>_count`` is REQUIRED (integer >= 1); ``<var>_missing`` may be omitted (it is then
  0) or given as 0. With count = 1 at least one of min/max/mean/last is required and all supplied
  values must be equal (a single reading's summaries are identical, so the others are set to it).
  With count >= 2 all four summaries are required and must satisfy min <= mean <= max and
  min <= last <= max. Values must lie in the feature spec's plausible range.
- ``bp_source_invasive`` (0/1) is required when sbp or dbp is measured and must be omitted otherwise;
  ``map_derived`` (0/1) is required when map is measured and must be omitted otherwise.
- ``age`` (18–120) and ``sex_male`` (0/1) are optional; when omitted they are imputed with the
  training median and listed in the response.

Contradictions (e.g. a positive count with ``_missing = 1``, values with ``_count = 0``) and
impossible values are rejected; nothing is assumed to have been measured.
"""

import math
from typing import Dict, List, Tuple

from src.datasources.mimic.mappings import VARIABLES
from src.training.features import forbidden_reason

SUMMARY_STATS = ("min", "max", "mean", "last")
MAX_COUNT = 10000
AGE_RANGE = (18.0, 120.0)
ENCODING_NOTE = (
    "Variables not supplied are encoded as not measured (<variable>_missing = 1, <variable>_count = 0), "
    "the same encoding Phase 1 uses for unmeasured stays; their summary values, and age/sex if omitted, are "
    "filled by the model's training-median imputer, as during training. No measurement is assumed."
)


class PredictionInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str):
    raise PredictionInputError(code, message)


def _number(key: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail("INVALID_VALUE", f"'{key}' must be a finite number or null.")
    return float(value)


def _binary(key: str, value) -> float:
    number = _number(key, value)
    if number not in (0.0, 1.0):
        _fail("INVALID_VALUE", f"'{key}' must be 0 or 1.")
    return number


def _count(key: str, value) -> int:
    number = _number(key, value)
    if not number.is_integer() or not 0 <= number <= MAX_COUNT:
        _fail("INVALID_VALUE", f"'{key}' must be a whole number between 0 and {MAX_COUNT}.")
    return int(number)


def _validate_keys(supplied: object, features: List[str]) -> Dict[str, object]:
    if not isinstance(supplied, dict) or not supplied:
        _fail("INVALID_INPUT", "'features' must be a non-empty object of feature values.")
    if len(supplied) > len(features):
        _fail("INVALID_INPUT", "Too many feature values.")
    for key in supplied:
        if not isinstance(key, str) or len(key) > 64:
            _fail("INVALID_INPUT", "Feature names must be short strings.")
        if forbidden_reason(key):
            _fail("FORBIDDEN_FEATURE", f"'{key}' can never be a model input (identifier, outcome or metadata).")
        if key not in features:
            _fail("UNKNOWN_FEATURE", f"'{key}' is not a permitted feature.")
    return {k: v for k, v in supplied.items() if v is not None}  # null means "not supplied"


def _variable(var, values: Dict[str, object], row: Dict[str, float]) -> bool:
    """Fills one base variable into row; returns True if it is measured."""
    name = var.name
    stats = {s: _number(f"{name}_{s}", values[f"{name}_{s}"]) for s in SUMMARY_STATS if f"{name}_{s}" in values}
    count = _count(f"{name}_count", values[f"{name}_count"]) if f"{name}_count" in values else None
    missing = _binary(f"{name}_missing", values[f"{name}_missing"]) if f"{name}_missing" in values else None

    if not stats and count in (None, 0) and missing in (None, 1.0):
        row[f"{name}_count"], row[f"{name}_missing"] = 0.0, 1.0
        return False

    if missing == 1.0:
        _fail("CONTRADICTORY_INPUT", f"'{name}_missing' = 1 contradicts the supplied {name} count or values.")
    if count == 0:
        _fail("CONTRADICTORY_INPUT", f"'{name}_count' = 0 means not measured, which contradicts the other {name} inputs.")
    if count is None:
        _fail("INCOMPLETE_VARIABLE", f"'{name}_count' (number of measurements, >= 1) is required when {name} is supplied.")
    if not stats:
        _fail("INCOMPLETE_VARIABLE", f"'{name}_count' = {count} needs at least one {name} summary value.")
    for stat, value in stats.items():
        if not var.plausible_min <= value <= var.plausible_max:
            _fail("INVALID_VALUE", f"'{name}_{stat}' is outside the plausible range "
                                   f"{var.plausible_min:g}–{var.plausible_max:g} {var.unit}.")

    if count == 1:
        if len(set(stats.values())) > 1:
            _fail("CONTRADICTORY_INPUT", f"With '{name}_count' = 1 all {name} summaries describe one reading and must be equal.")
        single = next(iter(stats.values()))
        stats = {s: single for s in SUMMARY_STATS}
    else:
        absent = [f"{name}_{s}" for s in SUMMARY_STATS if s not in stats]
        if absent:
            _fail("INCOMPLETE_VARIABLE", f"With '{name}_count' = {count}, all of {name}_min/max/mean/last are required "
                                         f"(missing: {', '.join(absent)}).")
        if not (stats["min"] <= stats["mean"] <= stats["max"] and stats["min"] <= stats["last"] <= stats["max"]):
            _fail("CONTRADICTORY_INPUT", f"{name} summaries must satisfy min <= mean <= max and min <= last <= max.")

    for stat, value in stats.items():
        row[f"{name}_{stat}"] = value
    row[f"{name}_count"], row[f"{name}_missing"] = float(count), 0.0
    return True


def _conditional_flag(key: str, values: Dict[str, object], required: bool, reason: str) -> float:
    if required:
        if key not in values:
            _fail("INCOMPLETE_VARIABLE", f"'{key}' (0 or 1) is required when {reason}.")
        return _binary(key, values[key])
    if key in values:
        _fail("CONTRADICTORY_INPUT", f"'{key}' must be omitted unless {reason}.")
    return math.nan


def build_whatif_row(supplied: object, features: List[str]) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Validates caller input and returns (model row, input summary). Raises PredictionInputError."""
    values = _validate_keys(supplied, features)
    row = {name: math.nan for name in features}

    measured, not_measured = [], []
    for var in VARIABLES:
        (measured if _variable(var, values, row) else not_measured).append(var.name)

    row["bp_source_invasive"] = _conditional_flag(
        "bp_source_invasive", values, "sbp" in measured or "dbp" in measured, "sbp or dbp is measured")
    row["map_derived"] = _conditional_flag("map_derived", values, "map" in measured, "map is measured")
    if "age" in values:
        row["age"] = _number("age", values["age"])
        if not AGE_RANGE[0] <= row["age"] <= AGE_RANGE[1]:
            _fail("INVALID_VALUE", f"'age' must be between {AGE_RANGE[0]:g} and {AGE_RANGE[1]:g} (adult cohort).")
    if "sex_male" in values:
        row["sex_male"] = _binary("sex_male", values["sex_male"])

    imputed = [name for name, value in row.items() if math.isnan(value)]
    summary = {
        "variables_measured": measured,
        "variables_not_measured": not_measured,
        "demographics_imputed": [d for d in ("age", "sex_male") if d in imputed],
        "features_imputed_with_training_median": len(imputed),
        "encoding": ENCODING_NOTE,
    }
    return row, summary
