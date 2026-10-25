"""What-if input contract tests (src/training/whatif.py).

Unmeasured variables must be encoded exactly as Phase 1 encodes unmeasured stays; measured variables
must be complete and consistent; contradictory or impossible inputs are rejected.
"""

import math

import pandas as pd
import pytest

from src.datasources.mimic.mappings import VARIABLES, feature_columns
from src.tasks.features import aggregate_features
from src.training.whatif import PredictionInputError, build_whatif_row

FEATURES = feature_columns()
VARIABLE_NAMES = [v.name for v in VARIABLES]
BP = {"sbp_count": 1, "sbp_last": 120, "dbp_count": 1, "dbp_last": 80, "bp_source_invasive": 0}


def _code(payload):
    with pytest.raises(PredictionInputError) as err:
        build_whatif_row(payload, FEATURES)
    return err.value.code


def test_omitted_variables_are_encoded_as_not_measured():
    row, summary = build_whatif_row({"age": 70}, FEATURES)
    for name in VARIABLE_NAMES:
        assert row[f"{name}_count"] == 0 and row[f"{name}_missing"] == 1
        assert all(math.isnan(row[f"{name}_{s}"]) for s in ("min", "max", "mean", "last"))
    assert math.isnan(row["bp_source_invasive"]) and math.isnan(row["map_derived"])
    assert summary["variables_measured"] == [] and summary["variables_not_measured"] == VARIABLE_NAMES
    assert summary["demographics_imputed"] == ["sex_male"]
    assert "_missing = 1" in summary["encoding"] and "_count = 0" in summary["encoding"]


def test_unmeasured_encoding_matches_phase1_aggregation():
    """A stay with no observations, aggregated by Phase 1, must look exactly like an empty what-if row."""
    empty_obs = pd.DataFrame(columns=["stay_id", "variable", "charttime", "value", "itemid", "bp_source"])
    cohort = pd.DataFrame({"stay_id": [1], "age": [70.0], "sex_male": [1.0]})
    phase1 = aggregate_features(empty_obs, cohort, pd.Series(dtype="float64"), pd.Series(dtype="float64")).iloc[0]
    row, _ = build_whatif_row({"age": 70, "sex_male": 1}, FEATURES)
    for column in FEATURES:
        expected, actual = float(phase1[column]), row[column]
        assert (math.isnan(expected) and math.isnan(actual)) or expected == actual, column


def test_single_reading_fills_all_summaries():
    row, summary = build_whatif_row({"heart_rate_last": 110, "heart_rate_count": 1}, FEATURES)
    assert [row[f"heart_rate_{s}"] for s in ("min", "max", "mean", "last")] == [110.0] * 4
    assert (row["heart_rate_count"], row["heart_rate_missing"]) == (1.0, 0.0)
    assert summary["variables_measured"] == ["heart_rate"]


def test_multiple_readings_with_complete_consistent_summaries():
    payload = {"lactate_count": 3, "lactate_min": 1.2, "lactate_max": 4.0, "lactate_mean": 2.5,
               "lactate_last": 3.1, "lactate_missing": 0, **BP}
    row, summary = build_whatif_row(payload, FEATURES)
    assert (row["lactate_min"], row["lactate_max"], row["lactate_mean"], row["lactate_last"]) == (1.2, 4.0, 2.5, 3.1)
    assert (row["lactate_count"], row["lactate_missing"]) == (3.0, 0.0)
    assert summary["variables_measured"] == ["sbp", "dbp", "lactate"]
    assert row["bp_source_invasive"] == 0.0
    assert summary["features_imputed_with_training_median"] == sum(math.isnan(v) for v in row.values())


def test_explicitly_not_measured_variable_is_accepted():
    row, summary = build_whatif_row({"lactate_count": 0, "lactate_missing": 1, "age": 50}, FEATURES)
    assert (row["lactate_count"], row["lactate_missing"]) == (0.0, 1.0)
    assert "lactate" in summary["variables_not_measured"]


@pytest.mark.parametrize("payload", [
    {"heart_rate_count": 2, "heart_rate_last": 90, "heart_rate_missing": 1},   # positive count + missing=1
    {"heart_rate_last": 90, "heart_rate_count": 0},                            # values + count 0
    {"heart_rate_count": 0, "heart_rate_missing": 0},                          # not measured yet missing=0
    {"heart_rate_count": 1, "heart_rate_last": 90, "heart_rate_min": 85},      # one reading, unequal values
    {"heart_rate_count": 3, "heart_rate_min": 95, "heart_rate_max": 120,
     "heart_rate_mean": 90, "heart_rate_last": 100},                           # mean below min
    {"heart_rate_count": 3, "heart_rate_min": 80, "heart_rate_max": 120,
     "heart_rate_mean": 90, "heart_rate_last": 130},                           # last above max
    {"bp_source_invasive": 1},                                                 # BP flag without BP
    {"map_derived": 0},                                                        # MAP flag without MAP
])
def test_contradictory_inputs_are_rejected(payload):
    assert _code(payload) == "CONTRADICTORY_INPUT"


@pytest.mark.parametrize("payload", [
    {"heart_rate_last": 90},                                                    # values without count
    {"heart_rate_count": 2},                                                    # count without values
    {"heart_rate_count": 2, "heart_rate_last": 90},                             # count >= 2, summaries incomplete
    {"sbp_count": 1, "sbp_last": 120},                                          # BP measured, source missing
    {"map_count": 1, "map_last": 80},                                           # MAP measured, map_derived missing
])
def test_incomplete_variables_are_rejected(payload):
    assert _code(payload) == "INCOMPLETE_VARIABLE"


@pytest.mark.parametrize("payload", [
    {"heart_rate_count": -1, "heart_rate_last": 90},
    {"heart_rate_count": 2.5, "heart_rate_last": 90},
    {"heart_rate_count": 10001, "heart_rate_last": 90},
    {"heart_rate_count": True, "heart_rate_last": 90},
    {"heart_rate_count": "3", "heart_rate_last": 90},
    {"heart_rate_missing": 2},
    {"heart_rate_count": 1, "heart_rate_last": 500},                            # implausible measurement
    {"temperature_count": 1, "temperature_last": 98.6},                        # Fahrenheit given as Celsius
    {"age": 12},
    {"sex_male": 2},
    {"heart_rate_count": 1, "heart_rate_last": float("inf")},
])
def test_invalid_counts_indicators_and_values_are_rejected(payload):
    assert _code(payload) == "INVALID_VALUE"


@pytest.mark.parametrize("payload, code", [
    ({"src_subject_id": 1}, "FORBIDDEN_FEATURE"),
    ({"outcome_anomaly_deathtime_after_discharge": 0}, "FORBIDDEN_FEATURE"),
    ({"label_inhospital_mortality": 1}, "FORBIDDEN_FEATURE"),
    ({"index_time": 0}, "FORBIDDEN_FEATURE"),
    ({"mystery_score": 1}, "UNKNOWN_FEATURE"),
    ({}, "INVALID_INPUT"),
    ([1, 2], "INVALID_INPUT"),
])
def test_forbidden_unknown_and_malformed_inputs_are_rejected(payload, code):
    assert _code(payload) == code


def test_null_values_mean_not_supplied():
    row, summary = build_whatif_row({"heart_rate_last": None, "age": 60}, FEATURES)
    assert "heart_rate" in summary["variables_not_measured"] and row["heart_rate_missing"] == 1
