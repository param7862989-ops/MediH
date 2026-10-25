"""Synthetic data-source adapter tests: unchanged raw loading and valid canonical mapping."""

import pandas as pd

from src.data.loader import load_raw_multimodal_dataset
from src.datasources.contract import CANONICAL_TABLES, CANONICAL_UNITS, validate_table
from src.datasources.synthetic import LAB_COLUMNS, VITAL_COLUMNS, SyntheticDataSource


def test_load_raw_matches_existing_loader():
    expected = load_raw_multimodal_dataset()
    actual = SyntheticDataSource().load_raw()
    assert set(actual) == set(expected)
    for name in expected:
        pd.testing.assert_frame_equal(actual[name], expected[name])


def test_canonical_tables_follow_contract():
    raw = load_raw_multimodal_dataset()
    canonical = SyntheticDataSource().to_canonical()

    assert set(canonical) == set(CANONICAL_TABLES)
    for name, table in canonical.items():
        validate_table(table, name)

    assert len(canonical["encounters"]) == len(raw["patients"])
    assert canonical["icu_stays"].empty
    assert canonical["encounters"]["death_time"].isna().all()
    assert canonical["encounters"]["in_hospital_death_flag"].isna().all()

    expected_rows = int(raw["vitals"][list(VITAL_COLUMNS)].notna().sum().sum()) + int(
        raw["labs"][list(LAB_COLUMNS)].notna().sum().sum()
    )
    observations = canonical["observations"]
    assert len(observations) == expected_rows
    assert set(observations["variable"]) <= set(CANONICAL_UNITS)
    assert (observations["unit"] == observations["variable"].map(CANONICAL_UNITS)).all()


def test_single_patient_canonical_view():
    canonical = SyntheticDataSource().to_canonical(patient_id=1)
    assert set(canonical["observations"]["subject_key"]) == {1}
    assert len(canonical["encounters"]) == 1
