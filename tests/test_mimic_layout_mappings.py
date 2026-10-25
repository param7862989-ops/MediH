"""MIMIC-IV adapter tests: file/schema validation, CSV handling, item mappings and units.

All data are hand-crafted fixtures (tests/mimic_fixtures.py); no real MIMIC-IV records.
"""

import pandas as pd
import pytest

from src.datasources.mimic.layout import MimicDataError, resolve_layout
from src.datasources.mimic.mappings import (
    MappingError,
    convert_to_canonical,
    normalize_unit,
    validate_item_dictionaries,
)
from src.datasources.mimic.reader import MimicIVReader
from tests.mimic_fixtures import build_tables, write_tables

DEMO_FOLDER = "mimic-iv-clinical-database-demo-2.2"


# ------------------------------------------------------------------------------
# Files and schemas
# ------------------------------------------------------------------------------
def test_missing_directory_gives_actionable_error(tmp_path):
    with pytest.raises(MimicDataError, match="does not exist.*hosp"):
        resolve_layout(tmp_path / "nowhere")


def test_missing_required_files_are_listed(tmp_path):
    tables = build_tables()
    del tables["icu/chartevents"], tables["hosp/d_labitems"]
    root = write_tables(tmp_path / DEMO_FOLDER, tables)
    with pytest.raises(MimicDataError) as err:
        resolve_layout(root)
    assert "icu/chartevents.csv.gz" in str(err.value)
    assert "hosp/d_labitems.csv.gz" in str(err.value)


def test_missing_required_columns_are_reported(tmp_path):
    tables = build_tables()
    tables["hosp/admissions"] = tables["hosp/admissions"].drop(columns=["deathtime"])
    root = write_tables(tmp_path / DEMO_FOLDER, tables)
    with pytest.raises(MimicDataError, match="deathtime"):
        resolve_layout(root)


@pytest.mark.parametrize("compress", [True, False])
def test_csv_and_compressed_csv_are_read(tmp_path, compress):
    root = write_tables(tmp_path / DEMO_FOLDER, build_tables(), compress=compress)
    layout = resolve_layout(root)
    suffix = ".csv.gz" if compress else ".csv"
    assert all(path.name.endswith(suffix) for path in layout.files.values())
    assert layout.release == "mimic-iv-demo-2.2"
    with MimicIVReader(root) as reader:
        admissions = reader.load_admissions()
    assert len(admissions) == len(build_tables()["hosp/admissions"])
    assert admissions["hospital_expire_flag"].dtype == "Int64"
    assert pd.api.types.is_datetime64_any_dtype(admissions["dischtime"])


def test_parent_folder_with_single_release_is_accepted(tmp_path):
    write_tables(tmp_path / "downloads" / DEMO_FOLDER, build_tables())
    assert resolve_layout(tmp_path / "downloads").root.name == DEMO_FOLDER


# ------------------------------------------------------------------------------
# Item mappings and units
# ------------------------------------------------------------------------------
def test_all_mapped_items_confirmed_by_dictionaries():
    tables = build_tables()
    validated, warnings = validate_item_dictionaries(tables["icu/d_items"], tables["hosp/d_labitems"])
    assert {220045, 223761, 223762, 220050, 220179, 50912, 50813, 50931} <= set(validated)
    assert validated[223761][0].name == "temperature"
    # The fixture dictionaries contain every mapped item, and the alarm item 220046 is never mapped
    assert warnings == []
    assert 220046 not in validated


def test_item_missing_from_dictionary_is_excluded_with_warning():
    tables = build_tables()
    d_items = tables["icu/d_items"][tables["icu/d_items"]["itemid"] != 224690]
    validated, warnings = validate_item_dictionaries(d_items, tables["hosp/d_labitems"])
    assert 224690 not in validated
    assert any("224690" in w for w in warnings)


def test_dictionary_label_mismatch_stops_the_build():
    tables = build_tables()
    d_items = tables["icu/d_items"].copy()
    d_items.loc[d_items["itemid"] == 220045, "label"] = "Heart rate Alarm - High"
    with pytest.raises(MappingError, match="220045"):
        validate_item_dictionaries(d_items, tables["hosp/d_labitems"])


def test_non_blood_lab_item_stops_the_build():
    tables = build_tables()
    d_labitems = tables["hosp/d_labitems"].copy()
    d_labitems.loc[d_labitems["itemid"] == 50912, "fluid"] = "Urine"
    with pytest.raises(MappingError, match="50912"):
        validate_item_dictionaries(tables["icu/d_items"], d_labitems)


def test_unit_normalization():
    assert normalize_unit("°F") == "f"
    assert normalize_unit("Â°C") == "c"
    assert normalize_unit("K/uL") == "k/ul"
    assert normalize_unit("mmol/L") == "mmol/l"
    assert normalize_unit("insp/min") == "insp/min"
    assert normalize_unit(None) is None
    assert normalize_unit(float("nan")) is None
    assert normalize_unit(pd.NA) is None


def test_fahrenheit_to_celsius_conversion():
    converted = convert_to_canonical(pd.Series([98.6, 212.0, 32.0]), "degF")
    assert converted.round(6).tolist() == [37.0, 100.0, 0.0]
    unchanged = convert_to_canonical(pd.Series([37.5]), "degC")
    assert unchanged.tolist() == [37.5]
