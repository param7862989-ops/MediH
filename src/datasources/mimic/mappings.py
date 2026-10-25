"""Versioned mapping from MIMIC-IV item IDs to MediHaven canonical variables (feature spec v1).

Every item ID is confirmed against the source dictionaries (icu/d_items, hosp/d_labitems)
before use: an item missing from the dictionary is excluded with a warning, and a dictionary
label that does not match the expected concept stops the build (the mapping would be wrong).

Plausibility ranges are inclusive and expressed in canonical units after conversion. They
remove physiologically impossible values (device artefacts, typing errors, unit mix-ups);
they are deliberately wide so that genuinely extreme ICU values survive.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

FEATURE_SPEC_VERSION = "mimic-iv-t1-features-v1"

# Summaries computed per variable over the observation window
AGGREGATIONS: Tuple[str, ...] = ("min", "max", "mean", "last", "count")


class MappingError(RuntimeError):
    """Raised when a source dictionary contradicts the expected item mapping."""


@dataclass(frozen=True)
class ItemMapping:
    itemid: int
    source_table: str  # "chartevents" or "labevents"
    label_keywords: Tuple[str, ...]  # all must appear in the dictionary label (case-insensitive)
    source_unit: str  # unit implied by the item ID; drives conversion
    accepted_units: Tuple[str, ...]  # normalized valueuom strings accepted for this item
    bp_source: Optional[str] = None  # "invasive" / "noninvasive" for blood-pressure items
    exclude_keywords: Tuple[str, ...] = ("alarm",)


@dataclass(frozen=True)
class VariableSpec:
    name: str
    unit: str
    plausible_min: float
    plausible_max: float
    items: Tuple[ItemMapping, ...]


def _vital(itemid, keywords, unit, accepted, bp_source=None):
    return ItemMapping(itemid, "chartevents", keywords, unit, accepted, bp_source)


def _lab(itemid, keywords, unit, accepted):
    return ItemMapping(itemid, "labevents", keywords, unit, accepted)


BP_UNITS = ("mmhg",)

VARIABLES: Tuple[VariableSpec, ...] = (
    VariableSpec("heart_rate", "bpm", 10, 300, (
        _vital(220045, ("heart rate",), "bpm", ("bpm",)),
    )),
    VariableSpec("resp_rate", "breaths/min", 2, 80, (
        _vital(220210, ("respiratory rate",), "breaths/min", ("insp/min", "breaths/min", "bpm")),
        _vital(224690, ("respiratory rate", "total"), "breaths/min", ("insp/min", "breaths/min", "bpm")),
    )),
    VariableSpec("spo2", "%", 50, 100, (
        _vital(220277, ("o2 saturation",), "%", ("%",)),
    )),
    VariableSpec("temperature", "degC", 25, 45, (
        _vital(223761, ("temperature", "fahrenheit"), "degF", ("f", "degf")),
        _vital(223762, ("temperature", "celsius"), "degC", ("c", "degc")),
    )),
    VariableSpec("sbp", "mmHg", 30, 300, (
        _vital(220050, ("arterial", "systolic"), "mmHg", BP_UNITS, "invasive"),
        _vital(220179, ("non invasive", "systolic"), "mmHg", BP_UNITS, "noninvasive"),
    )),
    VariableSpec("dbp", "mmHg", 10, 200, (
        _vital(220051, ("arterial", "diastolic"), "mmHg", BP_UNITS, "invasive"),
        _vital(220180, ("non invasive", "diastolic"), "mmHg", BP_UNITS, "noninvasive"),
    )),
    VariableSpec("map", "mmHg", 20, 250, (
        _vital(220052, ("arterial", "mean"), "mmHg", BP_UNITS, "invasive"),
        _vital(220181, ("non invasive", "mean"), "mmHg", BP_UNITS, "noninvasive"),
    )),
    VariableSpec("creatinine", "mg/dL", 0.1, 30, (
        _lab(50912, ("creatinine",), "mg/dL", ("mg/dl",)),
    )),
    VariableSpec("bun", "mg/dL", 1, 300, (
        _lab(51006, ("urea nitrogen",), "mg/dL", ("mg/dl",)),
    )),
    VariableSpec("wbc", "K/uL", 0.1, 500, (
        _lab(51301, ("white blood cells",), "K/uL", ("k/ul",)),
        _lab(51300, ("wbc",), "K/uL", ("k/ul",)),
    )),
    VariableSpec("hemoglobin", "g/dL", 2, 25, (
        _lab(51222, ("hemoglobin",), "g/dL", ("g/dl",)),
    )),
    VariableSpec("platelets", "K/uL", 1, 2000, (
        _lab(51265, ("platelet",), "K/uL", ("k/ul",)),
    )),
    VariableSpec("lactate", "mmol/L", 0.1, 30, (
        _lab(50813, ("lactate",), "mmol/L", ("mmol/l",)),
    )),
    VariableSpec("glucose", "mg/dL", 10, 2000, (
        _lab(50931, ("glucose",), "mg/dL", ("mg/dl",)),
        _lab(50809, ("glucose",), "mg/dL", ("mg/dl",)),
    )),
)

VARIABLES_BY_NAME: Dict[str, VariableSpec] = {v.name: v for v in VARIABLES}
BP_VARIABLES = ("sbp", "dbp", "map")


def all_items() -> List[Tuple[VariableSpec, ItemMapping]]:
    return [(var, item) for var in VARIABLES for item in var.items]


def feature_columns() -> List[str]:
    """Ordered feature columns produced by feature spec v1."""
    columns = []
    for var in VARIABLES:
        columns += [f"{var.name}_{agg}" for agg in AGGREGATIONS]
        columns.append(f"{var.name}_missing")
    return columns + ["map_derived", "bp_source_invasive", "age", "sex_male"]


def normalize_unit(raw_unit) -> Optional[str]:
    """Lower-cases a unit string and strips symbols/encoding debris ('°F' -> 'f', 'K/uL' -> 'k/ul')."""
    if raw_unit is None or (isinstance(raw_unit, float) and pd.isna(raw_unit)) or raw_unit is pd.NA:
        return None
    text = str(raw_unit).strip().lower().replace("µ", "u").replace("μ", "u")
    text = re.sub(r"[^a-z0-9/%]", "", text)
    return text or None


def convert_to_canonical(values: pd.Series, source_unit: str) -> pd.Series:
    """Converts a series from the item's source unit to the canonical unit."""
    if source_unit == "degF":
        return (values - 32.0) * 5.0 / 9.0
    return values


def validate_item_dictionaries(d_items: pd.DataFrame, d_labitems: pd.DataFrame):
    """Confirms every mapped item ID against the source dictionaries.

    Returns (validated_items, warnings) where validated_items maps itemid -> (VariableSpec, ItemMapping).
    Raises MappingError when a dictionary label contradicts the expected concept.
    """
    dictionaries = {
        "chartevents": d_items.set_index("itemid"),
        "labevents": d_labitems.set_index("itemid"),
    }
    validated, warnings, mismatches = {}, [], []
    for var, item in all_items():
        dictionary = dictionaries[item.source_table]
        if item.itemid not in dictionary.index:
            warnings.append(f"itemid {item.itemid} ({var.name}) is not in the {item.source_table} dictionary; excluded")
            continue
        entry = dictionary.loc[item.itemid]
        label = str(entry["label"]).lower()
        if not all(k in label for k in item.label_keywords) or any(k in label for k in item.exclude_keywords):
            mismatches.append(f"itemid {item.itemid} expected {var.name} but the dictionary label is '{entry['label']}'")
            continue
        if item.source_table == "chartevents" and "linksto" in entry and str(entry["linksto"]).lower() != "chartevents":
            mismatches.append(f"itemid {item.itemid} ({var.name}) links to '{entry['linksto']}', not chartevents")
            continue
        if item.source_table == "labevents" and "fluid" in entry and "blood" not in str(entry["fluid"]).lower():
            mismatches.append(f"itemid {item.itemid} ({var.name}) is a '{entry['fluid']}' test, expected blood")
            continue
        validated[item.itemid] = (var, item)
    if mismatches:
        raise MappingError(
            "Item mapping contradicts the source dictionary: " + "; ".join(mismatches)
            + ". Review src/datasources/mimic/mappings.py before building the dataset."
        )
    for var in VARIABLES:
        if not any(v.name == var.name for v, _ in validated.values()):
            warnings.append(f"no usable item IDs for {var.name}; its features will be missing for every stay")
    return validated, warnings
