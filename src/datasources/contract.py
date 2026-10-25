"""MediHaven's common clinical data contract shared by every data-source adapter.

Identifier policy
-----------------
Source identifiers (``subject_key``, ``encounter_key``, ``stay_key``) are the identifiers of
the originating dataset (synthetic ``patient_id``; MIMIC-IV ``subject_id``/``hadm_id``/``stay_id``).
They exist only for local joins and patient-level splitting. They are never printed, logged,
or shown in the web UI. Display identifiers (MRN-style labels in the portals) are separate
surrogate keys assigned by the serving layer; for MIMIC-IV they will be generated during
serving integration and must not be derivable from source identifiers.

Observations are kept in long format (one row per measurement) until feature extraction.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import pandas as pd


class DataContractError(ValueError):
    """Raised when an adapter returns a table that violates the canonical contract."""


@dataclass(frozen=True)
class TableSpec:
    """A canonical table: required columns mapped to their meaning."""

    name: str
    description: str
    columns: Dict[str, str] = field(default_factory=dict)

    @property
    def column_names(self) -> Tuple[str, ...]:
        return tuple(self.columns)


ENCOUNTERS = TableSpec(
    name="encounters",
    description="One row per hospital admission, including the outcome fields needed for labeling.",
    columns={
        "subject_key": "Source person identifier (internal only)",
        "encounter_key": "Source admission identifier (internal only)",
        "admit_time": "Hospital admission timestamp",
        "discharge_time": "Hospital discharge timestamp",
        "death_time": "In-hospital death timestamp, if any",
        "in_hospital_death_flag": "Source flag for death during the admission (1/0, may be missing)",
        "discharge_location": "Source discharge disposition",
        "age_at_admission": "Age in years at admission",
        "sex": "Recorded sex (M/F/Other)",
    },
)

ICU_STAYS = TableSpec(
    name="icu_stays",
    description="One row per ICU stay.",
    columns={
        "subject_key": "Source person identifier (internal only)",
        "encounter_key": "Source admission identifier (internal only)",
        "stay_key": "Source ICU stay identifier (internal only)",
        "icu_intime": "ICU admission timestamp",
        "icu_outtime": "ICU discharge timestamp",
        "first_careunit": "First ICU care unit",
    },
)

OBSERVATIONS = TableSpec(
    name="observations",
    description="Long-format measurements: one row per variable per timestamp.",
    columns={
        "subject_key": "Source person identifier (internal only)",
        "encounter_key": "Source admission identifier (internal only)",
        "stay_key": "Source ICU stay identifier when known (internal only)",
        "variable": "Canonical variable name (see CANONICAL_UNITS)",
        "charttime": "When the measurement was taken",
        "storetime": "When the measurement was recorded/available, if known",
        "value": "Numeric value in the canonical unit",
        "unit": "Canonical unit",
        "source_table": "Originating source table",
        "source_item": "Originating source item (column name or item ID)",
    },
)

MEDICATIONS = TableSpec(
    name="medications",
    description="One row per medication order.",
    columns={
        "subject_key": "Source person identifier (internal only)",
        "encounter_key": "Source admission identifier (internal only)",
        "drug": "Drug name",
        "route": "Administration route",
        "start_time": "Order start timestamp",
        "stop_time": "Order stop timestamp, if any",
    },
)

CANONICAL_TABLES: Dict[str, TableSpec] = {
    spec.name: spec for spec in (ENCOUNTERS, ICU_STAYS, OBSERVATIONS, MEDICATIONS)
}

# Canonical units for observation variables shared by all adapters
CANONICAL_UNITS: Dict[str, str] = {
    "heart_rate": "bpm",
    "resp_rate": "breaths/min",
    "spo2": "%",
    "temperature": "degC",
    "sbp": "mmHg",
    "dbp": "mmHg",
    "map": "mmHg",
    "creatinine": "mg/dL",
    "bun": "mg/dL",
    "wbc": "K/uL",
    "hemoglobin": "g/dL",
    "platelets": "K/uL",
    "lactate": "mmol/L",
    "glucose": "mg/dL",
    "cholesterol": "mg/dL",
}


def empty_table(table_name: str) -> pd.DataFrame:
    """Returns an empty DataFrame with the canonical columns of a table."""
    return pd.DataFrame(columns=list(CANONICAL_TABLES[table_name].column_names))


def validate_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Checks that df carries every canonical column of table_name; returns df unchanged."""
    spec = CANONICAL_TABLES[table_name]
    missing = [col for col in spec.column_names if col not in df.columns]
    if missing:
        raise DataContractError(f"Canonical table '{table_name}' is missing columns: {missing}")
    if table_name == "observations":
        unknown = sorted(set(df["variable"].dropna()) - set(CANONICAL_UNITS))
        if unknown:
            raise DataContractError(f"Observations contain unknown canonical variables: {unknown}")
    return df
