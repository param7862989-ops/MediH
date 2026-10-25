"""Synthetic data-source adapter over the existing seeded SQLite demo.

load_raw() delegates to the current loader unchanged, so the synthetic pipeline keeps its
exact behavior. to_canonical() exposes the same data through the common contract. The
synthetic database records no ICU stays or death outcomes, so those stay empty/missing
rather than being invented.
"""

from typing import Dict, Optional

import pandas as pd

from src.data.loader import load_raw_multimodal_dataset
from src.datasources.contract import CANONICAL_UNITS, empty_table, validate_table

# Wide synthetic columns -> canonical variables
VITAL_COLUMNS = {
    "heart_rate": "heart_rate",
    "blood_pressure_sys": "sbp",
    "blood_pressure_dia": "dbp",
    "temperature": "temperature",
    "oxygen_saturation": "spo2",
    "respiratory_rate": "resp_rate",
}
LAB_COLUMNS = {
    "glucose": "glucose",
    "creatinine": "creatinine",
    "cholesterol": "cholesterol",
    "wbc_count": "wbc",
    "bun": "bun",
    "hemoglobin": "hemoglobin",
    "platelets": "platelets",
}


class SyntheticDataSource:
    """Adapter for the seeded synthetic demo database."""

    name = "synthetic"

    def load_raw(self, patient_id: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Returns exactly what the existing synthetic loader returns."""
        return load_raw_multimodal_dataset(patient_id)

    def to_canonical(self, patient_id: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Maps the synthetic tables onto the canonical contract."""
        raw = self.load_raw(patient_id)
        patients = raw["patients"]

        encounters = pd.DataFrame({
            "subject_key": patients["patient_id"],
            "encounter_key": patients["patient_id"],
            "admit_time": patients["admission_date"],
            "discharge_time": patients["discharge_date"],
            "death_time": pd.Series(pd.NaT, index=patients.index, dtype="datetime64[ns]"),
            "in_hospital_death_flag": pd.Series(pd.NA, index=patients.index, dtype="Int64"),
            "discharge_location": pd.Series(pd.NA, index=patients.index, dtype="object"),
            "age_at_admission": patients["age"],
            "sex": patients["gender"],
        })

        observations = pd.concat(
            [
                self._melt(raw["vitals"], VITAL_COLUMNS, "vitals"),
                self._melt(raw["labs"], LAB_COLUMNS, "lab_results"),
            ],
            ignore_index=True,
        )

        meds = raw["medications"]
        medications = pd.DataFrame({
            "subject_key": meds["patient_id"],
            "encounter_key": meds["patient_id"],
            "drug": meds["drug_name"],
            "route": meds["route"],
            "start_time": meds["start_date"],
            "stop_time": meds["end_date"],
        })

        return {
            "encounters": validate_table(encounters, "encounters"),
            "icu_stays": validate_table(empty_table("icu_stays"), "icu_stays"),
            "observations": validate_table(observations, "observations"),
            "medications": validate_table(medications, "medications"),
        }

    @staticmethod
    def _melt(wide: pd.DataFrame, column_map: Dict[str, str], source_table: str) -> pd.DataFrame:
        present = [col for col in column_map if col in wide.columns]
        long = wide.melt(
            id_vars=["patient_id", "recorded_at"],
            value_vars=present,
            var_name="source_item",
            value_name="value",
        ).dropna(subset=["value"])
        long["variable"] = long["source_item"].map(column_map)
        return pd.DataFrame({
            "subject_key": long["patient_id"],
            "encounter_key": long["patient_id"],
            "stay_key": pd.Series(pd.NA, index=long.index, dtype="Int64"),
            "variable": long["variable"],
            "charttime": long["recorded_at"],
            "storetime": pd.Series(pd.NaT, index=long.index, dtype="datetime64[ns]"),
            "value": long["value"].astype(float),
            "unit": long["variable"].map(CANONICAL_UNITS),
            "source_table": source_table,
            "source_item": long["source_item"],
        }).reset_index(drop=True)
