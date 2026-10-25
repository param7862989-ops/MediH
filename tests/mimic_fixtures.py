"""Hand-crafted MIMIC-IV-layout fixtures for Task 1 tests.

Every identifier and value here is invented; no real MIMIC-IV records are used. Each fake
person exercises one rule (index (prediction) time = ICU intime + 24 h = T0 + 24 h):

    S1  negative; rich measurements incl. dropped rows (implausible, unit, storetime, window)
    S2  positive (dies after index); invasive BP present -> MAP derived from arterial pairs
    S3  dies before the index time                     -> died_before_index
    S4  leaves the ICU exactly at the index time       -> left_icu_before_index
    S5  leaves the ICU 1 minute after the index time   -> eligible negative
    S6  age 16                                         -> under_18
    S7  expire flag 1 but no deathtime                 -> ambiguous
    S8  expire flag 0 but a deathtime                  -> ambiguous
    S9  hospital discharge before the index time       -> hospital_discharge_before_index
    S10 two ICU stays in one admission                 -> first eligible, second not_first
    S11 ICU stay whose admission record is absent      -> missing_admission_record
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

T0 = pd.Timestamp("2150-03-01 10:00:00")
INDEX = T0 + pd.Timedelta(hours=24)
H = pd.Timedelta(hours=1)
D = pd.Timedelta(days=1)


def sid(n: int) -> int:
    return 91000000 + n


def hid(n: int) -> int:
    return 92000000 + n


def stid(n: int) -> int:
    return 93000000 + n


S10_SECOND_STAY = 93000110
ALL_FIXTURE_IDS: List[int] = (
    [sid(n) for n in range(1, 12)] + [hid(n) for n in range(1, 12)] + [stid(n) for n in range(1, 12)]
    + [S10_SECOND_STAY]
)


def _patients() -> pd.DataFrame:
    rows = []
    ages = {1: 60, 2: 72, 3: 80, 4: 55, 5: 45, 6: 16, 7: 66, 8: 59, 9: 70, 10: 50, 11: 64}
    for n, age in ages.items():
        dod = None
        if n == 2:
            dod = "2150-03-05"
        if n == 3:
            dod = "2150-03-01"
        rows.append({"subject_id": sid(n), "gender": "M" if n % 2 else "F", "anchor_age": age,
                     "anchor_year": 2150, "anchor_year_group": "2017 - 2019", "dod": dod})
    return pd.DataFrame(rows)


def _admissions() -> pd.DataFrame:
    def adm(n, disch, flag, death=None, loc="HOME"):
        return {"subject_id": sid(n), "hadm_id": hid(n), "admittime": T0 - 2 * H, "dischtime": disch,
                "deathtime": death, "admission_type": "EW EMER.", "discharge_location": loc,
                "hospital_expire_flag": flag}
    return pd.DataFrame([
        adm(1, T0 + 5 * D, 0),
        adm(2, T0 + 4 * D, 1, death=T0 + 4 * D, loc="DIED"),
        adm(3, T0 + 20 * H, 1, death=T0 + 20 * H, loc="DIED"),
        adm(4, T0 + 3 * D, 0),
        adm(5, T0 + 3 * D, 0),
        adm(6, T0 + 3 * D, 0),
        adm(7, T0 + 3 * D, 1, death=None),
        adm(8, T0 + 3 * D, 0, death=T0 + 3 * D),
        adm(9, T0 + 23 * H, 0),
        adm(10, T0 + 6 * D, 0),
        # S11 deliberately has no admission row
    ])


def _icustays() -> pd.DataFrame:
    def stay(n, outtime, stay_id=None, intime=T0):
        return {"subject_id": sid(n), "hadm_id": hid(n), "stay_id": stay_id or stid(n),
                "first_careunit": "Medical Intensive Care Unit (MICU)", "last_careunit": "Medical Intensive Care Unit (MICU)",
                "intime": intime, "outtime": outtime, "los": None}
    return pd.DataFrame([
        stay(1, T0 + 48 * H),
        stay(2, T0 + 72 * H),
        stay(3, T0 + 20 * H),
        stay(4, T0 + 24 * H),
        stay(5, T0 + 24 * H + pd.Timedelta(minutes=1)),
        stay(6, T0 + 48 * H),
        stay(7, T0 + 48 * H),
        stay(8, T0 + 48 * H),
        stay(9, T0 + 30 * H),
        stay(10, T0 + 36 * H),
        stay(10, T0 + 4 * D, stay_id=S10_SECOND_STAY, intime=T0 + 3 * D),
        stay(11, T0 + 48 * H),
    ])


def _chart(n, itemid, charttime, value, uom, storetime="auto", stay_id=None):
    store = charttime + pd.Timedelta(minutes=5) if isinstance(storetime, str) else storetime
    return {"subject_id": sid(n), "hadm_id": hid(n), "stay_id": stay_id or stid(n), "caregiver_id": 1,
            "charttime": charttime, "storetime": store, "itemid": itemid,
            "value": None if value is None else str(value), "valuenum": value, "valueuom": uom, "warning": 0}


def _chartevents() -> pd.DataFrame:
    return pd.DataFrame([
        # S1 heart rate: valid 80/90/100; the rest must be dropped or never extracted
        _chart(1, 220045, T0 + 1 * H, 80, "bpm"),
        _chart(1, 220045, T0 + 5 * H, 90, "bpm"),
        _chart(1, 220045, T0 + 23 * H, 100, "bpm"),
        _chart(1, 220045, T0 + 10 * H, 0, "bpm"),  # implausible
        _chart(1, 220045, T0 + 12 * H, 85, "mmHg"),  # unit mismatch
        _chart(1, 220045, T0 + 20 * H, 150, "bpm", storetime=INDEX + 1 * H),  # stored after index
        _chart(1, 220045, INDEX + 6 * H, 200, "bpm"),  # after index
        _chart(1, 220045, T0 - 1 * H, 60, "bpm"),  # before ICU intime
        # S1 temperature: 98.6 F -> 37.0 C, 38.0 C, 9 F implausible
        _chart(1, 223761, T0 + 2 * H, 98.6, "°F"),
        _chart(1, 223762, T0 + 6 * H, 38.0, "°C"),
        _chart(1, 223761, T0 + 8 * H, 9.0, "°F"),
        # S1 non-invasive BP with measured MAP
        _chart(1, 220179, T0 + 3 * H, 120, "mmHg"),
        _chart(1, 220180, T0 + 3 * H, 80, "mmHg"),
        _chart(1, 220181, T0 + 3 * H, 93, "mmHg"),
        # S1 respiratory rate and SpO2 (one SpO2 row without a unit)
        _chart(1, 220210, T0 + 4 * H, 18, "insp/min"),
        _chart(1, 220277, T0 + 4 * H, 97, "%"),
        _chart(1, 220277, T0 + 9 * H, 95, None),
        # S2 arterial line present -> invasive source; NIBP rows and NIBP MAP ignored
        _chart(2, 220045, T0 + 2 * H, 110, "bpm"),
        _chart(2, 220050, T0 + 2 * H, 100, "mmHg"),
        _chart(2, 220051, T0 + 2 * H, 60, "mmHg"),
        _chart(2, 220179, T0 + 2 * H, 130, "mmHg"),
        _chart(2, 220181, T0 + 2 * H, 90, "mmHg"),
        # S3 (excluded) and S5
        _chart(3, 220045, T0 + 1 * H, 140, "bpm"),
        _chart(5, 220045, T0 + 1 * H, 70, "bpm"),
    ])


def _lab(n, itemid, charttime, value, uom, storetime="auto"):
    store = charttime + H if isinstance(storetime, str) else storetime
    return {"labevent_id": 0, "subject_id": sid(n), "hadm_id": None, "specimen_id": 0, "itemid": itemid,
            "charttime": charttime, "storetime": store, "value": str(value), "valuenum": value,
            "valueuom": uom, "flag": None}


def _labevents() -> pd.DataFrame:
    rows = [
        _lab(1, 50912, T0 + 6 * H, 1.1, "mg/dL"),
        _lab(1, 50912, INDEX + 6 * H, 2.5, "mg/dL"),  # after index
        _lab(1, 50912, T0 - 3 * H, 0.9, "mg/dL"),  # before ICU intime
        _lab(1, 50813, T0 + 6 * H, 2.0, "mmol/L", storetime=INDEX + 2 * H),  # stored after index
        _lab(1, 50931, T0 + 6 * H, 140, "mg/dL", storetime=None),  # no storetime: charttime rule
        _lab(2, 50813, T0 + 3 * H, 4.5, "mmol/L"),
    ]
    df = pd.DataFrame(rows)
    df["labevent_id"] = range(1, len(df) + 1)
    return df


def _d_items() -> pd.DataFrame:
    items = [
        (220045, "Heart Rate"), (220210, "Respiratory Rate"), (224690, "Respiratory Rate (Total)"),
        (220277, "O2 saturation pulseoxymetry"), (223761, "Temperature Fahrenheit"),
        (223762, "Temperature Celsius"), (220050, "Arterial Blood Pressure systolic"),
        (220051, "Arterial Blood Pressure diastolic"), (220052, "Arterial Blood Pressure mean"),
        (220179, "Non Invasive Blood Pressure systolic"), (220180, "Non Invasive Blood Pressure diastolic"),
        (220181, "Non Invasive Blood Pressure mean"), (220046, "Heart rate Alarm - High"),
    ]
    return pd.DataFrame([
        {"itemid": i, "label": label, "abbreviation": label[:10], "linksto": "chartevents",
         "category": "Routine Vital Signs", "unitname": None}
        for i, label in items
    ])


def _d_labitems() -> pd.DataFrame:
    items = [
        (50912, "Creatinine", "Chemistry"), (51006, "Urea Nitrogen", "Chemistry"),
        (51301, "White Blood Cells", "Hematology"), (51300, "WBC Count", "Hematology"),
        (51222, "Hemoglobin", "Hematology"), (51265, "Platelet Count", "Hematology"),
        (50813, "Lactate", "Blood Gas"), (50931, "Glucose", "Chemistry"), (50809, "Glucose", "Blood Gas"),
    ]
    return pd.DataFrame([{"itemid": i, "label": label, "fluid": "Blood", "category": cat} for i, label, cat in items])


def build_tables() -> Dict[str, pd.DataFrame]:
    """Returns the fixture tables keyed by MIMIC-IV relative table name."""
    return {
        "hosp/patients": _patients(),
        "hosp/admissions": _admissions(),
        "hosp/labevents": _labevents(),
        "hosp/d_labitems": _d_labitems(),
        "icu/icustays": _icustays(),
        "icu/chartevents": _chartevents(),
        "icu/d_items": _d_items(),
    }


def write_tables(root: Path, tables: Dict[str, pd.DataFrame], compress: bool = True) -> Path:
    """Writes tables in the official layout (<root>/hosp/*.csv[.gz], <root>/icu/*.csv[.gz])."""
    suffix = ".csv.gz" if compress else ".csv"
    for table, df in tables.items():
        path = root / f"{table}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S",
                  compression="gzip" if compress else None)
    return root


def write_demo_fixture(root: Path, compress: bool = True) -> Path:
    return write_tables(root, build_tables(), compress=compress)
