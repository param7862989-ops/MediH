"""Task 1: in-hospital mortality after the first 24 hours of an adult ICU stay (MIMIC-IV).

Definition (task version 3)
---------------------------
- Unit of analysis: the first ICU stay (earliest ``intime``, ties broken by ``stay_id``) of each
  hospital admission.
- Population: adults (age at admission >= 18, age = anchor_age + admission year - anchor_year).
- Index (prediction) time: ICU ``intime`` + exactly 24 hours.
- Eligibility at the index time: alive, still in the ICU (``outtime`` strictly after the index
  time) and the hospital admission not yet ended (``dischtime`` strictly after the index time).
  These checks run before labeling, so the anomaly rule below never rescues a stay whose
  recorded discharge precedes the index time.
- Observation window: ``[intime, index_time]`` (see features.py for the storetime rule).

Outcome rule
- Positive (death during the index hospitalization after the index time), either:
  a) consistent: ``hospital_expire_flag = 1`` AND ``index_time < deathtime <= dischtime`` AND the
     discharge location is DIED or not recorded; or
  b) timestamp anomaly: ``hospital_expire_flag = 1`` AND discharge location DIED AND
     ``deathtime > index_time`` AND ``deathtime > dischtime``. Both the flag and the disposition
     record an in-hospital death; only the timestamps disagree. These stays are labeled positive,
     flagged in the ``outcome_anomaly_deathtime_after_discharge`` column and counted (with the
     death-after-discharge gap) in the manifest and research log. No upper bound on the gap is
     imposed; the gap distribution is reported instead.
- Negative (alive discharge): ``hospital_expire_flag = 0`` AND no ``deathtime`` AND
  ``dischtime > index_time`` AND the discharge location is not DIED AND ``dod`` is not before the
  discharge date. A death after a documented alive discharge (``dod`` on/after the discharge date)
  is an out-of-hospital death and stays negative.
- Ambiguous (excluded, counted by reason, never guessed): missing ``dischtime`` or flag; flag 1
  without ``deathtime``; ``deathtime`` with flag 0; ``deathtime`` after ``dischtime`` without a DIED
  disposition; flag 1 with an alive discharge location; DIED location with flag 0; ``dod`` before
  an alive discharge date.

This is death later in the same hospitalization, not a fixed 7- or 30-day outcome.
History: v1 accepted deaths up to 24 h after dischtime regardless of disposition; v2 excluded every
deathtime later than dischtime; v3 (this) accepts them only with flag 1 + DIED, as flagged anomalies.
"""

from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

TASK_ID = "t1_inhospital_mortality_after_icu24h"
TASK_VERSION = "3"
INDEX_OFFSET = pd.Timedelta(hours=24)
ADULT_MIN_AGE = 18
DIED_LOCATION = "DIED"
ANOMALY_COLUMN = "outcome_anomaly_deathtime_after_discharge"

# Reporting buckets for the deathtime - dischtime gap of anomaly positives (upper bounds, hours)
ANOMALY_GAP_BUCKETS = ((1, "(0, 1] h"), (6, "(1, 6] h"), (24, "(6, 24] h"))
ANOMALY_GAP_OVERFLOW = "> 24 h"

TASK_DEFINITION = {
    "task_id": TASK_ID,
    "task_version": TASK_VERSION,
    "unit_of_analysis": "first ICU stay per hospital admission",
    "population": f"age at admission >= {ADULT_MIN_AGE}",
    "index_time": "icu intime + 24 hours",
    "observation_window": "[icu intime, index_time]; charttime <= index_time and reliable storetime <= index_time",
    "eligibility": "alive at index_time, icu outtime > index_time, hospital dischtime > index_time "
                   "(checked before labeling)",
    "positive_label": "hospital_expire_flag = 1 and deathtime > index_time, and either "
                      "(deathtime <= dischtime and discharge_location in (DIED, not recorded)) or "
                      "(deathtime > dischtime and discharge_location = DIED, flagged as a timestamp anomaly)",
    "negative_label": "hospital_expire_flag = 0, no deathtime, dischtime > index_time, "
                      "discharge_location != DIED, no dod before the discharge date "
                      "(death after an alive discharge stays negative)",
    "outcome_anomalies": f"positives with deathtime later than dischtime are kept, flagged in {ANOMALY_COLUMN} "
                         "and counted with their gap distribution; no upper bound on the gap",
    "ambiguous_outcomes": "any other disagreement between expire flag, deathtime, dischtime, discharge_location "
                          "and dod is excluded and counted by reason",
    "outcome_horizon": "remainder of the index hospitalization (not a fixed-length window)",
}

AMBIGUITY_REASONS = (
    "missing_discharge_time",
    "missing_expire_flag",
    "expire_flag_without_deathtime",
    "deathtime_without_expire_flag",
    "deathtime_after_discharge_without_died_location",
    "expire_flag_with_alive_discharge_location",
    "discharge_location_died_but_flag_alive",
    "date_of_death_before_alive_discharge",
    "other",
)


def build_cohort(
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
) -> Tuple[pd.DataFrame, "OrderedDict[str, int]", Dict[str, int], Dict[str, object]]:
    """Applies the Task 1 eligibility and labeling rules.

    Returns (cohort, exclusion_counts, ambiguity_reason_counts, outcome_anomalies). The cohort has
    one row per eligible stay: subject_id, hadm_id, stay_id, window_start, index_time, age,
    sex_male, label and the anomaly flag. Exclusions are applied in order and each stay is counted
    under the first rule it fails.
    """
    counts: "OrderedDict[str, int]" = OrderedDict()
    counts["icu_stays_in_source"] = int(len(icustays))
    stays = icustays.copy()

    def exclude(mask: pd.Series, name: str) -> None:
        nonlocal stays
        mask = mask.fillna(True).astype(bool)
        counts[name] = int(mask.sum())
        stays = stays.loc[~mask]

    exclude(stays["intime"].isna(), "missing_icu_intime")

    adm = admissions.drop_duplicates("hadm_id", keep=False)
    stays = stays.merge(adm, on=["subject_id", "hadm_id"], how="left", indicator="_adm")
    exclude(stays["_adm"] != "both", "missing_admission_record")

    stays = stays.sort_values(["hadm_id", "intime", "stay_id"], kind="mergesort")
    exclude(stays.duplicated("hadm_id", keep="first"), "not_first_icu_stay_of_admission")
    exclude(stays["outtime"].isna() | (stays["outtime"] <= stays["intime"]), "invalid_icu_outtime")

    pts = patients.drop_duplicates("subject_id", keep=False)
    stays = stays.merge(pts, on="subject_id", how="left", indicator="_pt")
    exclude(stays["_pt"] != "both", "missing_patient_record")

    stays["age"] = stays["anchor_age"] + (stays["admittime"].dt.year - stays["anchor_year"])
    exclude(stays["age"].isna(), "age_unavailable")
    exclude(stays["age"] < ADULT_MIN_AGE, "under_18")

    stays["index_time"] = stays["intime"] + INDEX_OFFSET
    died_before = (stays["deathtime"].notna() & (stays["deathtime"] <= stays["index_time"])) | (
        stays["dod"].notna() & (stays["dod"].dt.normalize() < stays["index_time"].dt.normalize())
    )
    exclude(died_before, "died_before_index")
    exclude(stays["outtime"] <= stays["index_time"], "left_icu_before_index")
    exclude(stays["dischtime"].notna() & (stays["dischtime"] <= stays["index_time"]), "hospital_discharge_before_index")

    label, reasons, anomaly = assign_labels(stays)
    ambiguous = label.isna()
    ambiguity_counts = {reason: int((reasons[ambiguous] == reason).sum()) for reason in AMBIGUITY_REASONS}
    exclude(ambiguous, "ambiguous_outcome")
    stays = stays.assign(
        label=label.loc[stays.index].astype("int8"),
        anomaly=anomaly.loc[stays.index].astype("int8"),
    )

    sex = stays["gender"].astype("string")
    cohort = pd.DataFrame({
        "subject_id": stays["subject_id"].astype("int64"),
        "hadm_id": stays["hadm_id"].astype("int64"),
        "stay_id": stays["stay_id"].astype("int64"),
        "window_start": stays["intime"],
        "index_time": stays["index_time"],
        "age": stays["age"].astype("float64"),
        "sex_male": np.where(sex == "M", 1.0, np.where(sex == "F", 0.0, np.nan)),
        "label": stays["label"],
        ANOMALY_COLUMN: stays["anomaly"],
    }).sort_values("stay_id", kind="mergesort").reset_index(drop=True)

    counts["cohort_size"] = int(len(cohort))
    counts["positive_labels"] = int(cohort["label"].sum())
    counts["negative_labels"] = int(len(cohort) - cohort["label"].sum())

    gaps_hours = ((stays["deathtime"] - stays["dischtime"]).dt.total_seconds() / 3600.0)[stays["anomaly"] == 1]
    outcome_anomalies = {
        "positive_deathtime_after_discharge": int(stays["anomaly"].sum()),
        "deathtime_after_discharge_gap": summarize_gaps(gaps_hours),
    }
    return cohort, counts, ambiguity_counts, outcome_anomalies


def summarize_gaps(gaps_hours: pd.Series) -> Dict[str, int]:
    """Counts anomaly gaps (hours, all > 0) per reporting bucket."""
    summary, lower = {}, 0.0
    for upper, name in ANOMALY_GAP_BUCKETS:
        summary[name] = int(((gaps_hours > lower) & (gaps_hours <= upper)).sum())
        lower = upper
    summary[ANOMALY_GAP_OVERFLOW] = int((gaps_hours > lower).sum())
    return summary


def assign_labels(stays: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (label, ambiguity_reason, anomaly) per row.

    label is 1/0 or <NA> when ambiguous; anomaly is True for positives whose deathtime is later
    than dischtime (accepted because the flag and the DIED disposition both record the death).
    """
    flag = stays["hospital_expire_flag"]
    flag_dead = (flag == 1).fillna(False).astype(bool)
    flag_alive = (flag == 0).fillna(False).astype(bool)
    has_death = stays["deathtime"].notna()
    has_disch = stays["dischtime"].notna()
    location = stays["discharge_location"].fillna("").astype(str).str.strip().str.upper()
    location_died = location.eq(DIED_LOCATION)
    location_alive = location.ne("") & ~location_died
    dod_before_discharge = (
        stays["dod"].notna() & has_disch & (stays["dod"].dt.normalize() < stays["dischtime"].dt.normalize())
    )
    death_after_index = has_death & (stays["deathtime"] > stays["index_time"])
    death_after_discharge = has_death & has_disch & (stays["deathtime"] > stays["dischtime"])

    consistent_positive = flag_dead & death_after_index & has_disch & ~death_after_discharge & ~location_alive
    anomaly_positive = flag_dead & death_after_index & death_after_discharge & location_died
    negative = (
        flag_alive & ~has_death & has_disch & (stays["dischtime"] > stays["index_time"])
        & ~location_died & ~dod_before_discharge
    )

    label = pd.Series(pd.NA, index=stays.index, dtype="Int8")
    label[consistent_positive | anomaly_positive] = 1
    label[negative] = 0

    reason = pd.Series(
        np.select(
            [
                ~has_disch,
                flag.isna(),
                flag_dead & ~has_death,
                flag_alive & has_death,
                flag_dead & death_after_discharge & ~location_died,
                flag_dead & location_alive,
                flag_alive & location_died,
                flag_alive & dod_before_discharge,
            ],
            list(AMBIGUITY_REASONS[:-1]),
            default="other",
        ),
        index=stays.index,
    )
    return label, reason, anomaly_positive
