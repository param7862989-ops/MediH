"""Leakage-safe observation cleaning and window aggregation for MIMIC-IV Task 1 (feature spec v1).

Rules, applied to long-format measurements in this order:
1. Window: keep rows with window_start <= charttime <= index_time.
2. Availability: when storetime is reliable (present and not earlier than charttime) it must also
   be <= index_time; an unreliable or missing storetime falls back to the charttime rule alone.
3. Non-numeric values (no valuenum) are dropped.
4. Units: the recorded valueuom must be one of the item's accepted units; a missing valueuom is
   accepted because the item ID defines the unit (counted separately).
5. Conversion to canonical units (Fahrenheit -> Celsius).
6. Plausibility: canonical values outside the variable's inclusive range are dropped.
7. Blood pressure source, per stay: if any invasive (arterial line) systolic or diastolic reading
   survives, all BP variables use invasive readings only; otherwise non-invasive readings only.
8. MAP: measured MAP from the selected source; if the stay has none, MAP is derived from paired
   systolic/diastolic readings at the same charttime (DBP + (SBP - DBP) / 3, requiring SBP > DBP).

Missing measurements stay missing (NaN) with an explicit ``<variable>_missing`` flag; nothing
is filled with invented normal values.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.datasources.mimic.mappings import (
    AGGREGATIONS,
    BP_VARIABLES,
    VARIABLES,
    VARIABLES_BY_NAME,
    convert_to_canonical,
    feature_columns,
    normalize_unit,
)

DROP_REASONS = (
    "outside_window",
    "stored_after_index",
    "non_numeric",
    "unit_mismatch",
    "implausible",
)


def clean_observations(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    validated_items: Dict[int, tuple],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Applies rules 1-6. Returns (observations, per-variable aggregate diagnostics)."""
    obs_columns = ["stay_id", "variable", "charttime", "value", "itemid", "bp_source"]
    diagnostics = {var.name: {"rows_in": 0, **{r: 0 for r in DROP_REASONS}, "unit_missing_accepted": 0,
                              "storetime_unreliable_used_charttime": 0, "kept": 0} for var in VARIABLES}

    item_meta = pd.DataFrame(
        [
            {
                "itemid": itemid, "variable": var.name, "item_source_table": item.source_table,
                "source_unit": item.source_unit, "bp_source": item.bp_source,
                "plausible_min": var.plausible_min, "plausible_max": var.plausible_max,
            }
            for itemid, (var, item) in validated_items.items()
        ],
        columns=["itemid", "variable", "item_source_table", "source_unit", "bp_source",
                 "plausible_min", "plausible_max"],
    )
    if events.empty or item_meta.empty:
        return pd.DataFrame(columns=obs_columns), diagnostics

    df = events.merge(item_meta, on="itemid", how="inner")
    df = df[df["source_table"].astype(str) == df["item_source_table"]]
    df = df.merge(cohort[["stay_id", "window_start", "index_time"]], on="stay_id", how="inner")

    for name, group_size in df.groupby("variable").size().items():
        diagnostics[name]["rows_in"] = int(group_size)

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal df
        mask = mask.fillna(True).astype(bool)
        for name, n in df.loc[mask].groupby("variable").size().items():
            diagnostics[name][reason] += int(n)
        df = df.loc[~mask]

    drop(df["charttime"].isna() | (df["charttime"] < df["window_start"]) | (df["charttime"] > df["index_time"]),
         "outside_window")

    reliable_store = df["storetime"].notna() & (df["storetime"] >= df["charttime"])
    for name, n in df.loc[~reliable_store].groupby("variable").size().items():
        diagnostics[name]["storetime_unreliable_used_charttime"] += int(n)
    drop(reliable_store & (df["storetime"] > df["index_time"]), "stored_after_index")

    drop(df["valuenum"].isna(), "non_numeric")

    units = df["valueuom"].map(normalize_unit)
    unit_ok = pd.Series(False, index=df.index)
    for itemid, (var, item) in validated_items.items():
        rows = df["itemid"] == itemid
        unit_ok |= rows & (units.isna() | units.isin(item.accepted_units))
    for name, n in df.loc[units.isna()].groupby("variable").size().items():
        diagnostics[name]["unit_missing_accepted"] += int(n)
    drop(~unit_ok, "unit_mismatch")

    df = df.assign(value=df["valuenum"].astype(float))
    for source_unit in df["source_unit"].unique():
        rows = df["source_unit"] == source_unit
        df.loc[rows, "value"] = convert_to_canonical(df.loc[rows, "value"], source_unit)

    drop((df["value"] < df["plausible_min"]) | (df["value"] > df["plausible_max"]), "implausible")

    for name, n in df.groupby("variable").size().items():
        diagnostics[name]["kept"] = int(n)

    obs = df[obs_columns].reset_index(drop=True)
    obs["stay_id"] = obs["stay_id"].astype("int64")
    return obs, diagnostics


def select_blood_pressure_source(obs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Applies rule 7. Returns (observations, bp_source_invasive per stay: 1.0 / 0.0)."""
    is_bp = obs["variable"].isin(BP_VARIABLES)
    sys_dia = obs["variable"].isin(("sbp", "dbp"))
    invasive_stays = set(obs.loc[sys_dia & (obs["bp_source"] == "invasive"), "stay_id"])
    stays_with_bp = set(obs.loc[sys_dia, "stay_id"])

    chosen = np.where(obs["stay_id"].isin(invasive_stays), "invasive", "noninvasive")
    keep = ~is_bp | (obs["bp_source"] == chosen).fillna(False).astype(bool)
    flag = pd.Series({stay: (1.0 if stay in invasive_stays else 0.0) for stay in stays_with_bp}, dtype="float64")
    return obs.loc[keep].reset_index(drop=True), flag


def derive_map(obs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Applies rule 8. Returns (observations incl. derived MAP rows, map_derived per stay)."""
    measured_stays = set(obs.loc[obs["variable"] == "map", "stay_id"])
    flag = {stay: 0.0 for stay in measured_stays}

    pairs = obs[obs["variable"].isin(("sbp", "dbp")) & ~obs["stay_id"].isin(measured_stays)]
    if pairs.empty:
        return obs, pd.Series(flag, dtype="float64")

    wide = (
        pairs.groupby(["stay_id", "charttime", "bp_source", "variable"])["value"].mean()
        .unstack("variable")
        .reindex(columns=["sbp", "dbp"])
        .dropna()
    )
    wide = wide[wide["sbp"] > wide["dbp"]]
    spec = VARIABLES_BY_NAME["map"]
    wide = wide.assign(value=wide["dbp"] + (wide["sbp"] - wide["dbp"]) / 3.0)
    wide = wide[(wide["value"] >= spec.plausible_min) & (wide["value"] <= spec.plausible_max)].reset_index()
    if wide.empty:
        return obs, pd.Series(flag, dtype="float64")

    derived = pd.DataFrame({
        "stay_id": wide["stay_id"],
        "variable": "map",
        "charttime": wide["charttime"],
        "value": wide["value"],
        "itemid": pd.Series(pd.NA, index=wide.index, dtype="Int64"),
        "bp_source": wide["bp_source"],
    })
    flag.update({stay: 1.0 for stay in derived["stay_id"].unique()})
    return pd.concat([obs, derived], ignore_index=True), pd.Series(flag, dtype="float64")


def aggregate_features(
    obs: pd.DataFrame,
    cohort: pd.DataFrame,
    bp_invasive: pd.Series,
    map_derived: pd.Series,
) -> pd.DataFrame:
    """Summarizes each variable per stay over the window; one row per cohort stay, ordered by stay_id."""
    stay_ids = pd.Index(cohort["stay_id"].astype("int64"), name="stay_id")
    features = pd.DataFrame(index=stay_ids)

    # Deterministic ordering so "last" is reproducible when timestamps tie
    ordered = obs.assign(_item=obs["itemid"].astype("Int64").fillna(-1)).sort_values(
        ["stay_id", "variable", "charttime", "_item", "value"], kind="mergesort"
    )
    grouped = ordered.groupby(["variable", "stay_id"])["value"]
    summary = pd.DataFrame({
        "min": grouped.min(),
        "max": grouped.max(),
        "mean": grouped.mean(),
        "last": grouped.last(),
        "count": grouped.count(),
    })

    for var in VARIABLES:
        per_var = summary.xs(var.name, level="variable") if var.name in summary.index.get_level_values(0) else None
        for agg in AGGREGATIONS:
            col = f"{var.name}_{agg}"
            if per_var is None:
                features[col] = 0 if agg == "count" else np.nan
            else:
                series = per_var[agg].reindex(stay_ids)
                features[col] = series.fillna(0).astype("int64") if agg == "count" else series.astype("float64")
        features[f"{var.name}_missing"] = (features[f"{var.name}_count"] == 0).astype("int8")

    features["map_derived"] = map_derived.reindex(stay_ids).astype("float64")
    features["bp_source_invasive"] = bp_invasive.reindex(stay_ids).astype("float64")
    demographics = cohort.set_index("stay_id")[["age", "sex_male"]]
    features["age"] = demographics["age"].reindex(stay_ids).astype("float64")
    features["sex_male"] = demographics["sex_male"].reindex(stay_ids).astype("float64")
    return features[feature_columns()].reset_index()


def extract_features(events: pd.DataFrame, cohort: pd.DataFrame, validated_items: Dict[int, tuple]):
    """Runs rules 1-8 and aggregation. Returns (features, diagnostics)."""
    obs, diagnostics = clean_observations(events, cohort, validated_items)
    obs, bp_invasive = select_blood_pressure_source(obs)
    obs, map_derived = derive_map(obs)
    return aggregate_features(obs, cohort, bp_invasive, map_derived), diagnostics
