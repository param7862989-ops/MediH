"""Task 1 tests: cohort rules, labels, feature quality, leakage, reproducibility and output safety.

All data are hand-crafted fixtures (tests/mimic_fixtures.py); no real MIMIC-IV records.
"""

import json
import math
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.datasources.mimic.mappings import feature_columns, validate_item_dictionaries
from src.datasources.mimic.reader import MimicIVReader
from src.tasks.build_mimic_t1 import LABEL_COLUMN, build_dataset, dataset_dir, main, research_log_path
from src.tasks.features import clean_observations, derive_map, select_blood_pressure_source
from src.tasks.mortality_icu_24h import ANOMALY_COLUMN, build_cohort
from src.utils.config import Config
from src.utils.research_paths import UnsafeResearchPathError, unsafe_reasons
from tests.isolation_utils import PROJECT_ROOT
from tests.mimic_fixtures import (
    ALL_FIXTURE_IDS,
    INDEX,
    T0,
    H,
    S10_SECOND_STAY,
    build_tables,
    sid,
    stid,
    write_tables,
)

DEMO_FOLDER = "mimic-iv-clinical-database-demo-2.2"


def _cohort_full(tables):
    """Runs the cohort rules on fixture tables typed by the real reader (via a CSV round trip).

    Returns (cohort, exclusion_counts, ambiguity_reasons, outcome_anomalies).
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = write_tables(Path(tmp) / DEMO_FOLDER, tables)
        with MimicIVReader(root) as reader:
            typed = reader.load_patients(), reader.load_admissions(), reader.load_icustays()
    return build_cohort(*typed)


def _cohort_from(tables):
    """(cohort, exclusion_counts, ambiguity_reasons)"""
    return _cohort_full(tables)[:3]


def _build(tmp_path, tables=None, compress=True, out_name="out", overwrite=False):
    root = tmp_path / DEMO_FOLDER
    if not root.exists() or tables is not None:
        write_tables(root, tables or build_tables(), compress=compress)
    out = tmp_path / out_name
    manifest = build_dataset(root, out, overwrite=overwrite)
    return pd.read_parquet(dataset_dir(out) / "dataset.parquet"), manifest


def _row(df, stay_id):
    rows = df[df["src_stay_id"] == stay_id]
    assert len(rows) == 1
    return rows.iloc[0]


# ------------------------------------------------------------------------------
# Cohort and labels
# ------------------------------------------------------------------------------
def test_cohort_flow_and_labels():
    cohort, counts, reasons = _cohort_from(build_tables())

    assert dict(counts) == {
        "icu_stays_in_source": 12,
        "missing_icu_intime": 0,
        "missing_admission_record": 1,       # S11
        "not_first_icu_stay_of_admission": 1,  # S10 second stay
        "invalid_icu_outtime": 0,
        "missing_patient_record": 0,
        "age_unavailable": 0,
        "under_18": 1,                       # S6
        "died_before_index": 1,              # S3
        "left_icu_before_index": 1,          # S4 (outtime == index time)
        "hospital_discharge_before_index": 1,  # S9
        "ambiguous_outcome": 2,              # S7, S8
        "cohort_size": 4,
        "positive_labels": 1,
        "negative_labels": 3,
    }
    assert reasons["expire_flag_without_deathtime"] == 1
    assert reasons["deathtime_without_expire_flag"] == 1

    labels = dict(zip(cohort["stay_id"], cohort["label"]))
    assert labels == {stid(1): 0, stid(2): 1, stid(5): 0, stid(10): 0}
    assert S10_SECOND_STAY not in labels
    assert (cohort["index_time"] == cohort["window_start"] + pd.Timedelta(hours=24)).all()


def test_icu_boundary_at_exactly_24_hours():
    """Leaving the ICU exactly at the index time is ineligible; one minute later is eligible."""
    cohort, _, _ = _cohort_from(build_tables())
    assert stid(4) not in set(cohort["stay_id"])
    assert stid(5) in set(cohort["stay_id"])


def test_death_before_index_excluded_and_after_index_positive():
    tables = build_tables()
    adm = tables["hosp/admissions"]
    # S1 now dies 1 second after the index time -> positive
    adm.loc[adm["subject_id"] == sid(1), ["hospital_expire_flag", "deathtime", "dischtime", "discharge_location"]] = [
        1, INDEX + pd.Timedelta(seconds=1), INDEX + pd.Timedelta(seconds=1), "DIED"]
    # S5 dies exactly at the index time -> died_before_index
    adm.loc[adm["subject_id"] == sid(5), ["hospital_expire_flag", "deathtime", "dischtime", "discharge_location"]] = [
        1, INDEX, INDEX, "DIED"]
    cohort, counts, _ = _cohort_from(tables)
    labels = dict(zip(cohort["stay_id"], cohort["label"]))
    assert labels[stid(1)] == 1
    assert stid(5) not in labels
    assert counts["died_before_index"] == 2


def test_alive_discharge_is_negative_only_without_conflicts():
    tables = build_tables()
    adm = tables["hosp/admissions"]
    pts = tables["hosp/patients"]
    adm.loc[adm["subject_id"] == sid(1), "discharge_location"] = "DIED"   # conflicts with flag 0
    pts.loc[pts["subject_id"] == sid(5), "dod"] = "2150-03-02"            # dod before discharge date
    adm.loc[adm["subject_id"] == sid(2), "deathtime"] = T0 + pd.Timedelta(days=9)  # flag 1 + DIED: anomaly positive
    cohort, counts, reasons, anomalies = _cohort_full(tables)
    assert set(cohort["stay_id"]) == {stid(2), stid(10)}
    assert dict(zip(cohort["stay_id"], cohort["label"])) == {stid(2): 1, stid(10): 0}
    assert reasons["discharge_location_died_but_flag_alive"] == 1
    assert reasons["date_of_death_before_alive_discharge"] == 1
    assert anomalies["positive_deathtime_after_discharge"] == 1


# ------------------------------------------------------------------------------
# Outcome-rule boundaries (task version 3)
# ------------------------------------------------------------------------------
DISCH_S2 = T0 + pd.Timedelta(days=4)  # S2 is discharged (dead) at this time in the fixture


def _s2_outcome(**fields):
    """Rebuilds the cohort with S2's outcome fields overridden.

    Returns (label or None, anomaly flag or None, non-zero ambiguity reasons, outcome anomalies, exclusion counts).
    """
    tables = build_tables()
    adm, pts = tables["hosp/admissions"], tables["hosp/patients"]
    for column, value in fields.items():
        frame = pts if column == "dod" else adm
        frame.loc[frame["subject_id"] == sid(2), column] = value
    cohort, counts, reasons, anomalies = _cohort_full(tables)
    row = cohort[cohort["stay_id"] == stid(2)]
    label = None if row.empty else int(row["label"].iloc[0])
    anomaly = None if row.empty else int(row[ANOMALY_COLUMN].iloc[0])
    return label, anomaly, {k: v for k, v in reasons.items() if v}, anomalies, counts


def test_death_exactly_at_discharge_is_positive_without_anomaly():
    label, anomaly, _, anomalies, _ = _s2_outcome(deathtime=DISCH_S2, dischtime=DISCH_S2)
    assert (label, anomaly) == (1, 0)
    assert anomalies["positive_deathtime_after_discharge"] == 0


@pytest.mark.parametrize("after, bucket", [
    (pd.Timedelta(minutes=1), "(0, 1] h"),
    (pd.Timedelta(hours=1), "(0, 1] h"),
    (pd.Timedelta(hours=6), "(1, 6] h"),
    (pd.Timedelta(hours=23), "(6, 24] h"),
    (pd.Timedelta(days=3), "> 24 h"),  # no upper bound; the gap is reported instead
])
def test_flag_and_died_with_deathtime_after_discharge_is_flagged_positive(after, bucket):
    label, anomaly, reasons, anomalies, _ = _s2_outcome(
        deathtime=DISCH_S2 + after, dischtime=DISCH_S2, hospital_expire_flag=1, discharge_location="DIED")
    assert (label, anomaly) == (1, 1)
    assert anomalies["positive_deathtime_after_discharge"] == 1
    assert anomalies["deathtime_after_discharge_gap"][bucket] == 1
    assert sum(anomalies["deathtime_after_discharge_gap"].values()) == 1
    assert "deathtime_after_discharge_without_died_location" not in reasons


@pytest.mark.parametrize("location", [None, "HOME", "HOSPICE"])
def test_deathtime_after_discharge_without_died_location_is_excluded(location):
    label, _, reasons, anomalies, _ = _s2_outcome(
        deathtime=DISCH_S2 + pd.Timedelta(hours=2), hospital_expire_flag=1, discharge_location=location)
    assert label is None
    assert reasons.get("deathtime_after_discharge_without_died_location") == 1
    assert anomalies["positive_deathtime_after_discharge"] == 0


def test_anomaly_rule_never_rescues_discharge_before_index():
    # Recorded discharge before the index time: ineligible even with flag 1 + DIED and a later deathtime
    label, _, _, anomalies, counts = _s2_outcome(
        dischtime=INDEX - pd.Timedelta(hours=1), deathtime=INDEX + pd.Timedelta(hours=1),
        hospital_expire_flag=1, discharge_location="DIED", dod="2150-03-02")
    assert label is None
    assert counts["hospital_discharge_before_index"] == 2  # fixture S9 plus the modified S2
    assert anomalies["positive_deathtime_after_discharge"] == 0


def test_death_after_documented_alive_discharge_is_not_in_hospital_death():
    # Alive discharge (flag 0, HOME) followed hours later by a recorded death time: conflicting -> excluded
    label, _, reasons, anomalies, _ = _s2_outcome(hospital_expire_flag=0, discharge_location="HOME",
                                                  deathtime=DISCH_S2 + pd.Timedelta(hours=2), dod=None)
    assert label is None
    assert reasons.get("deathtime_without_expire_flag") == 2  # fixture S8 plus the modified S2
    assert anomalies["positive_deathtime_after_discharge"] == 0


@pytest.mark.parametrize("dod", ["2150-03-05", "2150-03-06"])
def test_out_of_hospital_death_on_or_after_alive_discharge_date_stays_negative(dod):
    label, anomaly, _, _, _ = _s2_outcome(hospital_expire_flag=0, discharge_location="HOME", deathtime=None, dod=dod)
    assert (label, anomaly) == (0, 0)  # discharged alive on 2150-03-05; a later date of death is not in-hospital


def test_expire_flag_with_alive_discharge_location_is_ambiguous():
    label, _, reasons, _, _ = _s2_outcome(discharge_location="HOME")
    assert label is None
    assert reasons.get("expire_flag_with_alive_discharge_location") == 1


def test_expire_flag_with_unrecorded_discharge_location_is_positive():
    label, anomaly, _, _, _ = _s2_outcome(discharge_location=None)
    assert (label, anomaly) == (1, 0)


def test_missing_discharge_time_is_ambiguous():
    tables = build_tables()
    adm = tables["hosp/admissions"]
    adm.loc[adm["subject_id"] == sid(1), "dischtime"] = None
    cohort, _, reasons = _cohort_from(tables)
    assert stid(1) not in set(cohort["stay_id"])
    assert reasons["missing_discharge_time"] == 1


# ------------------------------------------------------------------------------
# Feature quality rules
# ------------------------------------------------------------------------------
def test_feature_values_units_and_quality_rules(tmp_path):
    df, manifest = _build(tmp_path)
    assert list(df.columns) == (
        ["src_subject_id", "src_hadm_id", "src_stay_id", "index_time", LABEL_COLUMN, ANOMALY_COLUMN]
        + feature_columns() + ["dataset_version", "feature_spec_version"]
    )
    assert ANOMALY_COLUMN not in manifest["columns"]["features"]
    assert len(df) == 4

    s1 = _row(df, stid(1))
    # Heart rate: 0 bpm (implausible), mmHg unit, stored-after-index, post-index and pre-ICU rows removed
    assert (s1["heart_rate_min"], s1["heart_rate_max"], s1["heart_rate_last"], s1["heart_rate_count"]) == (80, 100, 100, 3)
    assert s1["heart_rate_mean"] == pytest.approx(90)
    # Temperature: 98.6 F -> 37.0 C kept, 9 F dropped as implausible
    assert s1["temperature_min"] == pytest.approx(37.0)
    assert s1["temperature_max"] == pytest.approx(38.0)
    assert s1["temperature_count"] == 2
    # SpO2 row without a unit accepted; last value by charttime
    assert (s1["spo2_count"], s1["spo2_last"]) == (2, 95)
    # Non-invasive BP with measured MAP
    assert (s1["sbp_last"], s1["dbp_last"], s1["map_last"]) == (120, 80, 93)
    assert (s1["bp_source_invasive"], s1["map_derived"]) == (0.0, 0.0)
    # Labs: creatinine inside window only; lactate stored after index -> missing; glucose without storetime kept
    assert (s1["creatinine_count"], s1["creatinine_last"]) == (1, 1.1)
    assert s1["lactate_count"] == 0 and s1["lactate_missing"] == 1 and math.isnan(s1["lactate_last"])
    assert s1["glucose_last"] == 140
    assert (s1["age"], s1["sex_male"]) == (60, 1.0)

    s2 = _row(df, stid(2))
    # Invasive line present: NIBP systolic 130 and NIBP MAP ignored; MAP derived from 100/60
    assert (s2["sbp_last"], s2["dbp_last"], s2["bp_source_invasive"], s2["map_derived"]) == (100, 60, 1.0, 1.0)
    assert s2["map_last"] == pytest.approx(60 + 40 / 3)
    assert s2[LABEL_COLUMN] == 1

    quality = manifest["aggregate_counts"]["observation_quality"]["heart_rate"]
    assert quality["implausible"] == 1
    assert quality["unit_mismatch"] == 1
    assert quality["stored_after_index"] == 1
    assert quality["kept"] == 5


def test_missing_measurements_stay_missing(tmp_path):
    df, _ = _build(tmp_path)
    s10 = _row(df, stid(10))  # no measurements at all
    for var in ("heart_rate", "sbp", "creatinine", "lactate"):
        assert s10[f"{var}_count"] == 0
        assert s10[f"{var}_missing"] == 1
        for agg in ("min", "max", "mean", "last"):
            assert math.isnan(s10[f"{var}_{agg}"]), f"{var}_{agg} was imputed"
    assert math.isnan(s10["bp_source_invasive"]) and math.isnan(s10["map_derived"])


def test_blood_pressure_rule_and_map_derivation_units():
    obs = pd.DataFrame({
        "stay_id": [1, 1, 1, 2, 2, 2],
        "variable": ["sbp", "dbp", "sbp", "sbp", "dbp", "map"],
        "charttime": [T0] * 6,
        "value": [110.0, 70.0, 140.0, 120.0, 80.0, 95.0],
        "itemid": [220050, 220051, 220179, 220179, 220180, 220181],
        "bp_source": ["invasive", "invasive", "noninvasive", "noninvasive", "noninvasive", "noninvasive"],
    })
    selected, invasive = select_blood_pressure_source(obs)
    assert sorted(selected[selected["stay_id"] == 1]["value"]) == [70.0, 110.0]
    assert invasive.to_dict() == {1: 1.0, 2: 0.0}
    with_map, derived = derive_map(selected)
    stay1_map = with_map[(with_map["stay_id"] == 1) & (with_map["variable"] == "map")]["value"]
    assert stay1_map.round(4).tolist() == [round(70 + 40 / 3, 4)]
    assert derived.to_dict() == {1: 1.0, 2: 0.0}


# ------------------------------------------------------------------------------
# Leakage
# ------------------------------------------------------------------------------
def test_post_index_rows_never_reach_observations():
    tables = build_tables()
    validated, _ = validate_item_dictionaries(tables["icu/d_items"], tables["hosp/d_labitems"])
    cohort = pd.DataFrame({"stay_id": [1], "window_start": [T0], "index_time": [INDEX]})
    events = pd.DataFrame({
        "stay_id": [1, 1, 1],
        "source_table": ["chartevents"] * 3,
        "itemid": [220045] * 3,
        "charttime": [T0 + H, INDEX + pd.Timedelta(seconds=1), INDEX - H],
        "storetime": [T0 + 2 * H, INDEX + 2 * H, INDEX + pd.Timedelta(seconds=1)],
        "valuenum": [80.0, 120.0, 130.0],
        "valueuom": ["bpm"] * 3,
    })
    obs, diag = clean_observations(events, cohort, validated)
    assert obs["value"].tolist() == [80.0]
    assert diag["heart_rate"]["outside_window"] == 1
    assert diag["heart_rate"]["stored_after_index"] == 1


def test_changing_post_index_data_cannot_change_features(tmp_path):
    baseline, base_manifest = _build(tmp_path, out_name="baseline")

    tables = build_tables()
    chart, labs = tables["icu/chartevents"], tables["hosp/labevents"]
    chart.loc[chart["charttime"] > INDEX, "valuenum"] += 500
    labs.loc[labs["charttime"] > INDEX, "valuenum"] += 500
    extra_chart = pd.DataFrame([
        {**chart.iloc[0].to_dict(), "stay_id": stid(n), "subject_id": sid(n), "charttime": INDEX + pd.Timedelta(minutes=1),
         "storetime": INDEX + pd.Timedelta(minutes=2), "valuenum": 250.0}
        for n in (1, 2, 5, 10)
    ] + [
        {**chart.iloc[0].to_dict(), "stay_id": stid(5), "subject_id": sid(5), "charttime": INDEX - H,
         "storetime": INDEX + pd.Timedelta(minutes=10), "valuenum": 180.0},
    ])
    extra_lab = pd.DataFrame([
        {**labs.iloc[0].to_dict(), "subject_id": sid(10), "charttime": INDEX - H,
         "storetime": INDEX + H, "valuenum": 9.9},
        {**labs.iloc[0].to_dict(), "subject_id": sid(10), "charttime": INDEX + H,
         "storetime": INDEX + 2 * H, "valuenum": 9.9},
    ])
    tables["icu/chartevents"] = pd.concat([chart, extra_chart], ignore_index=True)
    tables["hosp/labevents"] = pd.concat([labs, extra_lab], ignore_index=True)

    perturbed, perturbed_manifest = _build(tmp_path, tables=tables, out_name="perturbed")
    pd.testing.assert_frame_equal(baseline, perturbed)
    assert base_manifest["content_sha256"] == perturbed_manifest["content_sha256"]


# ------------------------------------------------------------------------------
# Reproducibility and output
# ------------------------------------------------------------------------------
def test_build_is_deterministic_and_format_independent(tmp_path):
    first, m1 = _build(tmp_path / "a", out_name="out1")
    second, m2 = _build(tmp_path / "a", out_name="out2")
    plain, m3 = _build(tmp_path / "b", compress=False, out_name="out")
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first, plain)
    assert m1["content_sha256"] == m2["content_sha256"] == m3["content_sha256"]
    assert m1["feature_spec_version"] == "mimic-iv-t1-features-v1"
    assert m1["source"]["release"] == "mimic-iv-demo-2.2"


def test_manifest_records_configuration_and_versions(tmp_path):
    _, manifest = _build(tmp_path)
    written = json.loads((dataset_dir(tmp_path / "out") / "manifest.json").read_text(encoding="utf-8"))
    assert written["content_sha256"] == manifest["content_sha256"]
    assert written["dataset_version"] == "1.2.0"
    assert written["task"]["task_version"] == "3"
    assert "timestamp anomaly" in written["task"]["positive_label"]
    assert written["aggregate_counts"]["outcome_anomalies"]["positive_deathtime_after_discharge"] == 0
    assert written["columns"]["label_quality"]["columns"] == [ANOMALY_COLUMN]
    assert written["aggregate_counts"]["cohort_size"] == 4
    assert written["aggregate_counts"]["outcome_prevalence"] == 0.25
    assert written["task"]["index_time"] == "icu intime + 24 hours"
    assert "220045" in written["configuration"]["item_mapping"]
    assert not list(dataset_dir(tmp_path / "out").glob("*.tmp"))


def test_outcome_anomaly_reaches_dataset_manifest_log_and_summary(tmp_path, capsys):
    tables = build_tables()
    adm = tables["hosp/admissions"]
    adm.loc[adm["subject_id"] == sid(2), "deathtime"] = DISCH_S2 + pd.Timedelta(hours=2)  # flag 1 + DIED
    root = write_tables(tmp_path / DEMO_FOLDER, tables)
    assert main(["--mimic-dir", str(root), "--output-dir", str(tmp_path / "out")]) == 0
    printed = capsys.readouterr().out

    df = pd.read_parquet(dataset_dir(tmp_path / "out") / "dataset.parquet")
    s2 = _row(df, stid(2))
    assert (s2[LABEL_COLUMN], s2[ANOMALY_COLUMN]) == (1, 1)
    assert int(df[ANOMALY_COLUMN].sum()) == 1

    manifest = json.loads((dataset_dir(tmp_path / "out") / "manifest.json").read_text(encoding="utf-8"))
    anomalies = manifest["aggregate_counts"]["outcome_anomalies"]
    assert anomalies["positive_deathtime_after_discharge"] == 1
    assert anomalies["deathtime_after_discharge_gap"] == {"(0, 1] h": 0, "(1, 6] h": 1, "(6, 24] h": 0, "> 24 h": 0}
    assert manifest["aggregate_counts"]["positive_labels"] == 1
    assert manifest["aggregate_counts"]["exclusions"]["ambiguous_outcome"] == 2  # unchanged: S7, S8

    assert "Outcome anomalies (labeled positive, flagged): deathtime after dischtime = 1 (gap (1, 6] h: 1)" in printed
    log_text = research_log_path(tmp_path / "out").read_text(encoding="utf-8")
    assert '"positive_deathtime_after_discharge": 1' in log_text


def test_cli_prints_only_aggregates(tmp_path, capsys):
    root = write_tables(tmp_path / DEMO_FOLDER, build_tables())
    assert main(["--mimic-dir", str(root), "--output-dir", str(tmp_path / "out")]) == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "Eligible stays: 4" in output and "prevalence: 25.0%" in output
    research_log = research_log_path(tmp_path / "out").read_text(encoding="utf-8")
    app_log = Config.LOG_FILE.read_text(encoding="utf-8") if Config.LOG_FILE.exists() else ""
    for identifier in ALL_FIXTURE_IDS:
        assert str(identifier) not in output, "the builder printed a source identifier"
        assert str(identifier) not in research_log, "the builder logged a source identifier"
        assert str(identifier) not in app_log, "the app log received a source identifier"
    assert "2150-03" not in output and "2150-03" not in research_log, "a patient timestamp leaked"


def test_research_logs_and_spill_files_stay_in_output_folder(tmp_path):
    """Research logs/diagnostics are written under --output-dir, never to the repository's logs/."""
    app_log_before = Config.LOG_FILE.read_text(encoding="utf-8") if Config.LOG_FILE.exists() else ""
    _, manifest = _build(tmp_path)
    out = (tmp_path / "out").resolve()

    paths = {name: Path(p) for name, p in manifest["configuration"]["runtime_paths"].items()}
    assert set(paths) == {"dataset_dir", "research_log", "duckdb_spill_dir"}
    for path in paths.values():
        assert out in path.parents
        assert unsafe_reasons(path) == []

    log_text = paths["research_log"].read_text(encoding="utf-8")
    assert "Build started" in log_text and "Cohort flow" in log_text and "Build finished" in log_text
    assert not paths["duckdb_spill_dir"].exists(), "DuckDB spill folder was not cleaned up"
    app_log_after = Config.LOG_FILE.read_text(encoding="utf-8") if Config.LOG_FILE.exists() else ""
    assert "build_mimic_t1" not in app_log_after[len(app_log_before):]
    assert not (PROJECT_ROOT / ".tmp").exists(), "DuckDB spilled into the repository"


def test_reader_spill_directory_is_explicit_and_safe(tmp_path):
    root = write_tables(tmp_path / DEMO_FOLDER, build_tables())
    with MimicIVReader(root) as reader:
        default_spill = reader.temp_directory
        reported = reader._con.execute("SELECT current_setting('temp_directory')").fetchone()[0]
        assert Path(reported).resolve() == default_spill
        assert unsafe_reasons(default_spill) == []
    assert not default_spill.exists(), "the reader's private spill folder was not removed"

    with pytest.raises(UnsafeResearchPathError):
        MimicIVReader(root, temp_directory=PROJECT_ROOT / ".tmp")
    assert not (PROJECT_ROOT / ".tmp").exists()


def test_cli_refuses_to_overwrite_without_flag(tmp_path, capsys):
    root = write_tables(tmp_path / DEMO_FOLDER, build_tables())
    args = ["--mimic-dir", str(root), "--output-dir", str(tmp_path / "out")]
    assert main(args) == 0
    assert main(args) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert main(args + ["--overwrite"]) == 0


def test_cli_rejects_unsafe_or_missing_locations(tmp_path, monkeypatch, capsys):
    root = write_tables(tmp_path / DEMO_FOLDER, build_tables())

    inside_repo = PROJECT_ROOT / "research_output_should_not_exist"
    assert main(["--mimic-dir", str(root), "--output-dir", str(inside_repo)]) == 2
    assert not inside_repo.exists()

    monkeypatch.setenv("OneDrive", str(tmp_path / "cloud"))
    assert main(["--mimic-dir", str(root), "--output-dir", str(tmp_path / "cloud" / "out")]) == 2
    assert not (tmp_path / "cloud").exists()
    monkeypatch.delenv("OneDrive")

    assert main(["--mimic-dir", str(root), "--output-dir", str(root / "out")]) == 2
    assert main(["--mimic-dir", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "out")]) == 2
    assert main([]) == 2  # nothing configured (MIMIC_IV_DIR / MIMIC_OUTPUT_DIR empty in tests)
    errors = capsys.readouterr().err
    assert "Git repository" in errors and "OneDrive" in errors and "MIMIC_IV_DIR" in errors
    assert not (tmp_path / "out").exists()


def test_builder_never_touches_synthetic_assets(tmp_path):
    from src.database.db import execute_query
    before = execute_query("SELECT COUNT(*) AS n FROM patients")[0]["n"]
    artifacts = sorted(p.name for p in Config.MODELS_STORE_DIR.iterdir())
    _build(tmp_path)
    assert execute_query("SELECT COUNT(*) AS n FROM patients")[0]["n"] == before
    assert sorted(p.name for p in Config.MODELS_STORE_DIR.iterdir()) == artifacts
