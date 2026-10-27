"""Phase 2 tests: dataset validation, leakage guards, grouped splits, preprocessing isolation,
metrics, sensitivity analyses, reproducibility and output safety.

All data are small synthetic fixtures (tests/training_fixtures.py); no real MIMIC-IV records.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.datasources.mimic.mappings import feature_columns
from src.tasks.build_mimic_t1 import ID_COLUMNS, LABEL_COLUMN, content_sha256
from src.tasks.mortality_icu_24h import ANOMALY_COLUMN
from src.training.dataset import DatasetValidationError, load_task1_dataset
from src.training.evaluation import run_analyses
from src.training.features import (
    EXCLUDED_COLUMNS,
    MEASUREMENT_COUNT_REASON,
    ForbiddenFeatureError,
    build_feature_matrix,
    forbidden_reason,
    select_features,
)
from src.training.metrics import binary_metrics
from src.training.models import MODEL_ORDER, DistilledDecisionTree, SmoothedDecisionTree, candidate_models
from src.training.splits import InsufficientDataError, make_grouped_folds
from src.training.train_mimic_t1 import main, output_dir_for, training_log_path, train_and_evaluate
from src.utils.config import Config
from tests.isolation_utils import PROJECT_ROOT
from tests.training_fixtures import SUBJECT_BASE, make_task1_frame, write_task1_dataset

FAST = {"n_folds": 5, "seed": 7, "n_boot": 100}
MODEL_INPUTS = [c for c in feature_columns() if not c.endswith("_count")]


def _cli(research_dir, *extra):
    return main(["--research-dir", str(research_dir), "--bootstrap", "100", *extra])


# ------------------------------------------------------------------------------
# Dataset schema and label validation
# ------------------------------------------------------------------------------
def test_valid_dataset_loads(tmp_path):
    dataset = load_task1_dataset(write_task1_dataset(tmp_path))
    assert dataset.features == feature_columns()
    assert len(dataset.frame) == 70


@pytest.mark.parametrize("overrides, message", [
    ({"dataset_version": "1.1.0"}, "dataset_version"),
    ({"feature_spec_version": "other"}, "feature_spec_version"),
    ({"task": {"task_id": "t1_inhospital_mortality_after_icu24h", "task_version": "2"}}, "task"),
    ({"row_count": 5}, "rows"),
])
def test_incompatible_metadata_is_rejected(tmp_path, overrides, message):
    with pytest.raises(DatasetValidationError, match=message):
        load_task1_dataset(write_task1_dataset(tmp_path, manifest_overrides=overrides))


def _label_value_two(f):
    f[LABEL_COLUMN] = f[LABEL_COLUMN].replace({1: 2})


def _label_missing(f):
    f[LABEL_COLUMN] = f[LABEL_COLUMN].astype(float)
    f.loc[0, LABEL_COLUMN] = np.nan


def _anomaly_on_negative(f):
    f.loc[f.index[f[LABEL_COLUMN] == 0][0], ANOMALY_COLUMN] = 1


def _duplicate_stay(f):
    f.loc[1, "src_stay_id"] = f.loc[0, "src_stay_id"]


@pytest.mark.parametrize("change, message", [
    (_label_value_two, "0/1"),
    (_label_missing, "0/1"),
    (_anomaly_on_negative, "anomaly"),
    (_duplicate_stay, "one row per ICU stay"),
])
def test_invalid_labels_and_identifiers_are_rejected(tmp_path, change, message):
    frame = make_task1_frame()
    change(frame)
    with pytest.raises(DatasetValidationError, match=message):
        load_task1_dataset(write_task1_dataset(tmp_path, frame=frame))


def test_modified_content_fails_hash_check(tmp_path):
    def tamper(frame):
        frame = frame.copy()
        frame.loc[0, "heart_rate_mean"] += 1
        return frame
    with pytest.raises(DatasetValidationError, match="content hash"):
        load_task1_dataset(write_task1_dataset(tmp_path, mutate_after_hash=tamper))


def test_unexpected_column_and_missing_dataset_are_rejected(tmp_path):
    frame = make_task1_frame()
    frame["discharge_location"] = "HOME"
    with pytest.raises(DatasetValidationError, match="unexpected"):
        load_task1_dataset(write_task1_dataset(tmp_path / "a", frame=frame))
    with pytest.raises(DatasetValidationError, match="build_mimic_t1"):
        load_task1_dataset(tmp_path / "missing")


# ------------------------------------------------------------------------------
# Forbidden features
# ------------------------------------------------------------------------------
def test_feature_spec_passes_leakage_guard():
    permitted, excluded = select_features(feature_columns())
    assert permitted == MODEL_INPUTS and len(permitted) == 74
    # measurement counts are dropped as care-process signals; missingness stays visible via _missing
    assert set(excluded) == {c for c in feature_columns() if c.endswith("_count")} and len(excluded) == 14
    assert set(excluded.values()) == {MEASUREMENT_COUNT_REASON}
    assert all(c.replace("_count", "_missing") in permitted for c in excluded)


@pytest.mark.parametrize("column", list(EXCLUDED_COLUMNS) + [
    "deathtime_hours", "discharge_location", "dischtime", "hospital_expire_flag", "dod", "los",
    "risk_tier", "risk_score", "predicted_risk", "icu_outtime", "subject_age", "hadm_count", "outcome_x",
])
def test_forbidden_columns_cannot_enter_training_matrix(column):
    assert forbidden_reason(column) is not None
    with pytest.raises(ForbiddenFeatureError):
        select_features(feature_columns() + [column])
    frame = make_task1_frame()
    frame[column] = 0.0
    with pytest.raises(ForbiddenFeatureError):
        build_feature_matrix(frame, feature_columns() + [column])


def test_feature_without_established_provenance_is_excluded():
    permitted, excluded = select_features(feature_columns() + ["mystery_score"])
    assert "mystery_score" not in permitted
    assert "cannot be established" in excluded["mystery_score"]


def test_trained_models_never_see_identifiers_label_or_anomaly_flag(tmp_path):
    write_task1_dataset(tmp_path)
    assert _cli(tmp_path) == 0
    forbidden = set(ID_COLUMNS) | {LABEL_COLUMN, ANOMALY_COLUMN, "index_time", "dataset_version", "feature_spec_version"}
    for name in MODEL_ORDER:
        bundle = joblib.load(output_dir_for(tmp_path) / "models" / f"{name}.joblib")
        assert bundle["features"] == MODEL_INPUTS
        if name != "dummy_prior":
            assert list(bundle["pipeline"].feature_names_in_) == MODEL_INPUTS
            assert not forbidden & set(bundle["pipeline"].feature_names_in_)
        assert bundle["dataset_version"] == "1.2.0" and bundle["task_version"] == "3"


# ------------------------------------------------------------------------------
# Splits and preprocessing
# ------------------------------------------------------------------------------
def test_no_subject_in_both_training_and_test():
    frame = make_task1_frame()
    y, groups = frame[LABEL_COLUMN].to_numpy(), frame["src_subject_id"].to_numpy()
    folds = make_grouped_folds(y, groups, n_folds=5, seed=3)
    tested = np.concatenate([test for _, test in folds])
    assert sorted(tested) == list(range(len(frame)))  # every stay tested exactly once
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        assert y[test].sum() > 0 and (y[test] == 0).sum() > 0


def test_preprocessing_is_fitted_on_training_rows_only():
    frame = make_task1_frame()
    X = build_feature_matrix(frame, feature_columns())
    y = frame[LABEL_COLUMN].to_numpy()
    train, test = make_grouped_folds(y, frame["src_subject_id"].to_numpy(), 5, 3)[0]
    X.iloc[test, X.columns.get_loc("glucose_last")] = 1000.0  # extreme values only in the test part
    pipeline = clone(candidate_models(3)["logistic_regression"]).fit(X.iloc[train], y[train])
    medians = dict(zip(feature_columns(), pipeline.named_steps["impute"].statistics_))
    assert medians["glucose_last"] == pytest.approx(np.nanmedian(X.iloc[train]["glucose_last"]))
    assert medians["glucose_last"] != pytest.approx(np.nanmedian(X["glucose_last"]))
    assert pipeline.named_steps["scale"].mean_.max() < 1000.0 / 10


def test_missing_values_and_empty_columns_are_handled(tmp_path):
    frame = make_task1_frame()
    frame["lactate_min"] = np.nan  # entirely missing column
    results = run_analyses(frame, feature_columns(), **FAST)
    for name in MODEL_ORDER:
        assert results["primary"]["models"][name]["pooled_out_of_fold"]["brier_score"] is not None


def test_too_few_positives_fail_safely(tmp_path, capsys):
    frame = make_task1_frame(positives=3, anomalies=0)
    with pytest.raises(InsufficientDataError, match="Too few"):
        make_grouped_folds(frame[LABEL_COLUMN].to_numpy(), frame["src_subject_id"].to_numpy(), 5, 0)
    write_task1_dataset(tmp_path, frame=frame)
    assert _cli(tmp_path) == 2
    assert "Too few" in capsys.readouterr().err
    assert not output_dir_for(tmp_path).exists()
    assert not output_dir_for(tmp_path).with_name(output_dir_for(tmp_path).name + ".staging").exists()


def test_smoothed_decision_tree_never_outputs_zero_or_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] > 1.2).astype(int)  # few positives: a plain tree would have pure leaves
    plain = SmoothedDecisionTree(max_depth=3, min_samples_leaf=8, smoothing=0.0, random_state=0).fit(X, y)
    assert {0.0, 1.0} & set(plain.predict_proba(X)[:, 1])
    smoothed = SmoothedDecisionTree(max_depth=3, min_samples_leaf=8, smoothing=5.0, random_state=0).fit(X, y)
    p = smoothed.predict_proba(X)[:, 1]
    assert ((p > 0) & (p < 1)).all()
    np.testing.assert_allclose(smoothed.predict_proba(X).sum(axis=1), 1.0)
    # every leaf rate lies between the leaf's raw rate and the training death rate
    leaves = smoothed.tree_.apply(X)
    for leaf, rate in smoothed.leaf_rate_.items():
        raw = y[leaves == leaf].mean()
        assert min(raw, smoothed.prior_) - 1e-12 <= rate <= max(raw, smoothed.prior_) + 1e-12


def test_distilled_tree_imitates_its_forest_and_never_claims_certainty():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 5))
    y = ((X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.7, size=120)) > 1.3).astype(int)  # ~10 % positives
    tree = DistilledDecisionTree(max_depth=4, min_samples_leaf=5, random_state=0).fit(X, y)
    p = tree.predict_proba(X)[:, 1]
    assert ((p > 0) & (p < 1)).all()
    np.testing.assert_allclose(tree.predict_proba(X).sum(axis=1), 1.0)
    assert len(np.unique(np.round(p, 6))) >= 6  # graded risks, not a handful of pure-leaf values
    forest = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3, max_features="sqrt",
                                    random_state=0).fit(X, y)
    assert np.corrcoef(p, forest.predict_proba(X)[:, 1])[0, 1] > 0.7  # follows the teacher's signal
    assert isinstance(candidate_models(0)["decision_tree"].named_steps["model"], DistilledDecisionTree)


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------
def test_single_class_subsets_report_undefined_metrics():
    negatives = binary_metrics([0, 0, 0], [0.1, 0.4, 0.2], 0.3)
    assert negatives["auroc"] is None and "both" in negatives["undefined"]["auroc"]
    assert negatives["average_precision"] is None and negatives["sensitivity"] is None
    assert negatives["specificity"] == pytest.approx(2 / 3)
    assert negatives["brier_score"] is not None

    positives = binary_metrics([1, 1], [0.9, 0.2], 0.5)
    assert positives["specificity"] is None and positives["auroc"] is None
    assert positives["sensitivity"] == 0.5

    nothing_flagged = binary_metrics([0, 1, 0], [0.1, 0.2, 0.1], 0.9)
    assert nothing_flagged["precision"] is None and nothing_flagged["f1"] is None
    assert nothing_flagged["sensitivity"] == 0.0

    all_wrong = binary_metrics([0, 1], [0.9, 0.1], 0.5)
    assert all_wrong["precision"] == 0.0 and all_wrong["sensitivity"] == 0.0
    assert all_wrong["f1"] is None and "both 0" in all_wrong["undefined"]["f1"]


def test_thresholded_metrics_and_confusion_matrix():
    m = binary_metrics([1, 1, 0, 0, 0], [0.8, 0.3, 0.6, 0.1, 0.2], 0.5)
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert (m["sensitivity"], m["specificity"], m["precision"]) == (0.5, pytest.approx(2 / 3), 0.5)
    assert m["auroc"] == pytest.approx(5 / 6)


# ------------------------------------------------------------------------------
# Sensitivity analyses and reproducibility
# ------------------------------------------------------------------------------
def test_sensitivity_population_counts():
    frame = make_task1_frame(positives=14, anomalies=3)
    results = run_analyses(frame, feature_columns(), **FAST)
    primary = results["primary"]["population"]
    for key in ("sensitivity_evaluation_only", "sensitivity_retrained"):
        pop = results[key]["population"]
        assert pop["stays"] == primary["stays"] - 3
        assert pop["positives"] == primary["positives"] - 3
        assert pop["negatives"] == primary["negatives"]
        assert pop["outcome_anomaly_stays"] == 0
    assert results["sensitivity_retrained"]["split"]["assignment_sha256"] != results["primary"]["split"]["assignment_sha256"]


def test_sensitivity_reports_not_evaluable_instead_of_forcing_metrics():
    frame = make_task1_frame(positives=6, anomalies=3)  # 3 positives remain for 5 folds
    results = run_analyses(frame, feature_columns(), **FAST)
    assert "models" in results["primary"]
    retrained = results["sensitivity_retrained"]
    assert retrained["evaluation"]["status"] == "not_evaluable" and "models" not in retrained
    assert retrained["population"]["positives"] == 3


def test_same_seed_reproduces_results_and_folds():
    frame = make_task1_frame()
    first = json.dumps(run_analyses(frame, feature_columns(), **FAST), sort_keys=True, default=str)
    second = json.dumps(run_analyses(frame, feature_columns(), **FAST), sort_keys=True, default=str)
    assert first == second
    other = run_analyses(frame, feature_columns(), n_folds=5, seed=8, n_boot=100)
    assert other["primary"]["split"]["assignment_sha256"] != json.loads(first)["primary"]["split"]["assignment_sha256"]


# ------------------------------------------------------------------------------
# Outputs, path safety and data safeguards
# ------------------------------------------------------------------------------
def test_outputs_are_complete_hashed_and_free_of_identifiers(tmp_path, capsys):
    write_task1_dataset(tmp_path)
    assert _cli(tmp_path) == 0
    printed = capsys.readouterr().out
    out = output_dir_for(tmp_path)
    expected = {"metrics.json", "feature_config.json", "split_metadata.json", "report.md", "run_manifest.json"}
    assert expected <= {p.name for p in out.iterdir()}
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    for relative, digest in manifest["output_files_sha256"].items():
        import hashlib
        assert hashlib.sha256((out / relative).read_bytes()).hexdigest() == digest
    assert manifest["dataset"]["dataset_version"] == "1.2.0" and manifest["configuration"]["seed"] == 42
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "NOT clinically validated" in report and "limited" in report

    texts = [printed, training_log_path(tmp_path).read_text(encoding="utf-8")]
    texts += [(out / f).read_text(encoding="utf-8") for f in expected]
    ids = {str(SUBJECT_BASE + i) for i in range(60)} | {str(78000000 + i) for i in range(70)}
    for text in texts:
        assert not any(i in text for i in ids), "an identifier was written or printed"


def test_unsafe_locations_and_overwrite_are_refused(tmp_path, monkeypatch, capsys):
    assert _cli(PROJECT_ROOT / "research_output_should_not_exist") == 2
    assert not (PROJECT_ROOT / "research_output_should_not_exist").exists()

    monkeypatch.setenv("OneDrive", str(tmp_path / "cloud"))
    assert _cli(tmp_path / "cloud" / "research") == 2
    monkeypatch.delenv("OneDrive")

    assert _cli(tmp_path / "empty") == 2  # dataset missing
    assert not output_dir_for(tmp_path / "empty").exists()
    assert main([]) == 2  # MIMIC_OUTPUT_DIR is empty in tests

    write_task1_dataset(tmp_path / "ok")
    assert _cli(tmp_path / "ok") == 0
    assert _cli(tmp_path / "ok") == 2
    assert _cli(tmp_path / "ok", "--overwrite") == 0
    errors = capsys.readouterr().err
    assert "Git repository" in errors and "OneDrive" in errors and "--overwrite" in errors


def _complete_run(folder):
    """True if folder holds a run whose files all match its manifest hashes."""
    import hashlib
    manifest_file = folder / "run_manifest.json"
    if not manifest_file.is_file():
        return False
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    return all(hashlib.sha256((folder / rel).read_bytes()).hexdigest() == digest
               for rel, digest in manifest["output_files_sha256"].items())


def _fail_swap_into_out_dir(monkeypatch):
    """Makes the final staging -> output rename fail, as a Windows file lock would."""
    import src.training.train_mimic_t1 as train_module
    real_replace = train_module.os.replace

    def flaky_replace(src, dst):
        if str(src).endswith(".staging"):
            raise PermissionError("simulated file lock")
        return real_replace(src, dst)

    monkeypatch.setattr(train_module.os, "replace", flaky_replace)


def test_failed_swap_restores_previous_run_and_keeps_new_run(tmp_path, monkeypatch):
    from src.training.train_mimic_t1 import OutputSwapError
    write_task1_dataset(tmp_path)
    train_and_evaluate(tmp_path, n_boot=50)
    out = output_dir_for(tmp_path)
    staging = out.with_name(out.name + ".staging")
    assert _complete_run(out)

    _fail_swap_into_out_dir(monkeypatch)
    with pytest.raises(OutputSwapError, match="previous run was restored"):
        train_and_evaluate(tmp_path, n_boot=50, overwrite=True)
    assert _complete_run(out), "the previous valid run must be back in place"
    assert _complete_run(staging), "the newly completed run must not be deleted during cleanup"
    assert not out.with_name(out.name + ".previous").exists()


def test_failed_first_swap_keeps_new_run_and_next_run_recovers_it(tmp_path, monkeypatch, capsys):
    write_task1_dataset(tmp_path)
    out = output_dir_for(tmp_path)
    staging = out.with_name(out.name + ".staging")
    _fail_swap_into_out_dir(monkeypatch)
    assert _cli(tmp_path) == 2
    assert "kept at" in capsys.readouterr().err
    assert not out.exists() and _complete_run(staging)

    monkeypatch.undo()
    assert _cli(tmp_path) == 2  # the recovered run now exists, so a plain re-run asks for --overwrite
    assert _complete_run(out) and not staging.exists()


def test_training_leaves_dataset_and_application_untouched(tmp_path):
    import hashlib
    from src.database.db import execute_query
    dataset_dir = write_task1_dataset(tmp_path)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset_dir.iterdir()}
    models_before = sorted((p.name, p.stat().st_size) for p in Config.MODELS_STORE_DIR.iterdir())
    patients_before = execute_query("SELECT COUNT(*) AS n FROM patients")[0]["n"]
    app_log_before = Config.LOG_FILE.read_text(encoding="utf-8") if Config.LOG_FILE.exists() else ""

    train_and_evaluate(tmp_path, n_boot=100)

    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset_dir.iterdir()} == before
    assert sorted((p.name, p.stat().st_size) for p in Config.MODELS_STORE_DIR.iterdir()) == models_before
    assert execute_query("SELECT COUNT(*) AS n FROM patients")[0]["n"] == patients_before
    app_log_after = Config.LOG_FILE.read_text(encoding="utf-8") if Config.LOG_FILE.exists() else ""
    assert "train_mimic_t1" not in app_log_after[len(app_log_before):]
    assert not (PROJECT_ROOT / ".tmp").exists()
