"""Phase 2: train and evaluate Task 1 baselines on the real MIMIC-IV (Demo) research dataset.

Usage:
    python -m src.training.train_mimic_t1 --research-dir C:\\research\\medihaven-output [--overwrite]

--research-dir defaults to MIMIC_OUTPUT_DIR. The Task 1 dataset is read from
<research-dir>/medihaven_t1_inhospital_mortality_icu24h_v1.2.0 (override with --dataset-dir), and
every output is written to a new versioned folder, never into the repository or Phase 1 folders:

    <research-dir>/medihaven_t1_phase2_baselines_v3.0.0/
        models/<model>.joblib     final pipelines fitted on all eligible stays (+ provenance)
        metrics.json              aggregate metrics for the primary and sensitivity analyses
        feature_config.json       permitted features, provenance, exclusions, preprocessing
        split_metadata.json       fold counts and a fold-assignment fingerprint (no identifiers)
        report.md                 human-readable evaluation report with limitations
        run_manifest.json         versions, seed, timestamp, software, output file hashes
    <research-dir>/logs/train_mimic_t1.log  (aggregate messages only)

No patient-level predictions or identifiers are written. Nothing is sent to external services.
These models are research baselines on demo data: not clinically validated, not for patient care.
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.tasks.build_mimic_t1 import LABEL_COLUMN, verify_runtime_paths
from src.tasks.mortality_icu_24h import ANOMALY_COLUMN, TASK_ID
from src.training.dataset import (
    EXPECTED_DATASET_VERSION,
    EXPECTED_TASK_VERSION,
    DatasetValidationError,
    default_dataset_dir,
    load_task1_dataset,
)
from src.training.evaluation import EVALUATION_STRATEGY, LIMITED_EVALUATION_MAX_POSITIVES, run_analyses
from src.training.features import EXCLUDED_COLUMNS, ForbiddenFeatureError, build_feature_matrix, feature_provenance, select_features
from src.training.metrics import AVERAGE_PRECISION_DEFINITION, THRESHOLD_RULE
from src.training.models import MODEL_ORDER, candidate_models, describe_models
from src.training.splits import InsufficientDataError
from src.utils.config import Config
from src.utils.research_paths import UnsafeResearchPathError, is_within, require_safe_research_path

RUN_ID = "medihaven_t1_phase2_baselines"
# 2.0.0: four real models (LR, smoothed tree, RF, MLP) + reference baseline; measurement counts not model inputs
# 3.0.0: the decision tree is distilled from a random forest's out-of-bag risks (more stable than 11 deaths)
RUN_VERSION = "3.0.0"
DEFAULT_SEED = 42
DEFAULT_FOLDS = 5
DEFAULT_BOOTSTRAP = 2000
RESEARCH_LOGGER_NAME = "medihaven.research.train_mimic_t1"
DISCLAIMER = (
    "Research baselines trained and evaluated on the MIMIC-IV Clinical Database Demo. The demo is far too "
    "small for stable estimates; these results validate the pipeline only. The models are NOT clinically "
    "validated, NOT production-ready and must NOT be used for patient-care decisions."
)


class TrainingConfigError(RuntimeError):
    """Raised for missing/invalid training configuration."""


class OutputSwapError(RuntimeError):
    """Raised when a completed run cannot be moved into place; no complete run is deleted."""


def output_dir_for(research_dir: Path) -> Path:
    return Path(research_dir) / f"{RUN_ID}_v{RUN_VERSION}"


def training_log_path(research_dir: Path) -> Path:
    return Path(research_dir) / "logs" / "train_mimic_t1.log"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_log(research_dir: Path) -> logging.Logger:
    path = training_log_path(research_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    research_logger = logging.getLogger(RESEARCH_LOGGER_NAME)
    _close_log(research_logger)
    research_logger.setLevel(logging.INFO)
    research_logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    research_logger.addHandler(handler)
    return research_logger


def _close_log(research_logger: logging.Logger) -> None:
    for handler in list(research_logger.handlers):
        handler.close()
        research_logger.removeHandler(handler)


def resolve_locations(research_dir: Optional[Path], dataset_dir: Optional[Path]):
    research_dir = research_dir or Config.MIMIC_OUTPUT_DIR
    if research_dir is None:
        raise TrainingConfigError("No research directory configured. Pass --research-dir or set MIMIC_OUTPUT_DIR.")
    research_dir = require_safe_research_path(research_dir, "the research output directory")
    dataset_dir = require_safe_research_path(dataset_dir or default_dataset_dir(research_dir), "the Task 1 dataset directory")
    out_dir = require_safe_research_path(output_dir_for(research_dir), "the Phase 2 output directory")
    if is_within(out_dir, dataset_dir) or is_within(dataset_dir, out_dir):
        raise TrainingConfigError("The Phase 2 output directory and the dataset directory must not contain each other.")
    return research_dir, dataset_dir, out_dir


def train_and_evaluate(
    research_dir: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
    seed: int = DEFAULT_SEED,
    n_folds: int = DEFAULT_FOLDS,
    n_boot: int = DEFAULT_BOOTSTRAP,
    overwrite: bool = False,
    command: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Runs the full Phase 2 workflow and returns the run manifest (aggregate information only)."""
    research_dir, dataset_dir, out_dir = resolve_locations(research_dir, dataset_dir)
    _recover_orphaned_run(out_dir)
    if (out_dir / "run_manifest.json").exists() and not overwrite:
        raise FileExistsError(f"Phase 2 outputs already exist at {out_dir}. Re-run with --overwrite to replace them.")

    dataset = load_task1_dataset(dataset_dir)  # validates before anything is written
    features, provenance_excluded = select_features(dataset.features)
    staging = out_dir.with_name(out_dir.name + ".staging")
    runtime_paths = verify_runtime_paths({
        "output_dir": out_dir, "staging_dir": staging, "training_log": training_log_path(research_dir),
    })

    log = _open_log(research_dir)
    keep_staging = False
    try:
        log.info(f"Phase 2 run started: {RUN_ID} v{RUN_VERSION}, dataset v{EXPECTED_DATASET_VERSION}, seed {seed}")
        results = run_analyses(dataset.frame, features, n_folds, seed, n_boot)
        manifest = _write_outputs(dataset, features, provenance_excluded, results, seed, n_folds, n_boot,
                                  research_dir, out_dir, staging, runtime_paths, command)
        primary = results["primary"]
        log.info(f"Primary population: {json.dumps(primary['population'])}")
        log.info(f"Split: {n_folds}-fold grouped stratified, assignment {primary['split']['assignment_sha256'][:12]}")
        for name in MODEL_ORDER:
            m = primary["models"][name]["pooled_out_of_fold"]
            log.info(f"{name}: auroc={m['auroc']} average_precision={m['average_precision']} brier={m['brier_score']}")
        log.info(f"Sensitivity populations: evaluation_only={json.dumps(results['sensitivity_evaluation_only']['population'])} "
                 f"retrained={json.dumps(results['sensitivity_retrained']['population'])}")
        log.info(f"Phase 2 run finished: {out_dir}")
        return manifest
    except OutputSwapError:
        keep_staging = True  # the staging folder holds a complete run; never delete it during cleanup
        log.error("Phase 2 run completed but could not be moved into place; kept in the staging folder")
        raise
    except Exception as e:
        log.error(f"Phase 2 run failed: {type(e).__name__}")
        raise
    finally:
        _close_log(log)
        if not keep_staging:
            shutil.rmtree(staging, ignore_errors=True)


def _is_complete_run(folder: Path) -> bool:
    return (folder / "run_manifest.json").is_file()


def _recover_orphaned_run(out_dir: Path) -> None:
    """After an interrupted swap, put a complete backup or staged run back in place before continuing."""
    if out_dir.exists():
        return
    # A complete staging folder is the newer run (its manifest is written last); prefer it over the backup
    for candidate in (out_dir.with_name(out_dir.name + ".staging"), out_dir.with_name(out_dir.name + ".previous")):
        if _is_complete_run(candidate):
            os.replace(candidate, out_dir)
            return


def _swap_into_place(staging: Path, out_dir: Path) -> None:
    """Moves a completed staging folder to out_dir, restoring the previous run if the move fails."""
    previous = out_dir.with_name(out_dir.name + ".previous")
    if previous.exists():
        if out_dir.exists():
            shutil.rmtree(previous)  # stale backup; the current run is intact
        else:
            os.replace(previous, out_dir)
    had_previous = out_dir.exists()
    if had_previous:
        os.replace(out_dir, previous)
    try:
        os.replace(staging, out_dir)
    except OSError as e:
        restored = False
        if had_previous:
            try:
                os.replace(previous, out_dir)
                restored = True
            except OSError:
                pass  # the backup stays at .previous and is recovered by the next run
        raise OutputSwapError(
            f"The completed run could not be moved into place ({type(e).__name__}). "
            f"{'The previous run was restored. ' if restored else ''}The new run is kept at {staging}."
        ) from e
    shutil.rmtree(previous, ignore_errors=True)


def _write_outputs(dataset, features, provenance_excluded, results, seed, n_folds, n_boot,
                   research_dir, out_dir, staging, runtime_paths, command) -> Dict[str, object]:
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "models").mkdir(parents=True)

    dataset_ref = {
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_version": dataset.manifest["dataset_version"],
        "dataset_content_sha256": dataset.manifest["content_sha256"],
        "feature_spec_version": dataset.manifest["feature_spec_version"],
        "task_id": TASK_ID,
        "task_version": EXPECTED_TASK_VERSION,
        "source_release": dataset.manifest["source"]["release"],
    }

    # Final models: fitted on every eligible stay; their expected performance is the CV estimate
    X = build_feature_matrix(dataset.frame, features)
    y = dataset.frame[LABEL_COLUMN].to_numpy(dtype=int)
    for name, pipeline in candidate_models(seed).items():
        fitted = pipeline.fit(X, y)
        joblib.dump({
            "pipeline": fitted,
            "features": features,
            "trained_on": "all eligible stays of the primary population",
            "seed": seed,
            "run_id": RUN_ID,
            "run_version": RUN_VERSION,
            **dataset_ref,
            "disclaimer": DISCLAIMER,
        }, staging / "models" / f"{name}.joblib")

    definitions = {
        "evaluation_strategy": EVALUATION_STRATEGY,
        "threshold_rule": THRESHOLD_RULE,
        "average_precision": AVERAGE_PRECISION_DEFINITION,
        "bootstrap": f"{n_boot} subject-level resamples of pooled out-of-fold predictions; percentile 95% CI; "
                     "reflects case-mix sampling only, not model-refitting variability",
        "limited_evaluation_rule": f"flagged 'limited' when positives < {LIMITED_EVALUATION_MAX_POSITIVES}",
        "undefined_metrics": "reported as null with the reason under 'undefined'; never replaced by 0",
    }
    _dump(staging / "metrics.json", {"disclaimer": DISCLAIMER, "dataset": dataset_ref,
                                     "definitions": definitions, "analyses": results})
    _dump(staging / "feature_config.json", {
        "dataset": dataset_ref,
        "features": [{"name": f, "provenance": feature_provenance(f)} for f in features],
        "feature_count": len(features),
        "excluded_columns": EXCLUDED_COLUMNS,
        "excluded_for_unestablished_provenance": provenance_excluded,
        "anomaly_flag": {"column": ANOMALY_COLUMN, "use": "sensitivity analysis only; never a model input"},
        "preprocessing_and_models": describe_models(seed),
        "missing_values": "median imputation fitted on training folds only; <variable>_missing flags retained",
    })
    _dump(staging / "split_metadata.json", {
        "dataset": dataset_ref,
        "grouping_column": "src_subject_id (identifiers are not stored here)",
        "primary": {"population": results["primary"]["population"], "split": results["primary"]["split"]},
        "sensitivity_retrained": {
            "population": results["sensitivity_retrained"]["population"],
            "split": results["sensitivity_retrained"].get("split"),
            "evaluation": results["sensitivity_retrained"]["evaluation"],
        },
        "sensitivity_evaluation_only": {
            "population": results["sensitivity_evaluation_only"]["population"],
            "split": results["sensitivity_evaluation_only"]["split"],
        },
    })
    (staging / "report.md").write_text(render_report(dataset_ref, features, results, definitions, seed, n_folds),
                                       encoding="utf-8")

    manifest = {
        "run_id": RUN_ID,
        "run_version": RUN_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_ref,
        "configuration": {
            "seed": seed, "n_folds": n_folds, "bootstrap_resamples": n_boot,
            "research_dir": str(research_dir), "dataset_dir": str(dataset.dataset_dir),
            "output_dir": str(out_dir), "runtime_paths": runtime_paths,
            "command": command or [],
        },
        "software": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
        },
        "summary": {
            "primary_population": results["primary"]["population"],
            "evaluation_status": results["primary"]["evaluation"],
            "models": list(MODEL_ORDER),
        },
        "disclaimer": DISCLAIMER,
    }
    manifest["output_files_sha256"] = {
        p.relative_to(staging).as_posix(): _sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()
    }
    _dump(staging / "run_manifest.json", manifest)

    # Swap the completed staging folder into place (an existing run is replaced only with --overwrite)
    _swap_into_place(staging, out_dir)
    return manifest


def _dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------
def _fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ci(ci: Dict[str, object]) -> str:
    return "n/a" if ci.get("low") is None else f"{ci['low']:.3f}–{ci['high']:.3f}"


def _metric_rows(models: Dict[str, Dict[str, object]]) -> List[str]:
    rows = [
        "| Model | AUROC (95% CI) | Average precision (95% CI) | Brier (95% CI) | Sens. | Spec. | Prec. | F1 | TP/FP/TN/FN |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name in MODEL_ORDER:
        m, ci = models[name]["pooled_out_of_fold"], models[name]["bootstrap_ci_95"]
        cm = m["confusion_matrix"]
        rows.append(
            f"| {name} | {_fmt(m['auroc'])} ({_ci(ci['auroc'])}) | {_fmt(m['average_precision'])} "
            f"({_ci(ci['average_precision'])}) | {_fmt(m['brier_score'])} ({_ci(ci['brier_score'])}) | "
            f"{_fmt(m['sensitivity'], 2)} | {_fmt(m['specificity'], 2)} | {_fmt(m['precision'], 2)} | "
            f"{_fmt(m['f1'], 2)} | {cm['tp']}/{cm['fp']}/{cm['tn']}/{cm['fn']} |"
        )
    undefined = {n: models[n]["pooled_out_of_fold"]["undefined"] for n in MODEL_ORDER
                 if models[n]["pooled_out_of_fold"]["undefined"]}
    if undefined:
        rows.append("")
        rows.append("Undefined metrics: " + "; ".join(f"{n}: {u}" for n, u in undefined.items()))
    return rows


def _population_line(pop: Dict[str, object]) -> str:
    return (f"{pop['stays']} stays, {pop['subjects']} subjects, {pop['positives']} positive / {pop['negatives']} negative "
            f"(prevalence {pop['prevalence']:.1%}), outcome-anomaly stays: {pop['outcome_anomaly_stays']}")


def render_report(dataset_ref, features, results, definitions, seed, n_folds) -> str:
    primary = results["primary"]
    lines = [
        "# MediHaven Phase 2 — Task 1 Baseline Evaluation (research only)",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "## Task and data",
        f"- Task: `{dataset_ref['task_id']}` v{dataset_ref['task_version']} — in-hospital mortality after ICU intime + 24 h, "
        "using only information available at that time.",
        f"- Dataset: `{dataset_ref['dataset_id']}` v{dataset_ref['dataset_version']} (source {dataset_ref['source_release']}, "
        f"feature spec {dataset_ref['feature_spec_version']}).",
        f"- Features: {len(features)} permitted pre-prediction features; identifiers, index time, label, outcome-anomaly "
        "flag and version stamps excluded (see feature_config.json).",
        "",
        "## Evaluation design",
        f"- {definitions['evaluation_strategy']}; {n_folds} folds, seed {seed}. No subject appears in both the training "
        "and test part of any fold. No hyperparameters were tuned.",
        f"- Evaluation status: **{primary['evaluation']['status']}** — {primary['evaluation']['reason']}.",
        f"- Threshold: {definitions['threshold_rule']}.",
        f"- Average precision: {definitions['average_precision']}.",
        f"- Confidence intervals: {definitions['bootstrap']}.",
        "",
        "## Primary analysis (all eligible stays)",
        f"Population: {_population_line(primary['population'])}.",
        "",
        "| Fold | Train stays (positives) | Test stays | Test positives | Test negatives |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ]
    for f in primary["split"]["folds"]:
        lines.append(f"| {f['fold']} | {f['train_stays']} ({f['train_positives']}) | {f['test_stays']} | "
                     f"{f['test_positives']} | {f['test_negatives']} |")
    lines += ["", "Pooled out-of-fold metrics:", ""] + _metric_rows(primary["models"])
    lines += ["", "Across-fold spread of AUROC (mean ± SD over folds where defined):", ""]
    for name in MODEL_ORDER:
        s = primary["models"][name]["fold_spread"]["auroc"]
        sd = "" if s["sd"] is None else f" ± {s['sd']:.3f}"
        lines.append(f"- {name}: {_fmt(s['mean'])}{sd} ({s['folds_defined']} folds)")

    eval_only = results["sensitivity_evaluation_only"]
    retrained = results["sensitivity_retrained"]
    lines += [
        "",
        "## Sensitivity analysis: outcome-anomaly stays",
        "Stays flagged `outcome_anomaly_deathtime_after_discharge` (flag 1 + DIED, deathtime after dischtime) are "
        "positive in the primary analysis. The flag is never a model input.",
        "",
        "### A. Evaluation-only exclusion (same folds and models as primary)",
        f"Population: {_population_line(eval_only['population'])}.",
        "",
    ] + _metric_rows(eval_only["models"]) + [
        "",
        "### B. Exclusion before splitting and training (new folds)",
        f"Population: {_population_line(retrained['population'])}.",
    ]
    if retrained["evaluation"]["status"] == "not_evaluable":
        lines.append(f"Not evaluable: {retrained['evaluation']['reason']}")
    else:
        lines += ["Folds are re-drawn over the smaller population, so B is not fold-for-fold comparable with the "
                  "primary analysis; compare the overall pattern only.", ""] + _metric_rows(retrained["models"])

    lines += [
        "",
        "## Interpretation and limitations",
        "- With this few positive stays, a single additional true or false positive moves every metric visibly; "
        "the bootstrap intervals are wide and still understate uncertainty (no model-refitting variability).",
        "- The prior-only baseline's pooled AUROC is not exactly 0.5: each fold predicts its own training "
        "prevalence, so pooling folds creates small artificial differences between stays (its per-fold AUROC "
        "is 0.5). Compare models with this in mind.",
        "- Cross-validated estimates describe the modeling procedure on this demo population only; there is no "
        "external or temporal validation.",
        "- The saved models were refitted on all eligible stays and are provided for traceability, not use.",
        "- Clinically meaningful claims require the full credentialed MIMIC-IV (and ideally external data), a "
        "pre-specified analysis plan, calibration assessment, subgroup analyses and clinical review.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def format_summary(manifest: Dict[str, object], metrics_file: Path) -> List[str]:
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))["analyses"]
    lines = [f"MediHaven Phase 2 ({RUN_ID} v{RUN_VERSION}) — dataset v{manifest['dataset']['dataset_version']}, "
             f"task v{manifest['dataset']['task_version']}, seed {manifest['configuration']['seed']}",
             f"Evaluation: {manifest['summary']['evaluation_status']['status']} — {manifest['summary']['evaluation_status']['reason']}"]
    for key in ("primary", "sensitivity_evaluation_only", "sensitivity_retrained"):
        analysis = metrics[key]
        lines.append(f"[{key}] {_population_line(analysis['population'])}")
        if "models" not in analysis:
            lines.append(f"  not evaluable: {analysis['evaluation']['reason']}")
            continue
        for name in MODEL_ORDER:
            m = analysis["models"][name]["pooled_out_of_fold"]
            lines.append(f"  {name:<20} AUROC {_fmt(m['auroc'])}  AP {_fmt(m['average_precision'])}  "
                         f"Brier {_fmt(m['brier_score'])}  Sens {_fmt(m['sensitivity'], 2)}  Spec {_fmt(m['specificity'], 2)}")
    lines.append(f"Outputs: {manifest['configuration']['output_dir']}")
    lines.append("Research baselines on demo data only — not clinically validated, not for patient care.")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate MediHaven Task 1 baselines (research only).")
    parser.add_argument("--research-dir", type=Path, default=None, help="Research output folder (default: MIMIC_OUTPUT_DIR)")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Task 1 dataset folder (default: inside --research-dir)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="Subject-level bootstrap resamples")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing Phase 2 outputs of this version")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks for unexpected errors (local use only)")
    args = parser.parse_args(argv)
    command = ["python", "-m", "src.training.train_mimic_t1"] + (argv if argv is not None else sys.argv[1:])

    try:
        manifest = train_and_evaluate(args.research_dir, args.dataset_dir, args.seed, args.folds, args.bootstrap,
                                      args.overwrite, command)
    except (TrainingConfigError, UnsafeResearchPathError, DatasetValidationError, ForbiddenFeatureError,
            InsufficientDataError, FileExistsError, OutputSwapError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # unexpected: messages could echo data values, so hide them by default
        print(f"ERROR: unexpected {type(e).__name__} during Phase 2 training. "
              f"Re-run with --debug locally to see the traceback.", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1

    print("\n".join(format_summary(manifest, Path(manifest["configuration"]["output_dir"]) / "metrics.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
