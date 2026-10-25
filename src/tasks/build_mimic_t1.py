"""Builds the Task 1 research dataset from a local MIMIC-IV (Demo) directory.

Usage:
    python -m src.tasks.build_mimic_t1 --mimic-dir C:\\research\\mimic-iv-demo \\
                                       --output-dir C:\\research\\medihaven-output [--overwrite]

--mimic-dir / --output-dir default to MIMIC_IV_DIR / MIMIC_OUTPUT_DIR. Both must be outside the
Git repository and outside OneDrive. The builder prints aggregate counts only (exclusions,
prevalence, coverage); it never prints patient rows, identifiers or individual labels, and it
never falls back to synthetic data.

Everything the builder writes stays under --output-dir, never in the repository's logs/:
    <output-dir>/medihaven_t1_inhospital_mortality_icu24h_v1.2.0/dataset.parquet
    <output-dir>/medihaven_t1_inhospital_mortality_icu24h_v1.2.0/manifest.json
    <output-dir>/logs/build_mimic_t1.log      (research log, aggregate messages only)
    <output-dir>/.duckdb_tmp/                 (DuckDB spill folder, removed after the build)
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

import pandas as pd

from src.datasources.mimic.layout import MimicDataError, resolve_layout
from src.datasources.mimic.mappings import (
    FEATURE_SPEC_VERSION,
    VARIABLES,
    MappingError,
    feature_columns,
    validate_item_dictionaries,
)
from src.datasources.mimic.reader import MimicIVReader
from src.tasks.features import extract_features
from src.tasks.mortality_icu_24h import ANOMALY_COLUMN, TASK_DEFINITION, build_cohort
from src.utils.config import Config
from src.utils.research_paths import (
    UnsafeResearchPathError,
    is_within,
    require_safe_research_path,
    unsafe_reasons,
)

DATASET_ID = "medihaven_t1_inhospital_mortality_icu24h"
# 1.1.0: task v2 (no post-discharge grace period); 1.2.0: task v3 (flag 1 + DIED deaths after
# dischtime labeled positive and flagged as timestamp anomalies)
DATASET_VERSION = "1.2.0"
ID_COLUMNS = ["src_subject_id", "src_hadm_id", "src_stay_id"]
LABEL_COLUMN = "label_inhospital_mortality"
RESEARCH_LOGGER_NAME = "medihaven.research.build_mimic_t1"


class BuildConfigError(RuntimeError):
    """Raised for missing/invalid builder configuration."""


def dataset_dir(output_dir: Path) -> Path:
    return output_dir / f"{DATASET_ID}_v{DATASET_VERSION}"


def research_log_path(output_dir: Path) -> Path:
    return output_dir / "logs" / "build_mimic_t1.log"


def duckdb_spill_dir(output_dir: Path) -> Path:
    return output_dir / ".duckdb_tmp"


def resolve_paths(mimic_dir: Optional[Path], output_dir: Optional[Path]):
    """Resolves and safety-checks the input and output locations."""
    mimic_dir = mimic_dir or Config.MIMIC_IV_DIR
    output_dir = output_dir or Config.MIMIC_OUTPUT_DIR
    if mimic_dir is None:
        raise BuildConfigError("No MIMIC-IV directory configured. Pass --mimic-dir or set MIMIC_IV_DIR.")
    if output_dir is None:
        raise BuildConfigError("No output directory configured. Pass --output-dir or set MIMIC_OUTPUT_DIR.")
    mimic_dir = require_safe_research_path(mimic_dir, "the MIMIC-IV input directory")
    output_dir = require_safe_research_path(output_dir, "the research output directory")
    if is_within(output_dir, mimic_dir) or is_within(mimic_dir, output_dir):
        raise BuildConfigError("The output directory and the MIMIC-IV input directory must not contain each other.")
    return mimic_dir, output_dir


def verify_runtime_paths(paths: Dict[str, Path]) -> Dict[str, str]:
    """Re-resolves every path the build writes to and fails if any is unsafe."""
    resolved = {}
    for name, path in paths.items():
        real = Path(path).resolve()
        reasons = unsafe_reasons(real)
        if reasons:
            raise UnsafeResearchPathError(f"Runtime path {name} resolved to '{real}', which is unsafe: " + " and ".join(reasons))
        resolved[name] = str(real)
    return resolved


def open_research_log(output_dir: Path) -> logging.Logger:
    """A file-only logger under the research output folder (never the app log or console)."""
    path = research_log_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    research_logger = logging.getLogger(RESEARCH_LOGGER_NAME)
    close_research_log(research_logger)
    research_logger.setLevel(logging.INFO)
    research_logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    research_logger.addHandler(handler)
    return research_logger


def close_research_log(research_logger: logging.Logger) -> None:
    for handler in list(research_logger.handlers):
        handler.close()
        research_logger.removeHandler(handler)


def content_sha256(df: pd.DataFrame) -> str:
    """Deterministic hash of the dataset's columns and values (independent of file metadata)."""
    digest = hashlib.sha256("|".join(df.columns).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return digest.hexdigest()


def build_dataset(mimic_dir: Path, output_dir: Path, overwrite: bool = False) -> Dict[str, object]:
    """Builds and writes the dataset. Returns the manifest (aggregate information only)."""
    mimic_dir, output_dir = resolve_paths(mimic_dir, output_dir)
    resolve_layout(mimic_dir)  # validate the inputs before anything is written under output_dir
    target_dir = dataset_dir(output_dir)
    if (target_dir / "dataset.parquet").exists() and not overwrite:
        raise FileExistsError(
            f"A dataset already exists at {target_dir}. Re-run with --overwrite to replace it."
        )
    runtime_paths = verify_runtime_paths({
        "dataset_dir": target_dir,
        "research_log": research_log_path(output_dir),
        "duckdb_spill_dir": duckdb_spill_dir(output_dir),
    })

    research_log = open_research_log(output_dir)
    try:
        research_log.info(f"Build started: {DATASET_ID} v{DATASET_VERSION}, feature spec {FEATURE_SPEC_VERSION}")
        manifest = _build(mimic_dir, output_dir, target_dir, runtime_paths, research_log)
        counts = manifest["aggregate_counts"]
        research_log.info(f"Cohort flow: {json.dumps(counts['exclusions'])}")
        research_log.info(f"Ambiguous outcome reasons: {json.dumps(counts['ambiguous_outcome_reasons'])}")
        research_log.info(f"Outcome anomalies (labeled positive, flagged): {json.dumps(counts['outcome_anomalies'])}")
        research_log.info(
            f"Eligible stays {counts['cohort_size']}, positive labels {counts['positive_labels']}, "
            f"prevalence {counts['outcome_prevalence']}"
        )
        research_log.info(f"Build finished: {target_dir}")
        return manifest
    except Exception as e:
        research_log.error(f"Build failed: {type(e).__name__}")
        raise
    finally:
        close_research_log(research_log)
        shutil.rmtree(duckdb_spill_dir(output_dir), ignore_errors=True)


def _build(mimic_dir, output_dir, target_dir, runtime_paths, research_log) -> Dict[str, object]:
    with MimicIVReader(mimic_dir, temp_directory=duckdb_spill_dir(output_dir)) as reader:
        validated_items, mapping_warnings = validate_item_dictionaries(reader.load_d_items(), reader.load_d_labitems())
        for warning in mapping_warnings:
            research_log.warning(warning)
        cohort, exclusion_counts, ambiguity_counts, outcome_anomalies = build_cohort(
            reader.load_patients(), reader.load_admissions(), reader.load_icustays()
        )
        chart_ids = [i for i, (_, item) in validated_items.items() if item.source_table == "chartevents"]
        lab_ids = [i for i, (_, item) in validated_items.items() if item.source_table == "labevents"]
        events = reader.extract_window_events(cohort, chart_ids, lab_ids)
        source_description = reader.layout.describe()

    features, diagnostics = extract_features(events, cohort, validated_items)

    dataset = pd.DataFrame({
        "src_subject_id": cohort["subject_id"],
        "src_hadm_id": cohort["hadm_id"],
        "src_stay_id": cohort["stay_id"],
        "index_time": cohort["index_time"],
        LABEL_COLUMN: cohort["label"].astype("int8"),
        ANOMALY_COLUMN: cohort[ANOMALY_COLUMN].astype("int8"),
    })
    dataset = dataset.merge(features.rename(columns={"stay_id": "src_stay_id"}), on="src_stay_id", how="left")
    dataset["dataset_version"] = DATASET_VERSION
    dataset["feature_spec_version"] = FEATURE_SPEC_VERSION
    dataset = dataset.sort_values("src_stay_id", kind="mergesort").reset_index(drop=True)

    cohort_size = len(dataset)
    positives = int(dataset[LABEL_COLUMN].sum())
    coverage = {
        var.name: round(float((dataset[f"{var.name}_count"] > 0).mean()), 4) if cohort_size else 0.0
        for var in VARIABLES
    }

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "task": TASK_DEFINITION,
        "source": source_description,
        "configuration": {
            "mimic_dir": str(mimic_dir),
            "output_dir": str(output_dir),
            "runtime_paths": runtime_paths,
            "item_mapping": {
                str(itemid): {"variable": var.name, "source_table": item.source_table, "source_unit": item.source_unit}
                for itemid, (var, item) in sorted(validated_items.items())
            },
            "mapping_warnings": mapping_warnings,
        },
        "columns": {
            "identifiers": ID_COLUMNS,
            "index_time": "index_time",
            "label": LABEL_COLUMN,
            "label_quality": {
                "columns": [ANOMALY_COLUMN],
                "note": "Outcome-derived label metadata for sensitivity analyses. Never use as a model feature.",
            },
            "features": feature_columns(),
            "versions": ["dataset_version", "feature_spec_version"],
        },
        "aggregate_counts": {
            "exclusions": dict(exclusion_counts),
            "ambiguous_outcome_reasons": ambiguity_counts,
            "outcome_anomalies": outcome_anomalies,
            "cohort_size": cohort_size,
            "positive_labels": positives,
            "outcome_prevalence": round(positives / cohort_size, 4) if cohort_size else None,
            "variable_coverage_fraction_of_stays": coverage,
            "observation_quality": diagnostics,
        },
        "row_count": cohort_size,
        "content_sha256": content_sha256(dataset),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "duckdb": __import__("duckdb").__version__,
            "pyarrow": __import__("pyarrow").__version__,
        },
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "notes": "Identifier columns are source identifiers for local joins and patient-level "
                 "splitting only. Never share this dataset or publish row-level content.",
    }

    target_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(target_dir / "dataset.parquet", lambda p: dataset.to_parquet(p, index=False))
    _atomic_write(
        target_dir / "manifest.json",
        lambda p: p.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8"),
    )
    return manifest


def _atomic_write(path: Path, writer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def format_summary(manifest: Dict[str, object], target_dir: Path) -> List[str]:
    """Human-readable aggregate summary (no row-level information)."""
    counts = manifest["aggregate_counts"]
    lines = [
        f"MediHaven Task 1 dataset ({manifest['dataset_id']} v{manifest['dataset_version']}, "
        f"task v{manifest['task']['task_version']})",
        f"Source release: {manifest['source']['release']} | Feature spec: {manifest['feature_spec_version']}",
        "Cohort flow (stays excluded at each step):",
    ]
    for step, n in counts["exclusions"].items():
        lines.append(f"  {step:<36} {n}")
    reasons = {k: v for k, v in counts["ambiguous_outcome_reasons"].items() if v}
    if reasons:
        lines.append("Ambiguous outcome reasons: " + ", ".join(f"{k}={v}" for k, v in reasons.items()))
    anomalies = counts["outcome_anomalies"]
    gaps = ", ".join(f"{k}: {v}" for k, v in anomalies["deathtime_after_discharge_gap"].items() if v)
    lines.append(
        f"Outcome anomalies (labeled positive, flagged): deathtime after dischtime = "
        f"{anomalies['positive_deathtime_after_discharge']}" + (f" (gap {gaps})" if gaps else "")
    )
    prevalence = counts["outcome_prevalence"]
    lines.append(
        f"Eligible stays: {counts['cohort_size']} | positive labels: {counts['positive_labels']} | "
        f"prevalence: {'n/a' if prevalence is None else f'{prevalence:.1%}'}"
    )
    lines.append("Variable coverage (fraction of stays with >= 1 valid measurement):")
    for name, fraction in counts["variable_coverage_fraction_of_stays"].items():
        lines.append(f"  {name:<12} {fraction:.0%}")
    for warning in manifest["configuration"]["mapping_warnings"]:
        lines.append(f"WARNING: {warning}")
    paths = manifest["configuration"]["runtime_paths"]
    lines.append(f"Dataset written to: {paths['dataset_dir']}")
    lines.append(f"Research log:       {paths['research_log']}")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the MediHaven Task 1 dataset from local MIMIC-IV files.")
    parser.add_argument("--mimic-dir", type=Path, default=None, help="Folder containing hosp/ and icu/ (default: MIMIC_IV_DIR)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Research output folder (default: MIMIC_OUTPUT_DIR)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing dataset of the same version")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks for unexpected errors (local use only)")
    args = parser.parse_args(argv)

    try:
        manifest = build_dataset(args.mimic_dir, args.output_dir, overwrite=args.overwrite)
    except (BuildConfigError, UnsafeResearchPathError, MimicDataError, MappingError, FileExistsError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # unexpected: messages could echo data values, so hide them by default
        print(f"ERROR: unexpected {type(e).__name__} while building the dataset. "
              f"Re-run with --debug locally to see the traceback.", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1

    output_dir = Path(manifest["configuration"]["output_dir"])
    print("\n".join(format_summary(manifest, dataset_dir(output_dir))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
