"""Verification and loading of Phase 2 model artifacts for research-mode inference.

Fail-closed rules, checked in order before any model is used:
1. The research directory passes the repository/OneDrive safety check.
2. run_manifest.json is readable JSON of the expected shape, matches the optional
   RESEARCH_ARTIFACT_MANIFEST_SHA256 pin, and names the expected run, dataset version (1.2.0),
   task (version 3), feature specification and the installed scikit-learn version.
3. The run's dataset content hash matches the Phase 1 dataset manifest (traceability).
4. Every required output file is listed in the manifest and stays inside the run folder. Each file
   is read ONCE; its SHA-256 is checked on those bytes, and only those exact bytes are parsed or
   deserialized (no second read, so the file cannot be swapped between check and load). joblib uses
   pickle, which can execute code, so only verified bytes from the local research folder are loaded.
5. Every model bundle carries the expected feature list (no forbidden or unverifiable features), the
   same dataset/task versions and content hash, and a fitted pipeline whose inputs are those features.
6. metrics.json has the structure the research endpoints read.

Any failure raises ArtifactVerificationError with a reason code; callers keep inference disabled.
Hashes detect corruption and accidental mismatches. Authenticity against someone who can rewrite the
research folder requires the manifest pin, which lives outside that folder.
"""

import hashlib
import io
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import sklearn

from src.datasources.mimic.mappings import FEATURE_SPEC_VERSION, feature_columns
from src.tasks.mortality_icu_24h import TASK_ID
from src.training.dataset import EXPECTED_DATASET_VERSION, EXPECTED_TASK_VERSION, default_dataset_dir
from src.training.features import ForbiddenFeatureError, select_features
from src.training.models import MODEL_ORDER
from src.training.train_mimic_t1 import RUN_ID, RUN_VERSION, output_dir_for
from src.utils.research_paths import UnsafeResearchPathError, is_within, require_safe_research_path

REQUIRED_FILES = ("metrics.json", "feature_config.json") + tuple(f"models/{m}.joblib" for m in MODEL_ORDER)
BUNDLE_KEYS = ("pipeline", "features", "dataset_version", "task_version", "dataset_content_sha256", "run_id", "run_version")
ANALYSES = ("primary", "sensitivity_evaluation_only", "sensitivity_retrained")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ArtifactVerificationError(RuntimeError):
    """Raised when research artifacts are missing, inconsistent or tampered. Messages hold no paths or data."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class VerifiedArtifacts:
    run_manifest: Dict[str, object]
    metrics: Dict[str, object]
    features: List[str]
    models: Dict[str, object] = field(default_factory=dict)  # name -> fitted pipeline
    cohort: Optional[Dict[str, object]] = None  # aggregate Phase 1 cohort counts, for display only


def cohort_summary(phase1_manifest: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Aggregate cohort counts from the Phase 1 dataset manifest (display only; None if absent/malformed)."""
    counts = phase1_manifest.get("aggregate_counts")
    if not isinstance(counts, dict) or not isinstance(counts.get("exclusions"), dict):
        return None
    numeric = lambda d: isinstance(d, dict) and all(isinstance(v, (int, float)) for v in d.values())
    summary = {
        "source_release": (phase1_manifest.get("source") or {}).get("release"),
        "exclusions": counts["exclusions"] if numeric(counts["exclusions"]) else None,
        "ambiguous_outcome_reasons": counts.get("ambiguous_outcome_reasons") if numeric(counts.get("ambiguous_outcome_reasons")) else {},
        "outcome_anomalies": counts.get("outcome_anomalies") if isinstance(counts.get("outcome_anomalies"), dict) else {},
        "variable_coverage": counts.get("variable_coverage_fraction_of_stays") if numeric(counts.get("variable_coverage_fraction_of_stays")) else {},
        "observation_quality": {},
    }
    quality = counts.get("observation_quality")
    if isinstance(quality, dict) and all(numeric(v) for v in quality.values()):
        summary["observation_quality"] = quality
    return summary if summary["exclusions"] is not None else None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path, code: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as e:
        raise ArtifactVerificationError(code, f"Could not read {path.name} ({type(e).__name__}).") from e


def _parse_json_object(data: bytes, code: str, name: str) -> Dict[str, object]:
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ArtifactVerificationError(code, f"{name} is not valid JSON.") from e
    if not isinstance(parsed, dict):
        raise ArtifactVerificationError(code, f"{name} must contain a JSON object.")
    return parsed


def verify_and_load(research_dir: Optional[Path], manifest_pin: str = "") -> VerifiedArtifacts:
    if research_dir is None:
        raise ArtifactVerificationError("NOT_CONFIGURED", "MIMIC_OUTPUT_DIR is not configured.")
    try:
        research_dir = require_safe_research_path(research_dir, "the research output directory")
    except UnsafeResearchPathError as e:
        raise ArtifactVerificationError("UNSAFE_LOCATION",
                                        "The research directory is inside the repository or OneDrive.") from e

    run_dir = output_dir_for(research_dir)
    manifest_file = run_dir / "run_manifest.json"
    if not manifest_file.is_file():
        raise ArtifactVerificationError("ARTIFACTS_MISSING",
                                        "Phase 2 artifacts not found. Run: python -m src.training.train_mimic_t1")
    manifest_bytes = _read_bytes(manifest_file, "MANIFEST_UNREADABLE")
    if manifest_pin and _sha256(manifest_bytes) != manifest_pin:
        raise ArtifactVerificationError("MANIFEST_PIN_MISMATCH",
                                        "run_manifest.json does not match RESEARCH_ARTIFACT_MANIFEST_SHA256.")
    manifest = _parse_json_object(manifest_bytes, "MANIFEST_INVALID", "run_manifest.json")

    dataset, software = manifest.get("dataset"), manifest.get("software")
    if not isinstance(dataset, dict) or not isinstance(software, dict):
        raise ArtifactVerificationError("MANIFEST_INVALID", "run_manifest.json lacks its dataset/software sections.")
    expected = {
        "run_id": (manifest.get("run_id"), RUN_ID),
        "run_version": (manifest.get("run_version"), RUN_VERSION),
        "dataset.dataset_version": (dataset.get("dataset_version"), EXPECTED_DATASET_VERSION),
        "dataset.task_id": (dataset.get("task_id"), TASK_ID),
        "dataset.task_version": (dataset.get("task_version"), EXPECTED_TASK_VERSION),
        "dataset.feature_spec_version": (dataset.get("feature_spec_version"), FEATURE_SPEC_VERSION),
    }
    for key, (actual, value) in expected.items():
        if actual != value:
            raise ArtifactVerificationError("VERSION_MISMATCH", f"Artifact {key} is '{actual}', expected '{value}'.")
    if software.get("scikit-learn") != sklearn.__version__:
        raise ArtifactVerificationError(
            "SKLEARN_VERSION_MISMATCH",
            f"Artifacts were built with scikit-learn {software.get('scikit-learn')}, but {sklearn.__version__} is "
            "installed. Install the recorded version or retrain before enabling research inference.")

    phase1_manifest_file = default_dataset_dir(research_dir) / "manifest.json"
    if not phase1_manifest_file.is_file():
        raise ArtifactVerificationError("DATASET_MANIFEST_MISSING", "The Phase 1 dataset manifest is missing.")
    phase1_manifest = _parse_json_object(_read_bytes(phase1_manifest_file, "DATASET_MANIFEST_UNREADABLE"),
                                         "DATASET_MANIFEST_INVALID", "the Phase 1 dataset manifest")
    if phase1_manifest.get("content_sha256") != dataset.get("dataset_content_sha256"):
        raise ArtifactVerificationError("DATASET_MISMATCH",
                                        "The models were trained on a different dataset build than the current one.")

    hashes = manifest.get("output_files_sha256")
    if not isinstance(hashes, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and SHA256_PATTERN.match(v) for k, v in hashes.items()):
        raise ArtifactVerificationError("MANIFEST_INVALID", "output_files_sha256 must map file names to SHA-256 hashes.")
    for required in REQUIRED_FILES:
        if required not in hashes:
            raise ArtifactVerificationError("MANIFEST_INCOMPLETE", f"{required} is not listed in run_manifest.json.")

    verified: Dict[str, bytes] = {}
    for relative, digest in hashes.items():
        path = run_dir / relative
        if not is_within(path, run_dir):
            raise ArtifactVerificationError("UNSAFE_MANIFEST_ENTRY", "run_manifest.json lists a file outside the run folder.")
        if not path.is_file():
            raise ArtifactVerificationError("ARTIFACT_MISSING", f"{relative} is missing.")
        data = _read_bytes(path, "ARTIFACT_UNREADABLE")  # read once; only these bytes are used below
        if _sha256(data) != digest:
            raise ArtifactVerificationError("ARTIFACT_HASH_MISMATCH", f"{relative} does not match its recorded hash.")
        if relative in REQUIRED_FILES:
            verified[relative] = data

    features = feature_columns()
    try:
        permitted, excluded = select_features(features)
    except ForbiddenFeatureError as e:  # pragma: no cover - feature spec itself is guarded by tests
        raise ArtifactVerificationError("FORBIDDEN_FEATURE", "The feature specification contains forbidden columns.") from e
    if excluded or permitted != features:
        raise ArtifactVerificationError("FEATURE_SCHEMA_MISMATCH", "Feature provenance could not be established.")
    feature_config = _parse_json_object(verified["feature_config.json"], "FEATURE_CONFIG_INVALID", "feature_config.json")
    config_features = feature_config.get("features")
    if not isinstance(config_features, list) or not all(isinstance(f, dict) for f in config_features):
        raise ArtifactVerificationError("FEATURE_CONFIG_INVALID", "feature_config.json has no valid feature list.")
    if [f.get("name") for f in config_features] != features:
        raise ArtifactVerificationError("FEATURE_SCHEMA_MISMATCH", "feature_config.json does not match the feature spec.")

    models = {}
    for name in MODEL_ORDER:
        bundle = deserialize_verified(name, verified[f"models/{name}.joblib"])
        models[name] = _check_bundle(name, bundle, features, dataset)

    metrics = _parse_json_object(verified["metrics.json"], "METRICS_INVALID", "metrics.json")
    _check_metrics(metrics)
    return VerifiedArtifacts(run_manifest=manifest, metrics=metrics, features=features, models=models,
                             cohort=cohort_summary(phase1_manifest))


def deserialize_verified(name: str, data: bytes):
    """Deserializes already-verified bytes; any failure or version warning becomes a verification error."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            bundle = joblib.load(io.BytesIO(data))
        except Exception as e:
            raise ArtifactVerificationError("BUNDLE_UNLOADABLE",
                                            f"Model {name} could not be deserialized ({type(e).__name__}).") from e
    if any(w.category.__name__ == "InconsistentVersionWarning" for w in caught):
        raise ArtifactVerificationError("SKLEARN_VERSION_MISMATCH",
                                        f"Model {name} was pickled with a different scikit-learn version.")
    return bundle


def _check_bundle(name: str, bundle, features: List[str], dataset: Dict[str, object]):
    if not isinstance(bundle, dict) or any(key not in bundle for key in BUNDLE_KEYS):
        raise ArtifactVerificationError("BUNDLE_INVALID", f"Model bundle {name} is incomplete.")
    if bundle["features"] != features:
        raise ArtifactVerificationError("FEATURE_SCHEMA_MISMATCH", f"Model {name} uses a different feature list.")
    try:
        select_features(bundle["features"])
    except ForbiddenFeatureError as e:
        raise ArtifactVerificationError("FORBIDDEN_FEATURE", f"Model {name} declares a forbidden input.") from e
    if (bundle["dataset_version"], bundle["task_version"], bundle["dataset_content_sha256"]) != (
            dataset.get("dataset_version"), dataset.get("task_version"), dataset.get("dataset_content_sha256")):
        raise ArtifactVerificationError("VERSION_MISMATCH", f"Model {name} was trained on different data.")
    if (bundle["run_id"], bundle["run_version"]) != (RUN_ID, RUN_VERSION):
        raise ArtifactVerificationError("VERSION_MISMATCH", f"Model {name} belongs to a different run.")
    pipeline = bundle["pipeline"]
    names_in = getattr(pipeline, "feature_names_in_", None)
    if name != "dummy_prior" and (names_in is None or list(names_in) != features):
        raise ArtifactVerificationError("FEATURE_SCHEMA_MISMATCH", f"Model {name} was fitted on different inputs.")
    if not callable(getattr(pipeline, "predict_proba", None)) or list(getattr(pipeline, "classes_", [])) != [0, 1]:
        raise ArtifactVerificationError("BUNDLE_INVALID", f"Model {name} is not a fitted binary classifier.")
    return pipeline


def _check_metrics(metrics: Dict[str, object]) -> None:
    """Confirms the structure that the model card reads, so requests cannot fail on malformed metrics."""
    def bad(reason):
        raise ArtifactVerificationError("METRICS_INVALID", f"metrics.json {reason}.")

    analyses = metrics.get("analyses")
    if not isinstance(metrics.get("definitions"), dict) or not isinstance(analyses, dict):
        bad("lacks definitions/analyses")
    for key in ANALYSES:
        analysis = analyses.get(key)
        if not isinstance(analysis, dict) or not isinstance(analysis.get("population"), dict) \
                or not isinstance(analysis.get("evaluation"), dict):
            bad(f"has an invalid '{key}' analysis")
        if key == "primary" or "models" in analysis:
            models = analysis.get("models")
            if not isinstance(models, dict) or any(
                    not isinstance(models.get(m), dict) or not isinstance(models[m].get("pooled_out_of_fold"), dict)
                    or not isinstance(models[m].get("bootstrap_ci_95"), dict) for m in MODEL_ORDER):
                bad(f"has incomplete model results in '{key}'")
