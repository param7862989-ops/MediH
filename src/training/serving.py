"""Research-mode inference service over verified Phase 2 artifacts.

Serves aggregate model information, "what-if" risk estimates for feature values supplied by the
caller, and (via stays.py) a stay-level triage view of the research cohort for this machine only.
MIMIC identifiers never reach an API response, log line or browser: stays are numbered 1..N.
What-if inputs follow the contract in whatif.py: identifiers, outcome/anomaly columns and other
forbidden names are rejected, unmeasured variables are encoded as in training, and contradictory or
impossible inputs are refused.
"""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.datasources.mimic.mappings import VARIABLES, feature_columns
from src.training.artifacts import ArtifactVerificationError, VerifiedArtifacts, verify_and_load
from src.training.features import build_feature_matrix
from src.training.models import MODEL_ORDER
from src.training.stays import TIERS, LazyStayView, StayViewError  # noqa: F401 (StayViewError re-exported)
from src.training.train_mimic_t1 import DISCLAIMER
from src.training.whatif import PredictionInputError, build_whatif_row  # noqa: F401 (re-exported)
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("medihaven.research.serving")

PREDICTION_POINT = "ICU admission time + 24 hours (features summarize only the first 24 ICU hours)"
TASK_DESCRIPTION = "In-hospital mortality after the first 24 hours of an adult ICU stay (MIMIC-IV Demo, research)"
PIN_WARNING = ("RESEARCH_ARTIFACT_MANIFEST_SHA256 is not set: artifact hashes detect corruption and mismatches "
               "but not deliberate replacement by someone who can write to the research folder.")


class ResearchModelService:
    """Holds verified artifacts, or the reason verification failed (fail closed)."""

    def __init__(self, artifacts: Optional[VerifiedArtifacts], failure: Optional[ArtifactVerificationError],
                 manifest_pinned: bool = False, stays: Optional[LazyStayView] = None):
        self.artifacts = artifacts
        self.failure = failure
        self.manifest_pinned = manifest_pinned
        self.stays = stays

    @classmethod
    def from_config(cls) -> "ResearchModelService":
        pinned = bool(Config.RESEARCH_ARTIFACT_MANIFEST_SHA256)
        if not pinned:
            logger.warning(f"SECURITY: {PIN_WARNING}")
        try:
            artifacts = verify_and_load(Config.MIMIC_OUTPUT_DIR, Config.RESEARCH_ARTIFACT_MANIFEST_SHA256)
            logger.info("Research artifacts verified; research inference enabled.")
            stays = LazyStayView(Config.MIMIC_OUTPUT_DIR, artifacts.features, artifacts.metrics)
            return cls(artifacts, None, pinned, stays)
        except ArtifactVerificationError as e:
            logger.warning(f"Research inference disabled: {e.code}")
            return cls(None, e, pinned)
        except Exception as e:  # any unexpected problem still fails closed, without crashing startup
            logger.warning(f"Research inference disabled: unexpected {type(e).__name__} during verification")
            return cls(None, ArtifactVerificationError(
                "VERIFICATION_ERROR", f"Unexpected {type(e).__name__} while verifying research artifacts."), pinned)

    @property
    def available(self) -> bool:
        return self.artifacts is not None

    def status(self) -> Dict[str, object]:
        if not self.available:
            return {"research_inference_available": False, "reason_code": self.failure.code,
                    "reason": str(self.failure), "manifest_pinned": self.manifest_pinned, "disclaimer": DISCLAIMER}
        manifest = self.artifacts.run_manifest
        return {
            "research_inference_available": True,
            "run": {"run_id": manifest["run_id"], "run_version": manifest["run_version"],
                    "created_at": manifest.get("created_at")},
            "dataset": {k: manifest["dataset"][k] for k in
                        ("dataset_id", "dataset_version", "task_id", "task_version", "feature_spec_version", "source_release")},
            "task": TASK_DESCRIPTION,
            "prediction_point": PREDICTION_POINT,
            "feature_count": len(self.artifacts.features),
            "models": list(MODEL_ORDER),
            "manifest_pinned": self.manifest_pinned,
            "disclaimer": DISCLAIMER,
        }

    def model_card(self) -> Dict[str, object]:
        """Aggregate evaluation results (no patient-level content)."""
        analyses = self.artifacts.metrics["analyses"]

        def summarize(analysis):
            card = {"population": analysis["population"], "evaluation": analysis["evaluation"]}
            if "models" in analysis:
                card["models"] = {}
                for name in MODEL_ORDER:
                    m, ci = analysis["models"][name]["pooled_out_of_fold"], analysis["models"][name]["bootstrap_ci_95"]
                    card["models"][name] = {
                        **{k: m[k] for k in ("auroc", "average_precision", "brier_score", "sensitivity", "specificity",
                                             "precision", "f1")},
                        "confusion_matrix": m["confusion_matrix"], "undefined": m["undefined"],
                        "ci_95": {k: {"low": ci[k]["low"], "high": ci[k]["high"]} for k in ci},
                    }
            return card

        return {
            "disclaimer": DISCLAIMER,
            "task": TASK_DESCRIPTION,
            "prediction_point": PREDICTION_POINT,
            "definitions": self.artifacts.metrics["definitions"],
            "primary": summarize(analyses["primary"]),
            "sensitivity_evaluation_only": summarize(analyses["sensitivity_evaluation_only"]),
            "sensitivity_retrained": summarize(analyses["sensitivity_retrained"]),
        }

    def dashboard(self) -> Dict[str, object]:
        """Everything the research dashboard charts: cohort-level aggregates and model-level weights only."""
        analyses = self.artifacts.metrics["analyses"]
        primary = analyses["primary"]
        population = primary["population"]
        models = {}
        for name in MODEL_ORDER:
            result = primary["models"][name]
            pooled = result["pooled_out_of_fold"]
            models[name] = {
                **{k: pooled[k] for k in ("auroc", "average_precision", "brier_score", "sensitivity",
                                          "specificity", "precision", "f1")},
                "confusion_matrix": pooled["confusion_matrix"],
                "ci_95": {k: {"low": v.get("low"), "high": v.get("high")} for k, v in result["bootstrap_ci_95"].items()},
                "per_fold_auroc": [f.get("auroc") for f in result.get("per_fold", [])],
            }
        candidates = [n for n in MODEL_ORDER if n != "dummy_prior" and models[n]["auroc"] is not None]
        top = max(candidates, key=lambda n: models[n]["auroc"]) if candidates else None

        sensitivity = {}
        for key in ("primary", "sensitivity_evaluation_only", "sensitivity_retrained"):
            analysis = analyses[key]
            sensitivity[key] = {
                "population": analysis["population"],
                "evaluation": analysis["evaluation"],
                "auroc": ({n: analysis["models"][n]["pooled_out_of_fold"]["auroc"] for n in MODEL_ORDER}
                          if "models" in analysis else None),
                "average_precision": ({n: analysis["models"][n]["pooled_out_of_fold"]["average_precision"]
                                       for n in MODEL_ORDER} if "models" in analysis else None),
            }

        return {
            "disclaimer": DISCLAIMER,
            "task": TASK_DESCRIPTION,
            "prediction_point": PREDICTION_POINT,
            "dataset": self.status()["dataset"],
            "manifest_pinned": self.manifest_pinned,
            "kpis": {
                "stays": population["stays"],
                "subjects": population["subjects"],
                "deaths": population["positives"],
                "prevalence": population["prevalence"],
                "features": len(self.artifacts.features),
                "evaluation_status": primary["evaluation"]["status"],
                "top_model": top,
                "top_auroc": models[top]["auroc"] if top else None,
                "top_auroc_ci": models[top]["ci_95"].get("auroc") if top else None,
            },
            "evaluation": primary["evaluation"],
            "folds": primary["split"]["folds"],
            "models": models,
            "sensitivity": sensitivity,
            "cohort": self._cohort_charts(),
            "feature_weights": self._feature_weights(),
            "definitions": self.artifacts.metrics["definitions"],
            "whatif_variables": [
                {"name": v.name, "unit": v.unit, "min": v.plausible_min, "max": v.plausible_max} for v in VARIABLES
            ],
        }

    def _cohort_charts(self) -> Optional[Dict[str, object]]:
        cohort = self.artifacts.cohort
        if not cohort:
            return None
        exclusions = dict(cohort["exclusions"])
        remaining = exclusions.get("icu_stays_in_source")
        flow = [{"step": "ICU stays in source", "excluded": 0, "remaining": remaining}]
        for step, excluded in exclusions.items():
            if step in ("icu_stays_in_source", "cohort_size", "positive_labels", "negative_labels") or remaining is None:
                continue
            remaining -= excluded
            flow.append({"step": step.replace("_", " "), "excluded": excluded, "remaining": remaining})
        quality = cohort["observation_quality"]
        reasons = ("outside_window", "stored_after_index", "non_numeric", "unit_mismatch", "implausible")
        return {
            "source_release": cohort["source_release"],
            "flow": flow,
            "ambiguous_outcome_reasons": {k: v for k, v in cohort["ambiguous_outcome_reasons"].items() if v},
            "outcome_anomalies": cohort["outcome_anomalies"].get("positive_deathtime_after_discharge"),
            "coverage": cohort["variable_coverage"],
            "data_quality": {
                "kept": sum(int(q.get("kept", 0)) for q in quality.values()),
                "dropped": {r: sum(int(q.get(r, 0)) for q in quality.values()) for r in reasons},
            } if quality else None,
        }

    def _feature_weights(self, top_n: int = 10) -> Optional[Dict[str, List[Dict[str, object]]]]:
        """Model-level weights of the final models (no patient data). None if a pipeline has another shape."""
        features = self.artifacts.features

        def kept(pipeline) -> List[str]:
            selector = pipeline.named_steps.get("drop_constant")
            mask = selector.get_support() if selector is not None else np.ones(len(features), dtype=bool)
            return [f for f, keep in zip(features, mask) if keep]

        try:
            forest = self.artifacts.models["random_forest"]
            importance = forest.named_steps["model"].feature_importances_
            forest_top = sorted(zip(kept(forest), importance), key=lambda x: -x[1])[:top_n]
            logistic = self.artifacts.models["logistic_regression"]
            coefficients = logistic.named_steps["model"].coef_[0]
            logistic_top = sorted(zip(kept(logistic), coefficients), key=lambda x: -abs(x[1]))[:top_n]
        except (AttributeError, KeyError, IndexError, ValueError):
            return None
        return {
            "random_forest_importance": [{"feature": f, "weight": round(float(w), 4)} for f, w in forest_top],
            "logistic_regression_coefficient": [{"feature": f, "weight": round(float(w), 4)} for f, w in logistic_top],
        }

    def stay_roster(self) -> Dict[str, object]:
        """All stays with out-of-fold risk scores, tiers and a few latest values (no identifiers).

        Raises StayViewError if the stay view cannot be prepared or verified.
        """
        view = self.stays.get()
        tiers = Counter(row["tier"] for row in view.rows)
        deaths = sum(row["died"] for row in view.rows)
        return {
            "disclaimer": DISCLAIMER,
            "task": TASK_DESCRIPTION,
            "prediction_point": PREDICTION_POINT,
            "scoring": "Each stay is scored by models trained without that patient's stays (out-of-fold). "
                       "The consensus is the mean of the four models.",
            "death_rate": view.death_rate,
            "bands": view.bands,
            "kpis": {
                "stays": len(view.rows),
                "deaths": deaths,
                "tiers": {t: tiers.get(t, 0) for t in TIERS},
                "deaths_in_high_or_critical": sum(r["died"] for r in view.rows if r["tier"] in ("High", "Critical")),
            },
            "stays": view.rows,
        }

    def stay_detail(self, number: int) -> Optional[Dict[str, object]]:
        """One stay's 24-hour summaries, model scores and outcome, or None if the number is unknown."""
        detail = self.stays.get().details.get(number)
        if detail is None:
            return None
        return {"disclaimer": DISCLAIMER, "death_rate": self.stays.get().death_rate, **detail}

    def predict(self, supplied: object) -> Dict[str, object]:
        """Risk estimates for caller-supplied inputs under the whatif.py contract."""
        features = self.artifacts.features
        # The contract is validated on the full feature spec (counts define measured/unmeasured);
        # only the model-input subset is passed to the models
        row, input_summary = build_whatif_row(supplied, feature_columns())
        X = build_feature_matrix(pd.DataFrame([row], columns=features), features)
        probabilities = {
            name: round(float(self.artifacts.models[name].predict_proba(X)[0, 1]), 4) for name in MODEL_ORDER
        }
        logger.info(f"Research what-if estimate served ({len(input_summary['variables_measured'])} of "
                    f"{len(input_summary['variables_measured']) + len(input_summary['variables_not_measured'])} "
                    f"variables measured)")
        manifest = self.artifacts.run_manifest
        return {
            "research_only": True,
            "disclaimer": DISCLAIMER,
            "task": TASK_DESCRIPTION,
            "prediction_point": PREDICTION_POINT,
            "probabilities": probabilities,
            "reference_model": "dummy_prior",
            "evaluation_status": self.artifacts.metrics["analyses"]["primary"]["evaluation"],
            "input_summary": input_summary,
            "limitations": "Manually entered summaries are not validated against any patient record; the estimate "
                           "is only as meaningful as the inputs and the demo-trained model (not clinically validated).",
            "artifact": {"run_id": manifest["run_id"], "run_version": manifest["run_version"],
                         "dataset_version": manifest["dataset"]["dataset_version"],
                         "task_version": manifest["dataset"]["task_version"]},
        }
