"""Research-mode inference service over verified Phase 2 artifacts.

Serves only aggregate model information and "what-if" risk estimates for feature values supplied by
the caller. It never reads the patient-level research dataset, so no MIMIC record or identifier can
reach an API response, log line or browser. Inputs follow the contract in whatif.py: identifiers,
outcome/anomaly columns and other forbidden names are rejected, unmeasured variables are encoded as
in training, and contradictory or impossible inputs are refused.
"""

from typing import Dict, Optional

import pandas as pd

from src.training.artifacts import ArtifactVerificationError, VerifiedArtifacts, verify_and_load
from src.training.features import build_feature_matrix
from src.training.models import MODEL_ORDER
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
                 manifest_pinned: bool = False):
        self.artifacts = artifacts
        self.failure = failure
        self.manifest_pinned = manifest_pinned

    @classmethod
    def from_config(cls) -> "ResearchModelService":
        pinned = bool(Config.RESEARCH_ARTIFACT_MANIFEST_SHA256)
        if not pinned:
            logger.warning(f"SECURITY: {PIN_WARNING}")
        try:
            artifacts = verify_and_load(Config.MIMIC_OUTPUT_DIR, Config.RESEARCH_ARTIFACT_MANIFEST_SHA256)
            logger.info("Research artifacts verified; research inference enabled.")
            return cls(artifacts, None, pinned)
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

    def predict(self, supplied: object) -> Dict[str, object]:
        """Risk estimates for caller-supplied inputs under the whatif.py contract."""
        features = self.artifacts.features
        row, input_summary = build_whatif_row(supplied, features)
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
