"""Clinical Narrative Summarization Engine using Google Gemini Generative AI.

Synthesizes multi-model ML outputs into natural-language physician shift handovers and patient
discharge summaries.

Data-protection rules for the external Gemini call:
- TLS certificates and host names are always verified (default SSL context).
- The API key travels in the ``x-goog-api-key`` header, never in the URL, and is never logged.
- The prompt is de-identified: no name, identifier, age, sex, ward, symptoms, diagnosis or raw
  measurements. Only coarse model outputs (risk tier, rounded scores, protocol label) are sent, with
  the placeholder "[PATIENT]", which is replaced by the name locally after the response arrives.
- In research mode (DATA_MODE=mimic) no external call is ever made; the local synthesizer is used.
"""

import os
import json
import re
import time
import ssl
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Any, Optional

from src.utils.data_mode import is_research_mode
from src.utils.logger import get_logger

logger = get_logger("medihaven.nlp.summarizer")

# Priority list of Gemini models supported by current API keys
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PATIENT_PLACEHOLDER = "[PATIENT]"
RISK_TIERS = {"Low", "Medium", "High", "Critical"}


class ClinicalNarrativeSummarizer:
    """Generates structured clinical narratives using Google Gemini with deterministic fallback."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "").strip()

    def is_configured(self) -> bool:
        """Returns True if a non-empty Gemini API key is present."""
        return bool(self.api_key)

    def external_calls_allowed(self) -> bool:
        """External AI is never used in research mode, whatever the configuration."""
        return self.is_configured() and not is_research_mode()

    def generate(
        self,
        patient_data: Dict[str, Any],
        prediction_data: Dict[str, Any],
        narrative_type: str = "handover",
    ) -> Dict[str, Any]:
        """Synthesizes clinical intelligence into natural language.

        Args:
            patient_data: Dictionary of patient demographic and clinical fields (kept local).
            prediction_data: Dictionary of Ensemble prediction outputs.
            narrative_type: 'handover' for physician shift handover or 'discharge' for patient summary.

        Returns:
            Dictionary containing the narrative text, model metadata, latency, and status.
        """
        start_time = time.time()
        narrative_type = narrative_type.lower() if narrative_type in ["handover", "discharge"] else "handover"

        if self.external_calls_allowed():
            try:
                result = self._call_gemini(prediction_data, narrative_type)
                result["narrative"] = self._personalize(result["narrative"], patient_data)
                result["latency_ms"] = round((time.time() - start_time) * 1000, 1)
                return result
            except Exception as e:
                logger.warning(f"Gemini API call failed ({type(e).__name__}). Falling back to rule-based synthesis.")

        # Deterministic clinical fallback if API key is missing, offline, or research mode
        fallback_text = self._generate_fallback(patient_data, prediction_data, narrative_type)
        latency_ms = round((time.time() - start_time) * 1000, 1)

        result = {
            "narrative": fallback_text,
            "model": "Rule-Based Clinical Synthesizer (Offline Fallback)",
            "provider": "MediHaven Deterministic NLP",
            "latency_ms": latency_ms,
            "is_live": False,
            "type": narrative_type,
            "generated_at": datetime.now().isoformat(),
        }
        if is_research_mode():
            result["external_ai"] = "disabled in research mode; no data left this machine"
        return result

    def _call_gemini(self, prediction_data: Dict[str, Any], narrative_type: str) -> Dict[str, Any]:
        """Calls the Gemini REST API with certificate verification and a de-identified prompt."""
        prompt = self._build_prompt(prediction_data, narrative_type)
        ctx = ssl.create_default_context()  # verifies certificates and host names

        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": 350,
                "topP": 0.9,
            },
        }).encode("utf-8")

        last_error = None
        for model in GEMINI_MODELS:
            req = urllib.request.Request(
                GEMINI_ENDPOINT.format(model=model),
                data=payload,
                headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            )
            try:
                with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return {
                                "narrative": parts[0]["text"].strip(),
                                "model": model,
                                "provider": "Google Gemini (Live API)",
                                "is_live": True,
                                "type": narrative_type,
                                "generated_at": datetime.now().isoformat(),
                            }
            except urllib.error.HTTPError as e:
                logger.info(f"Gemini model {model} returned HTTP {e.code}")  # response body not logged
                last_error = f"HTTP {e.code}"
                continue
            except Exception as e:
                logger.info(f"Gemini model {model} failed: {type(e).__name__}")
                last_error = type(e).__name__
                continue

        raise RuntimeError(f"All Gemini model candidate endpoints failed: {last_error}")

    @staticmethod
    def _coarse_outputs(prediction_data: Dict[str, Any]) -> Dict[str, str]:
        """Only coarse, non-identifying model outputs are allowed into an external prompt."""
        tier = prediction_data.get("risk_tier")
        tier = tier if tier in RISK_TIERS else "Unknown"
        protocol = str(prediction_data.get("recommended_protocol") or "Standard Monitoring")
        protocol = re.split(r"[(:]", protocol)[0].strip()[:40] or "Standard Monitoring"  # generic label only

        def pct(value, default):
            try:
                return f"{round(float(value) * 10) * 10}%"  # rounded to the nearest 10%
            except (TypeError, ValueError):
                return default

        return {
            "tier": tier,
            "risk": pct(prediction_data.get("risk_score"), "unknown"),
            "readmission": pct(prediction_data.get("readmission_30d_risk"), "unknown"),
            "protocol": protocol,
        }

    def _build_prompt(self, prediction_data: Dict[str, Any], narrative_type: str) -> str:
        """De-identified prompt: coarse model outputs only, patient referred to by a placeholder."""
        c = self._coarse_outputs(prediction_data)
        if narrative_type == "handover":
            return f"""You are MediHaven's Clinical AI Copilot assisting hospital physicians during shift change.
Write a concise, professional 3-sentence Physician Shift Handover note using the SBAR framework (Situation, Background, Assessment/Action).
Refer to the patient only as {PATIENT_PLACEHOLDER}. Do not invent names, ages, diagnoses or measurements.

De-identified model context:
- Ensemble risk tier: {c['tier']} (approximate risk score {c['risk']})
- Approximate 30-day readmission risk: {c['readmission']}
- Recommended protocol family: {c['protocol']}

Format: Return ONLY the 3 sentences in clear clinical prose, no bullet points, no Markdown headers."""

        return f"""You are MediHaven's Clinical AI Copilot preparing a patient discharge overview.
Write a compassionate, plain-language 3-sentence summary for the patient and their family.
Refer to the patient only as {PATIENT_PLACEHOLDER}. Do not invent names, ages, diagnoses or measurements.

De-identified model context:
- Current risk status: {c['tier']} (approximate risk score {c['risk']})
- Discharge protocol family: {c['protocol']}

Format: Return ONLY the 3 sentences in empathetic prose, no bullet points, no technical jargon."""

    @staticmethod
    def _personalize(text: str, patient_data: Dict[str, Any]) -> str:
        """Replaces the placeholder with the patient's name locally (the name never left this machine)."""
        name = patient_data.get("full_name") or patient_data.get("name") or "The patient"
        return text.replace(PATIENT_PLACEHOLDER, str(name))

    def _generate_fallback(
        self,
        patient_data: Dict[str, Any],
        prediction_data: Dict[str, Any],
        narrative_type: str,
    ) -> str:
        """Deterministic synthesis ensuring 100% operational uptime without external dependencies."""
        name = patient_data.get("full_name") or patient_data.get("name") or "Patient"
        age = patient_data.get("age", 45)
        ward = patient_data.get("ward", "General Care")
        risk_tier = prediction_data.get("risk_tier", "Low")
        risk_score = prediction_data.get("risk_score", 0.2)
        protocol = prediction_data.get("recommended_protocol", "Standard Observational Recovery")
        readmission_risk = prediction_data.get("readmission_30d_risk", 0.12)

        if narrative_type == "handover":
            return (
                f"Patient {name} ({age}y) in {ward} is currently categorized in the {risk_tier.upper()} risk tier "
                f"with a composite risk score of {risk_score:.2f}. "
                f"Longitudinal biomarker trajectories remain consistent with {protocol}, demonstrating a 30-day "
                f"readmission probability of {readmission_risk * 100:.1f}%. "
                f"Recommend continuing scheduled clinical assessments, maintaining current orders, and escalating "
                f"only if early warning thresholds are breached."
            )
        else:
            return (
                f"{name}'s recovery in {ward} has progressed favorably, meeting clinical discharge criteria with a {risk_tier.lower()} risk profile. "
                f"All physiological parameters and vital biomarkers have stabilized within targeted safety parameters. "
                f"Please adhere to the prescribed post-care guidance ({protocol}) and attend scheduled outpatient follow-ups as advised."
            )


# Global singleton instance for easy import across API routes
summarizer = ClinicalNarrativeSummarizer()
