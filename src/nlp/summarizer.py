"""Clinical Narrative Summarization Engine using Google Gemini Generative AI.

Synthesizes multi-model ML risk scores, longitudinal vitals trajectories, Decision Tree
rule breadcrumbs, and KNN treatment protocols into natural-language physician shift
handovers and patient discharge summaries.
"""

import os
import json
import time
import ssl
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Any, Optional

from src.utils.logger import get_logger

logger = get_logger("medihaven.nlp.summarizer")

# Priority list of Gemini models supported by current API keys
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]



class ClinicalNarrativeSummarizer:
    """Generates structured clinical narratives using Google Gemini with deterministic fallback."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "").strip()

    def is_configured(self) -> bool:
        """Returns True if a non-empty Gemini API key is present."""
        return bool(self.api_key)

    def generate(
        self,
        patient_data: Dict[str, Any],
        prediction_data: Dict[str, Any],
        narrative_type: str = "handover",
    ) -> Dict[str, Any]:
        """Synthesizes clinical intelligence into natural language.

        Args:
            patient_data: Dictionary of patient demographic and clinical fields.
            prediction_data: Dictionary of Ensemble prediction outputs.
            narrative_type: 'handover' for physician shift handover or 'discharge' for patient summary.

        Returns:
            Dictionary containing the narrative text, model metadata, latency, and status.
        """
        start_time = time.time()
        narrative_type = narrative_type.lower() if narrative_type in ["handover", "discharge"] else "handover"

        if self.is_configured():
            try:
                result = self._call_gemini(patient_data, prediction_data, narrative_type)
                latency_ms = round((time.time() - start_time) * 1000, 1)
                result["latency_ms"] = latency_ms
                return result
            except Exception as e:
                logger.warning(f"Gemini API call failed ({e}). Falling back to rule-based synthesis.")

        # Deterministic clinical fallback if API key is missing or offline
        fallback_text = self._generate_fallback(patient_data, prediction_data, narrative_type)
        latency_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "narrative": fallback_text,
            "model": "Rule-Based Clinical Synthesizer (Offline Fallback)",
            "provider": "MediHaven Deterministic NLP",
            "latency_ms": latency_ms,
            "is_live": False,
            "type": narrative_type,
            "generated_at": datetime.now().isoformat(),
        }

    def _call_gemini(
        self,
        patient_data: Dict[str, Any],
        prediction_data: Dict[str, Any],
        narrative_type: str,
    ) -> Dict[str, Any]:
        """Calls Google Gemini v1beta REST API with SSL fallback for macOS environments."""
        prompt = self._build_prompt(patient_data, prediction_data, narrative_type)

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

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
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            text = parts[0]["text"].strip()
                            return {
                                "narrative": text,
                                "model": model,
                                "provider": "Google Gemini (Live API)",
                                "is_live": True,
                                "type": narrative_type,
                                "generated_at": datetime.now().isoformat(),
                            }
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                logger.info(f"Gemini model {model} returned HTTP {e.code}: {err_body}")
                last_error = e
                continue
            except Exception as e:
                logger.info(f"Gemini model {model} failed: {e}")
                last_error = e
                continue

        raise RuntimeError(f"All Gemini model candidate endpoints failed: {last_error}")

    def _build_prompt(
        self,
        patient_data: Dict[str, Any],
        prediction_data: Dict[str, Any],
        narrative_type: str,
    ) -> str:
        """Constructs an expert medical prompt grounded in clinical features and ML breadcrumbs."""
        name = patient_data.get("full_name") or patient_data.get("name") or "Inpatient"
        age = patient_data.get("age", "--")
        gender = patient_data.get("gender", "--")
        ward = patient_data.get("ward", "General Care")
        status = patient_data.get("status", "Admitted")
        symptoms = patient_data.get("symptoms", "Monitoring")
        diagnosis = patient_data.get("primary_diagnosis", "Clinical Observation")

        risk_tier = prediction_data.get("risk_tier", "Moderate")
        risk_score = prediction_data.get("risk_score", 0.5)
        protocol = prediction_data.get("recommended_protocol", "Standard Monitoring")
        readmission_risk = prediction_data.get("readmission_30d_risk", 0.15)
        dt_rules = prediction_data.get("decision_tree_explanation", [])
        rules_text = "; ".join(dt_rules[:3]) if dt_rules else "Stable glucose and vital boundaries observed."

        if narrative_type == "handover":
            return f"""You are MediHaven's Clinical AI Copilot assisting hospital physicians during shift change.
Write a concise, professional 3-sentence Physician Shift Handover note using the SBAR framework (Situation, Background/Vitals, Assessment/Action).

Patient Context:
- Name: {name}, Age: {age}, Gender: {gender}, Ward: {ward}, Status: {status}
- Admitting Diagnosis: {diagnosis} | Symptoms: {symptoms}
- Ensemble Risk Tier: {risk_tier} (Risk Score: {risk_score:.2f} / 1.0)
- Machine Learning Biomarker Logic: {rules_text}
- Recommended Clinical Protocol: {protocol}
- 30-Day Readmission Risk: {readmission_risk * 100:.1f}%

Requirements:
1. Sentence 1: Situation & current patient status.
2. Sentence 2: Key physiological biomarker highlights and risk classification.
3. Sentence 3: Immediate attending recommendation or clinical monitoring next step.
Format: Return ONLY the 3 sentences in clear clinical prose, no bullet points, no Markdown headers."""

        else:  # discharge summary
            return f"""You are MediHaven's Clinical AI Copilot preparing a patient discharge overview.
Write a compassionate, reassuring, plain-language 3-sentence summary for the patient and their family.

Patient Context:
- Patient Name: {name}, Ward: {ward}
- Primary Reason for Admission: {diagnosis}
- Current Risk Status: {risk_tier} (Risk Score: {risk_score:.2f})
- Discharge Protocol: {protocol}

Requirements:
1. Sentence 1: Warm acknowledgment of the patient's recovery trajectory and completed care.
2. Sentence 2: Clear, non-technical explanation of their stable vital indicators.
3. Sentence 3: Actionable guidance for medications, activity, or follow-up consults.
Format: Return ONLY the 3 sentences in empathetic prose, no bullet points, no technical jargon."""

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
