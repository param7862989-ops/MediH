"""Clinically Grounded Database Seeding Engine for MediHaven.

Synthesizes 100+ patient records divided across 4 distinct clinical risk cohorts
(Low/Stable, Medium/At-Risk, High/CVD Deterioration, Critical/Sepsis Onset).
Generates realistic 7-14 day longitudinal vitals time series, laboratory biomarkers,
medications, model prediction history, and cryptographic Medical Vault QR tokens.
"""

import json
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jwt
from src.database.db import get_connection, init_db
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("medihaven.seeder")

# Set deterministic random seed for reproducible clinical cohorts
random.seed(42)

# First names and last names for realistic patient names
FIRST_NAMES_M = ["Aarav", "Rohan", "Vikram", "Kabir", "Aditya", "Dev", "Arjun", "Rajesh", "Sanjay", "Anand"]
FIRST_NAMES_F = ["Ananya", "Diya", "Pooja", "Meera", "Neha", "Priya", "Sunita", "Isha", "Kavita", "Rhea"]
LAST_NAMES = ["Sharma", "Patel", "Mehta", "Deshmukh", "Verma", "Joshi", "Iyer", "Nair", "Kulkarni", "Gupta"]


def generate_patient_name(gender: str) -> str:
    first = random.choice(FIRST_NAMES_M) if gender == "M" else random.choice(FIRST_NAMES_F)
    last = random.choice(LAST_NAMES)
    return f"{first} {last}"


def seed_database(num_patients: int = 100) -> None:
    """Generates and inserts clinically realistic patient cohorts and vault records."""
    logger.info(f"Starting database seeding for {num_patients} patients...")

    # Ensure schema exists
    init_db(force_recreate=False)

    now = datetime.now()

    # Cohort breakdown: 35 Low, 30 Medium, 20 High, 15 Critical
    cohort_counts = {
        "Low": 35,
        "Medium": 30,
        "High": 20,
        "Critical": 15,
    }

    total_target = sum(cohort_counts.values())
    patient_id_counter = 1

    with get_connection() as conn:
        cursor = conn.cursor()

        # Check existing count
        cursor.execute("SELECT COUNT(*) FROM patients")
        existing_count = cursor.fetchone()[0]
        if existing_count >= total_target:
            logger.info(f"Database already contains {existing_count} patients. Skipping seeding.")
            print(f"Database already seeded with {existing_count} patients.")
            return

        for risk_tier, count in cohort_counts.items():
            for i in range(count):
                patient_idx = patient_id_counter
                patient_id_counter += 1

                mrn = f"MRN-2026-{patient_idx:04d}"
                gender = random.choice(["M", "F"])
                name = generate_patient_name(gender)

                # Days admitted: 8 to 14 days ago
                admission_days_ago = random.randint(8, 14)
                admission_date = now - timedelta(days=admission_days_ago, hours=random.randint(1, 12))

                # Status and ward based on tier
                if risk_tier == "Critical":
                    status = "ICU"
                    ward = "Intensive Care Unit (ICU)"
                    bed = f"ICU-{random.randint(1, 12):02d}"
                    age = random.randint(45, 82)
                elif risk_tier == "High":
                    status = "Admitted"
                    ward = "Cardiac Care Unit (CCU)"
                    bed = f"CCU-{random.randint(101, 115)}"
                    age = random.randint(52, 85)
                elif risk_tier == "Medium":
                    status = random.choice(["Admitted", "Observation"])
                    ward = random.choice(["Endocrine & Metabolic Ward", "Internal Medicine A"])
                    bed = f"Bed-{random.randint(201, 230)}"
                    age = random.randint(38, 74)
                else:  # Low
                    status = random.choice(["Admitted", "Discharged"])
                    ward = random.choice(["General Ward A", "General Ward B", "Surgical Recovery"])
                    bed = f"Bed-{random.randint(301, 350)}"
                    age = random.randint(19, 58)

                discharge_date = (
                    (admission_date + timedelta(days=random.randint(5, 7)))
                    if status == "Discharged"
                    else None
                )

                cursor.execute(
                    """
                    INSERT INTO patients (mrn, full_name, age, gender, admission_date, discharge_date, status, ward, bed_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (mrn, name, age, gender, admission_date, discharge_date, status, ward, bed),
                )
                patient_id = cursor.lastrowid

                # -------------------------------------------------------------
                # 1. Generate Longitudinal Time-Series Vitals (Every 4-6h)
                # -------------------------------------------------------------
                num_vitals = admission_days_ago * 4  # 4 readings per day (every 6h)
                for v_step in range(num_vitals):
                    v_time = admission_date + timedelta(hours=v_step * 6)
                    progress = v_step / max(num_vitals - 1, 1)  # 0.0 (day 1) to 1.0 (today)

                    # Clinical trajectory modeling
                    if risk_tier == "Critical":
                        # Accelerating deterioration (sepsis progression)
                        # Heart rate climbs 82 -> 126 bpm
                        hr = 82 + (progress * 44) + random.gauss(0, 3)
                        # Blood pressure drops as shock deepens (Sys 124 -> 86)
                        sbp = 124 - (progress * 38) + random.gauss(0, 4)
                        dbp = 78 - (progress * 22) + random.gauss(0, 3)
                        # SpO2 drops 97% -> 88%
                        spo2 = 97.0 - (progress * 8.5) + random.gauss(0, 1.0)
                        # Temperature spikes to 38.9 C
                        temp = 37.1 + (progress * 1.8) + random.gauss(0, 0.3)
                        resp = 16 + int(progress * 12) + random.randint(-1, 2)

                    elif risk_tier == "High":
                        # Deteriorating CVD / Severe Hypertension
                        # SBP climbs 136 -> 168
                        sbp = 136 + (progress * 32) + random.gauss(0, 4)
                        dbp = 86 + (progress * 16) + random.gauss(0, 3)
                        hr = 74 + (progress * 18) + random.gauss(0, 2.5)
                        spo2 = 96.5 - (progress * 3.5) + random.gauss(0, 0.8)
                        temp = 36.8 + random.gauss(0, 0.2)
                        resp = 16 + int(progress * 4) + random.randint(-1, 1)

                    elif risk_tier == "Medium":
                        # Borderline fluctuating vitals
                        sbp = 130 + random.gauss(0, 5)
                        dbp = 84 + random.gauss(0, 4)
                        hr = 78 + random.gauss(0, 4)
                        spo2 = 95.5 + random.gauss(0, 1.0)
                        temp = 36.7 + random.gauss(0, 0.25)
                        resp = 16 + random.randint(-1, 2)

                    else:  # Low / Stable
                        sbp = 118 + random.gauss(0, 3)
                        dbp = 76 + random.gauss(0, 2.5)
                        hr = 70 + random.gauss(0, 3)
                        spo2 = 98.5 + random.gauss(0, 0.5)
                        temp = 36.6 + random.gauss(0, 0.2)
                        resp = 14 + random.randint(-1, 1)

                    # Clamp to realistic clinical bounds
                    cursor.execute(
                        """
                        INSERT INTO vitals (patient_id, recorded_at, heart_rate, blood_pressure_sys, 
                                            blood_pressure_dia, temperature, oxygen_saturation, respiratory_rate)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            patient_id,
                            v_time,
                            round(max(40, min(hr, 200)), 1),
                            round(max(60, min(sbp, 250)), 1),
                            round(max(40, min(dbp, 150)), 1),
                            round(max(34.0, min(temp, 42.0)), 2),
                            round(max(70.0, min(spo2, 100.0)), 1),
                            max(8, min(resp, 45)),
                        ),
                    )

                # -------------------------------------------------------------
                # 2. Generate Laboratory Biomarkers (Admission and Recent)
                # -------------------------------------------------------------
                for lab_step, lab_offset in enumerate([admission_days_ago, 1]):
                    lab_time = now - timedelta(days=lab_offset, hours=2)
                    if risk_tier == "Critical":
                        glucose = 140 + (lab_step * 35) + random.uniform(-10, 15)
                        creatinine = 1.2 + (lab_step * 1.1) + random.uniform(-0.1, 0.2)  # AKI
                        wbc = 9.8 + (lab_step * 8.5) + random.uniform(-0.5, 2.0)  # Sepsis leukocytosis
                        bun = 22 + (lab_step * 24)
                        chol = 175 + random.uniform(-10, 20)
                        hgb = 13.5 - (lab_step * 2.1)
                        plt = 220 - (lab_step * 85)
                    elif risk_tier == "High":
                        glucose = 135 + random.uniform(-15, 25)
                        creatinine = 1.3 + (lab_step * 0.4) + random.uniform(-0.1, 0.1)
                        wbc = 8.2 + random.uniform(-0.5, 1.2)
                        bun = 24 + (lab_step * 8)
                        chol = 238 + random.uniform(-15, 30)
                        hgb = 12.8 + random.uniform(-0.8, 0.8)
                        plt = 240 + random.uniform(-20, 30)
                    elif risk_tier == "Medium":
                        glucose = 128 + random.uniform(-10, 20)  # Impaired fasting glucose
                        creatinine = 1.1 + random.uniform(-0.1, 0.1)
                        wbc = 7.5 + random.uniform(-0.6, 1.0)
                        bun = 18 + random.uniform(-2, 4)
                        chol = 215 + random.uniform(-15, 25)
                        hgb = 13.8 + random.uniform(-0.6, 0.6)
                        plt = 260 + random.uniform(-20, 20)
                    else:
                        glucose = 92 + random.uniform(-8, 12)
                        creatinine = 0.85 + random.uniform(-0.1, 0.1)
                        wbc = 6.4 + random.uniform(-0.8, 0.8)
                        bun = 14 + random.uniform(-2, 3)
                        chol = 172 + random.uniform(-15, 15)
                        hgb = 14.2 + random.uniform(-0.5, 0.5)
                        plt = 280 + random.uniform(-30, 30)

                    cursor.execute(
                        """
                        INSERT INTO lab_results (patient_id, recorded_at, glucose, creatinine, 
                                                cholesterol, wbc_count, bun, hemoglobin, platelets)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            patient_id,
                            lab_time,
                            round(glucose, 1),
                            round(creatinine, 2),
                            round(chol, 1),
                            round(wbc, 2),
                            round(bun, 1),
                            round(hgb, 1),
                            round(plt, 1),
                        ),
                    )

                # -------------------------------------------------------------
                # 3. Medications
                # -------------------------------------------------------------
                if risk_tier == "Critical":
                    meds = [
                        ("Norepinephrine", "0.08 mcg/kg/min", "IV", "Continuous", 1),
                        ("Piperacillin/Tazobactam", "4.5g", "IV", "TID", 1),
                        ("Normal Saline 0.9%", "100 mL/hr", "IV", "Continuous", 1),
                    ]
                elif risk_tier == "High":
                    meds = [
                        ("Amlodipine", "10mg", "Oral", "QD", 1),
                        ("Furosemide", "40mg", "Oral", "BID", 1),
                        ("Atorvastatin", "40mg", "Oral", "QHS", 1),
                        ("Aspirin", "81mg", "Oral", "QD", 1),
                    ]
                elif risk_tier == "Medium":
                    meds = [
                        ("Metformin", "500mg", "Oral", "BID", 1),
                        ("Lisinopril", "10mg", "Oral", "QD", 1),
                    ]
                else:
                    meds = [
                        ("Acetaminophen", "500mg", "Oral", "PRN", 1),
                        ("Multivitamin", "1 tab", "Oral", "QD", 1),
                    ]

                for drug, dosage, route, freq, is_act in meds:
                    cursor.execute(
                        """
                        INSERT INTO medications (patient_id, drug_name, dosage, route, frequency, start_date, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (patient_id, drug, dosage, route, freq, admission_date, is_act),
                    )

                # -------------------------------------------------------------
                # 4. Predictions Log (AI Intelligence Engine)
                # -------------------------------------------------------------
                if risk_tier == "Critical":
                    score = round(random.uniform(0.88, 0.98), 2)
                    rule_path = json.dumps([
                        "respiratory_rate > 24 breaths/min",
                        "oxygen_saturation < 92.0%",
                        "shock_index > 0.95 (Severe Tachycardia)",
                        "wbc_count > 15.0 x10^3/uL -> CRITICAL (Sepsis Onset Flagged)"
                    ])
                    protocol = "Protocol A: Sepsis Bundle (Aggressive IV resuscitation, blood cultures, dual broad-spectrum)"
                    readmit_risk = 0.65
                elif risk_tier == "High":
                    score = round(random.uniform(0.72, 0.85), 2)
                    rule_path = json.dumps([
                        "blood_pressure_sys > 155 mmHg",
                        "creatinine > 1.4 mg/dL",
                        "age > 55 -> HIGH RISK (CVD Progression)"
                    ])
                    protocol = "Protocol C: Cardioprotective Strategy (Diuretic dose escalation + CCB titration)"
                    readmit_risk = 0.42
                elif risk_tier == "Medium":
                    score = round(random.uniform(0.38, 0.62), 2)
                    rule_path = json.dumps([
                        "glucose > 125 mg/dL",
                        "blood_pressure_sys > 130 mmHg -> MEDIUM (Metabolic Strain)"
                    ])
                    protocol = "Protocol B: Glycemic Control & Glycemic Monitoring"
                    readmit_risk = 0.18
                else:
                    score = round(random.uniform(0.05, 0.22), 2)
                    rule_path = json.dumps([
                        "oxygen_saturation >= 98.0%",
                        "heart_rate < 80 bpm",
                        "vitals_trajectory == STABLE -> LOW RISK"
                    ])
                    protocol = "Protocol S: Standard Observational Recovery"
                    readmit_risk = 0.04

                cursor.execute(
                    """
                    INSERT INTO predictions (patient_id, predicted_at, risk_tier, risk_score, 
                                            model_used, decision_tree_path, similar_patient_ids, 
                                            recommended_protocol, readmission_30d_risk)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_id,
                        now - timedelta(hours=1),
                        risk_tier,
                        score,
                        "Ensemble",
                        rule_path,
                        json.dumps([max(1, patient_id - x) for x in range(1, 6)]),
                        protocol,
                        readmit_risk,
                    ),
                )

                # -------------------------------------------------------------
                # 5. Ground Truth Outcomes
                # -------------------------------------------------------------
                if risk_tier == "Critical":
                    dx = "Sepsis secondary to acute bacterial pneumonia"
                    readmitted = True if random.random() < 0.60 else False
                    icu_trans = True
                elif risk_tier == "High":
                    dx = "Hypertensive Emergency with Congestive Heart Failure"
                    readmitted = True if random.random() < 0.40 else False
                    icu_trans = False
                elif risk_tier == "Medium":
                    dx = "Type 2 Diabetes with moderate metabolic decompensation"
                    readmitted = False
                    icu_trans = False
                else:
                    dx = "Uncomplicated acute appendectomy / routine observational recovery"
                    readmitted = False
                    icu_trans = False

                cursor.execute(
                    """
                    INSERT INTO outcomes (patient_id, primary_diagnosis, readmitted_30d, icu_transfer, outcome_date)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (patient_id, dx, readmitted, icu_trans, now),
                )

                # -------------------------------------------------------------
                # 6. Patient Medical Vault Records
                # -------------------------------------------------------------
                vault_items = [
                    (
                        "allergy",
                        "Penicillin G & Derivatives",
                        "Severe anaphylactic reaction documented in 2021. Requires strict avoidance.",
                        json.dumps({"severity": "Severe", "reaction": "Anaphylaxis", "year": 2021}),
                        1,
                    ),
                    (
                        "diagnosis",
                        f"Primary: {dx}",
                        f"Diagnosed during admission by SPIT Department of Computer Engineering Clinical Intelligence.",
                        json.dumps({"icd10": "A41.9" if risk_tier == "Critical" else "I10"}),
                        0,
                    ),
                    (
                        "vaccination",
                        "COVID-19 Updated Booster & Seasonal Influenza",
                        "Administered annually without adverse events.",
                        json.dumps({"booster": "Spikevax 2025", "status": "Current"}),
                        0,
                    ),
                ]

                if risk_tier in ("High", "Critical"):
                    vault_items.append((
                        "surgery",
                        "Cardiac Stent Placement (LAD)",
                        "Percutaneous coronary intervention with drug-eluting stent (2023).",
                        json.dumps({"surgeon": "Dr. V. Patel", "facility": "Lilavati Hospital"}),
                        1,
                    ))

                for cat, title, desc, s_data, is_sens in vault_items:
                    cursor.execute(
                        """
                        INSERT INTO medical_vault (patient_id, category, title, description, 
                                                   structured_data, uploaded_at, is_sensitive)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (patient_id, cat, title, desc, s_data, admission_date, is_sens),
                    )

                # -------------------------------------------------------------
                # 7. Cryptographic QR Access Tokens & Access Logs
                # -------------------------------------------------------------
                # Pass 1: Scoped Pass (Allergies + Medications)
                token_hash_1 = f"tok_{uuid.uuid4().hex[:16]}"
                issued_at = now - timedelta(hours=6)
                expires_at = now + timedelta(hours=18)
                scope_1 = ["allergy", "medication"]

                jwt_payload_1 = {
                    "patient_id": patient_id,
                    "mrn": mrn,
                    "token_hash": token_hash_1,
                    "scope": scope_1,
                    "iat": int(issued_at.timestamp()),
                    "exp": int(expires_at.timestamp()),
                }
                signed_jwt_1 = jwt.encode(jwt_payload_1, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)

                cursor.execute(
                    """
                    INSERT INTO vault_access_tokens (patient_id, token_hash, scope_json, issued_at, 
                                                    expires_at, max_uses, use_count, revoked, qr_payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_id,
                        token_hash_1,
                        json.dumps(scope_1),
                        issued_at,
                        expires_at,
                        5,
                        1,
                        0,
                        signed_jwt_1,
                    ),
                )
                token_id_1 = cursor.lastrowid

                # Log access for token 1
                cursor.execute(
                    """
                    INSERT INTO vault_access_log (token_id, patient_id, accessed_by, access_status, 
                                                accessed_at, ip_address, user_agent, accessed_scope)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        token_id_1,
                        patient_id,
                        "Dr. Nirupam Gupta (Triage Physician)",
                        "GRANTED",
                        now - timedelta(hours=2),
                        "192.168.1.45",
                        "MediHaven Provider Scanner v1.0",
                        json.dumps(scope_1),
                    ),
                )

                # Pass 2: Full History Emergency QR Card
                token_hash_2 = f"tok_{uuid.uuid4().hex[:16]}"
                issued_at_2 = now - timedelta(days=2)
                expires_at_2 = now + timedelta(days=5)
                scope_2 = ["*"]

                jwt_payload_2 = {
                    "patient_id": patient_id,
                    "mrn": mrn,
                    "token_hash": token_hash_2,
                    "scope": scope_2,
                    "iat": int(issued_at_2.timestamp()),
                    "exp": int(expires_at_2.timestamp()),
                }
                signed_jwt_2 = jwt.encode(jwt_payload_2, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)

                cursor.execute(
                    """
                    INSERT INTO vault_access_tokens (patient_id, token_hash, scope_json, issued_at, 
                                                    expires_at, max_uses, use_count, revoked, qr_payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_id,
                        token_hash_2,
                        json.dumps(scope_2),
                        issued_at_2,
                        expires_at_2,
                        None,
                        2,
                        0,
                        signed_jwt_2,
                    ),
                )

        conn.commit()

    logger.info(f"Successfully seeded database with {total_target} clinically coherent patients!")
    print(f"Success: Seeded {total_target} patients across all 4 cohorts with longitudinal time-series vitals.")


if __name__ == "__main__":
    seed_database(100)
