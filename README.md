# MediHaven

**Integrated Multimodal Clinical Intelligence System for Patient Risk Assessment and Treatment Optimization**

TE Sem V Mini Project I — Phase 1: Problem Definition
Department of Computer Engineering, SPIT | Academic Year 2026–27 (Odd Semester)

**Faculty Guide:** Prof. Swapnali Kurhade
**Team:** Nirupam Gupta · Vidhi Patel · Param Thakkar

---

## Table of Contents

1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Core Features](#4-core-features)
5. [User Stories](#5-user-stories)
6. [Patient Medical Vault & QR-Based Access Sharing](#6-patient-medical-vault--qr-based-access-sharing)
7. [System Architecture](#7-system-architecture)
8. [Machine Learning Components](#8-machine-learning-components)
9. [Technology Stack](#9-technology-stack)
10. [Project Structure](#10-project-structure)
11. [Data Pipeline](#11-data-pipeline)
12. [Database Schema](#12-database-schema)
13. [API Design](#13-api-design)
14. [Installation & Setup](#14-installation--setup)
15. [How to Run](#15-how-to-run)
16. [Evaluation Metrics & Targets](#16-evaluation-metrics--targets)
17. [Project Roadmap (Phases)](#17-project-roadmap-phases)
18. [Course Mapping](#18-course-mapping)
19. [Related Work / References](#19-related-work--references)
20. [Limitations & Future Scope](#20-limitations--future-scope)
21. [Team & Acknowledgements](#21-team--acknowledgements)

---

## 1. Overview

**MediHaven** is an AI-powered clinical decision support system that predicts adverse patient outcomes and recommends personalized treatment protocols across hospital and clinic settings. The system integrates multimodal patient data — real-time vital signs, laboratory biomarkers, electronic health records (EHR), and medication profiles — to generate predictive, interpretable risk assessments.

Instead of waiting for a patient's condition to visibly deteriorate, MediHaven continuously analyzes incoming clinical data, compares the current patient against a historical population, and surfaces risk 7–14 days before a critical event is likely to occur. Every prediction is traceable back to real patient precedent and explainable decision logic, so physicians can verify the reasoning rather than trust a black box.

Beyond prediction, MediHaven also gives **patients ownership of their own medical history** through a personal Medical Vault, which they can share with any physician or hospital instantly and securely via a QR code — removing the friction of carrying physical files or re-explaining history at every visit (see [Section 6](#6-patient-medical-vault--qr-based-access-sharing)).

The system is being developed as a semester-long academic mini project, structured into four phases: Problem Definition, Architecture & Design, Implementation, and Integration & Testing. This document describes the project end-to-end, including what is being built, how it will be built, and the structure the final application will follow.

---

## 2. Problem Statement

> Patient deterioration and disease risk are often identified too late, because clinical data is reviewed manually and reactively rather than predicted in advance.

### 2.1 Pain Points Being Addressed

| # | Problem | Description |
|---|---------|-------------|
| 1 | **Delayed Diagnosis** | Manual review of patient charts slows down identification of high-risk conditions. |
| 2 | **No Early Warning** | Hospitals lack systems that flag deteriorating patients before a critical event occurs. |
| 3 | **Experience-Based Treatment** | Treatment decisions rely on physician intuition rather than outcome data from similar cases. |
| 4 | **Rising Readmissions** | Absence of predictive discharge planning leads to preventable hospital readmissions. |
| 5 | **Fragmented Medical History** | Patients' records are scattered across different hospitals/clinics, with no single, patient-owned source of truth they can share on demand. |

### 2.2 Why Existing Systems Fall Short

- **Reactive, not predictive** — most hospital information systems act only after symptoms escalate.
- **Siloed data** — EHRs, lab systems, and vitals monitors rarely talk to each other.
- **No standardized comparison** — there's no systematic way to compare a new patient against similar historical cases.
- **Doesn't scale** — manual chart review cannot keep pace with rising patient volumes.
- **No patient-controlled portability** — patients have no simple, secure way to carry and share their own history across providers; records stay locked inside a single hospital's system.

---

## 3. Objectives

1. Predict patient risk **7–14 days in advance** of clinical deterioration.
2. **Stratify patients** into clinically meaningful risk tiers (Low / Medium / High / Critical).
3. **Recommend treatment paths** by matching the current patient against similar historical cases and their outcomes.
4. Provide **interpretable output** that physicians can verify, not a black-box prediction.
5. Give patients a **secure, self-owned medical history vault** that they can share with any physician or hospital in seconds via QR code, with full control over what is shared and for how long.

### Scope (Current Semester — Phase 1)

Phase 1 establishes the problem definition, objectives, and literature grounding described in this document. Model design (Phase 2), implementation (Phase 3), and integration/testing (Phase 4) are addressed in subsequent phases of the project.

---

## 4. Core Features

- **Risk Stratification Dashboard** — a single view for physicians showing each patient's current risk tier and the factors driving it.
- **Early-Warning Alerts** — automated flags when a patient's trajectory starts resembling historical patients who later deteriorated.
- **Treatment Recommendation Engine** — surfaces treatments that worked for clinically similar past patients.
- **Explainable Predictions** — every output is backed by a traceable rule path (Decision Tree) and/or a list of similar historical patients (KNN), not an opaque score.
- **Readmission Risk Scoring** — estimates 30-day readmission likelihood at the point of discharge.
- **Patient Medical Vault** — patients store their complete medical history (diagnoses, prescriptions, allergies, lab reports, surgeries, imaging, vaccination records) directly on the platform.
- **QR-Based Access Sharing** — patients generate a QR code to grant physicians or hospitals direct, scoped, time-boxed access to their vault — no manual file transfers, no repeated paperwork.

---

## 5. User Stories

User stories are grouped by persona to capture the needs of everyone who touches the system — patients, physicians, hospital staff, and caregivers.

### 5.1 Patient

| # | User Story |
|---|---|
| P1 | As a **patient**, I want to upload and store my complete medical history (diagnoses, prescriptions, lab reports, allergies, surgeries) in one secure place, so that I don't have to carry physical files between doctors. |
| P2 | As a **patient**, I want to generate a QR code that grants temporary access to my medical history, so that I can share my records with a new doctor instantly without emailing or printing anything. |
| P3 | As a **patient**, I want to choose exactly which parts of my history are shared (e.g., only allergies and current medications, not my full psychiatric history), so that I retain control and privacy over sensitive information. |
| P4 | As a **patient**, I want my QR code to automatically expire after a set time or after first use, so that my data isn't accessible indefinitely once an appointment is over. |
| P5 | As a **patient**, I want to manually revoke a previously shared QR code at any time, so that I stay in control even after sharing it. |
| P6 | As a **patient**, I want to see an access log of who viewed my medical history and when, so that I have full transparency into how my data has been used. |
| P7 | As a **patient**, I want to receive an early-warning notification if the system detects a concerning trend in my own health data, so that I can seek care proactively instead of waiting for my next scheduled appointment. |

### 5.2 Physician / Treating Doctor

| # | User Story |
|---|---|
| D1 | As a **physician**, I want to scan a patient's QR code at the start of a consultation, so that I immediately see their relevant medical history without re-asking questions the patient has already answered elsewhere. |
| D2 | As a **physician**, I want to see a patient's predicted risk tier the moment I open their profile, so that I can prioritize the most urgent cases during ward rounds. |
| D3 | As a **physician**, I want to see *why* the system flagged a patient as high-risk (e.g., the Decision Tree rule path or similar historical cases), so that I can make an informed clinical judgment rather than blindly trusting an AI score. |
| D4 | As a **physician**, I want to view treatment protocols that worked well for clinically similar past patients, so that I can make faster, evidence-backed treatment decisions. |

### 5.3 Emergency Room / Urgent Care Staff

| # | User Story |
|---|---|
| E1 | As an **ER doctor**, I want to scan an unconscious or incoming patient's MediHaven QR code (e.g., from their phone lock screen or a physical wallet card) so that I can instantly access critical allergy and medication history during an emergency. |
| E2 | As an **ER doctor**, I want critical life-saving information (allergies, blood type, chronic conditions) to be accessible quickly in emergencies, even when the patient cannot actively consent in the moment, so that treatment isn't delayed. |

### 5.4 Hospital Administrator

| # | User Story |
|---|---|
| A1 | As a **hospital administrator**, I want a dashboard view of risk distribution across all admitted patients, so that I can allocate staff and ICU beds more effectively. |
| A2 | As a **hospital administrator**, I want to track readmission risk trends across the hospital, so that I can identify systemic gaps in discharge planning. |

### 5.5 Specialist (Referral Physician)

| # | User Story |
|---|---|
| S1 | As a **specialist** receiving a referral, I want direct, read-only access to the relevant portion of a patient's history via their shared QR code, so that I can assess the case immediately instead of waiting for records to be faxed or emailed over. |

### 5.6 Caregiver / Family Member

| # | User Story |
|---|---|
| C1 | As a **caregiver** managing an elderly parent's care, I want to help them maintain and share their medical history, so that every specialist they visit has full context without the family repeating the same explanations. |

---

## 6. Patient Medical Vault & QR-Based Access Sharing

This is a core patient-facing feature of MediHaven: a secure, **patient-owned** repository of medical history that can be shared with any physician or hospital **directly and instantly**, without manual file transfers, through a dynamically generated QR code.

### 6.1 Concept

Rather than medical history living inside a single hospital's siloed system, the patient holds their own complete record inside MediHaven. The patient — not the hospital — decides who gets to see it, what they see, and for how long. A physician or hospital simply scans the patient's QR code to get **direct, read-only access** to the shared portion of the vault.

### 6.2 What Can Be Stored in the Vault

- Past diagnoses and conditions
- Prescriptions and current medications
- Allergies
- Past surgeries / procedures
- Lab reports and imaging (uploaded documents/scans)
- Vaccination records
- Chronic condition history (e.g., diabetes, hypertension)

### 6.3 End-to-End Sharing Flow

```
┌───────────────┐   1. Upload history    ┌──────────────────┐
│    Patient     │ ─────────────────────▶ │  Medical Vault    │
│   (mobile/web) │                        │  (patient-owned)  │
└───────┬───────┘                        └──────────────────┘
        │ 2. Choose what to share
        │    (scope + expiry)
        ▼
┌───────────────────────┐
│  Generate QR Code       │   Encodes a signed, time-boxed
│  (signed access token)  │   access token (JWT)
└───────────┬────────────┘
            │ 3. Patient shows QR code
            ▼
┌───────────────────────┐   4. Scan QR code     ┌───────────────────────┐
│  Physician / Hospital   │ ─────────────────────▶ │  Token Validation       │
│  (provider portal/app)  │                        │  (expiry, revocation,   │
└───────────┬────────────┘                        │   scope check)          │
            │                                       └───────────┬───────────┘
            │ 5. Access granted (scoped, read-only)              │
            ▼                                                    ▼
┌───────────────────────┐                        ┌───────────────────────┐
│  Views shared vault     │ ◀───────────────────── │  Vault Access Log       │
│  data for consultation  │   6. Every access       │  (visible to patient)  │
└───────────────────────┘      is logged           └───────────────────────┘
```

**Step-by-step:**

1. **Upload** — the patient uploads documents and/or fills structured history (conditions, allergies, medications, surgeries) into their Medical Vault.
2. **Select scope** — before sharing, the patient chooses *what* to share (e.g., "full history" or "only allergies + current medications") and *for how long* (e.g., 24 hours, single use, or until manually revoked).
3. **Generate QR code** — the system creates a cryptographically signed, time-boxed access token and encodes it into a QR code shown on the patient's phone (or printable as a physical card).
4. **Scan** — the physician or hospital reception scans the QR code using the MediHaven provider portal/app.
5. **Validate & grant access** — the backend verifies the token's signature, checks it hasn't expired or been revoked, and returns only the scoped data — never the patient's full vault unless explicitly granted.
6. **Audit trail** — every scan is logged (who accessed it, what was accessed, and when) and is visible to the patient at any time.
7. **Revoke** — the patient can revoke a token at any point, immediately invalidating that QR code even if it hasn't expired yet.

### 6.4 Design Principles

- **Patient-owned, patient-initiated** — there is no standing/default access; every share is explicitly triggered by the patient.
- **Granular scope control** — share everything, or only specific categories (e.g., allergies and medications only).
- **Time-boxed by default** — tokens can be configured to expire after a set duration or after first use, minimizing the window of exposure.
- **Full transparency** — every access event is logged and visible to the patient, building trust in how their data is used.
- **Read-only for providers** — physicians and hospitals can view shared data but cannot edit the patient's original vault record.
- **Emergency access (future consideration)** — a restricted "critical info only" mode (allergies, blood type, chronic conditions) for emergency scenarios where the patient cannot actively grant access in the moment; this requires careful ethical and consent-design discussion and is flagged under [Future Scope](#20-limitations--future-scope) rather than assumed as implemented in this phase.

### 6.5 Why QR Codes

- **Zero manual transfer** — no emailing PDFs, no faxing, no physical photocopies.
- **Instant** — a scan takes seconds versus days for inter-hospital record requests.
- **Verifiable** — the token embedded in the QR code is signed, so it can't be forged or tampered with.
- **Familiar UX** — patients and hospital staff are already used to scanning QR codes (payments, check-ins), lowering the adoption barrier.

---

## 7. System Architecture

### 7.1 High-Level Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   Patient Data   │ ──▶ │  Preprocessing   │ ──▶ │   ML-Based Prediction │ ──▶ │   Risk Output    │
│                  │     │                  │     │                       │     │                  │
│ Vitals, labs,    │     │ Cleaning,        │     │ Risk clustering &     │     │ Risk tier +      │
│ EHR, medication  │     │ normalization,   │     │ disease classification│     │ treatment         │
│ history          │     │ feature scaling  │     │                       │     │ recommendation   │
└─────────────────┘     └─────────────────┘     └──────────────────────┘     └─────────────────┘
```

### 7.2 Conceptual Layered Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                        │
│   Physician Dashboard (Web UI) · Patient Portal (Vault & QR)     │
│   Provider QR Scanner View · Alerts/Notifications Panel          │
└───────────────────────────────┬────────────────────────────────┘
                                 │  REST API (JSON over HTTP)
┌───────────────────────────────▼────────────────────────────────┐
│                         APPLICATION LAYER                        │
│   Flask Backend · Request Routing · Auth · Alert Engine          │
│   Vault Access Token Service (QR generation & validation)        │
└───────────────────────────────┬────────────────────────────────┘
                                 │
        ┌────────────────────────┴─────────────────────────┐
        ▼                                                    ▼
┌──────────────────────────┐                   ┌───────────────────────────┐
│   ML INTELLIGENCE LAYER    │                   │   MEDICAL VAULT SERVICE    │
│ Preprocessing → KNN        │                   │ Upload/store history       │
│ → K-Means → Decision Tree  │                   │ QR token issuance          │
│ → Neural Network           │                   │ Scoped data retrieval       │
└─────────────┬─────────────┘                   │ Access logging             │
              │                                   └──────────────┬────────────┘
              ▼                                                   ▼
┌────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                              │
│   SQLite/MySQL — patient records, predictions, outcomes,         │
│   medical_vault, vault_access_tokens, vault_access_log           │
│   Feature store (engineered/normalized features)                  │
└────────────────────────────────────────────────────────────────┘
```

### 7.3 Design Principle

Every prediction must trace back to comparable historical patients and interpretable clinical rules — never an unexplainable black box. This principle governs model selection: Decision Trees and KNN are preferred for their transparency, and the Neural Network component is used only where non-linear pattern recognition is required, with its influence on the final decision kept auditable.

The same transparency principle extends to the Medical Vault: every access to a patient's data must be explicitly authorized, scoped, and logged — there is no implicit or default access for any provider.

---

## 8. Machine Learning Components

MediHaven is built on four complementary models, each covering a distinct responsibility in the pipeline.

### 8.1 K-Nearest Neighbors (KNN) — Similarity Matching

- **Role:** Given a new patient's feature vector, finds the K most similar past patients (by Euclidean/Manhattan distance on normalized features) and retrieves their documented outcomes.
- **Use case:** Treatment recommendation — "patients similar to this one responded best to Protocol C."
- **Why it's used:** Fully transparent — the recommendation is literally backed by real precedent cases a physician can inspect.

### 8.2 K-Means Clustering — Risk Stratification

- **Role:** Unsupervised grouping of the patient population into clusters based on shared clinical characteristics (vitals stability, lab abnormality counts, comorbidity load, etc.).
- **Use case:** Assigns every patient to a risk tier (e.g., Healthy / At-Risk / Diagnosed) without needing pre-labeled outcomes.
- **Why it's used:** Surfaces natural patterns in the population that a rules-based system might miss, and gives a reusable "risk tier" label consumed by the dashboard and alerting engine.

### 8.3 Decision Tree — Interpretable Classification

- **Role:** Learns if-then splitting rules (using information gain / Gini impurity) to classify a patient's disease risk from their features.
- **Use case:** The primary explainability engine — each prediction can be traced as a short, human-readable path of conditions (e.g., "blood sugar > 110 → BMI > 27 → BP > 130/80 → HIGH RISK").
- **Why it's used:** Directly satisfies the "interpretable by design" objective.

### 8.4 Neural Network — Complex Pattern Recognition

- **Role:** Captures non-linear interactions across multimodal features (temporal vitals trends, combined lab/medication effects) that simpler models may underfit.
- **Use case:** Used as a secondary, higher-capacity model for the hardest prediction tasks (e.g., sepsis onset), with its output cross-checked against the Decision Tree and KNN outputs before being surfaced.
- **Why it's used:** Provides a ceiling on predictive performance for complex, multi-factor conditions while the rest of the pipeline keeps the overall system auditable.

### 8.5 Model Composition (Illustrative Split)

| Model | Contribution | Primary Output |
|---|---|---|
| KNN | 25% | Similar-patient treatment matches |
| K-Means | 25% | Risk tier assignment |
| Decision Tree | 25% | Interpretable risk classification |
| Neural Network | 25% | Complex pattern-based risk score |

### 8.6 End-to-End Prediction Flow

1. New patient data is ingested and run through the **preprocessing** step (cleaning, normalization, feature scaling).
2. **K-Means** assigns the patient to a risk cluster using their current feature vector.
3. **Decision Tree** independently classifies the patient's risk level and produces a readable rule path.
4. **KNN** retrieves the K most similar historical patients and their outcomes/treatments.
5. **Neural Network** scores more complex, multi-factor risk patterns (used selectively for conditions where the simpler models show low confidence).
6. All four outputs are reconciled into a single **risk tier + recommendation + explanation** object returned to the physician dashboard.

---

## 9. Technology Stack

| Category | Tools / Libraries |
|---|---|
| **Programming Language** | Python 3 |
| **Machine Learning** | Scikit-learn (KNN, K-Means, Decision Tree), TensorFlow / Keras (Neural Network) |
| **Data Processing** | Pandas, NumPy |
| **Backend Framework** | Flask (REST API) |
| **Database** | SQLite (development) / MySQL (scalable deployment) |
| **QR Code Generation** | `qrcode` (Python library) |
| **QR Code Scanning** | `pyzbar`, `opencv-python` |
| **Secure Access Tokens** | `PyJWT` (signed, time-boxed vault access tokens) |
| **Data Visualization** | Matplotlib, Seaborn |
| **API Testing** | Postman |
| **Development Environment** | Visual Studio Code / Jupyter Notebook |
| **Version Control** | Git & GitHub |
| **Dataset Sources** | UCI Machine Learning Repository, MIMIC-III Clinical Database, Kaggle Healthcare Datasets |
| **Deployment Target** | Laptop and mobile-accessible web application |

---

## 10. Project Structure

Proposed repository layout for implementation (Phases 2–3):

```
medihaven/
│
├── README.md                     # This file
├── requirements.txt               # Python dependencies
├── .gitignore
│
├── data/
│   ├── raw/                       # Original, unmodified datasets
│   ├── processed/                 # Cleaned, feature-engineered datasets
│   └── external/                  # Reference datasets (UCI, MIMIC-III, Kaggle)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_kmeans_clustering.ipynb
│   ├── 04_decision_tree.ipynb
│   ├── 05_knn_similarity.ipynb
│   ├── 06_neural_network.ipynb
│   └── 07_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── loader.py               # Data ingestion utilities
│   │   ├── cleaner.py              # Missing value handling, outlier detection
│   │   └── feature_engineering.py  # Normalization, scaling, derived features
│   │
│   ├── models/
│   │   ├── knn_model.py            # KNN similarity matching
│   │   ├── kmeans_model.py         # Risk-tier clustering
│   │   ├── decision_tree_model.py  # Interpretable classification
│   │   ├── neural_network_model.py # Complex pattern recognition
│   │   └── ensemble.py             # Reconciles outputs from all models
│   │
│   ├── vault/
│   │   ├── vault_service.py        # Upload/store/retrieve medical history
│   │   ├── qr_generator.py         # Generates signed QR access tokens
│   │   ├── qr_scanner.py           # Decodes/validates scanned QR tokens
│   │   └── access_logger.py        # Writes to vault_access_log
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # Accuracy, precision, recall, F1, AUROC
│   │   └── cross_validation.py     # K-fold validation utilities
│   │
│   ├── api/
│   │   ├── app.py                  # Flask application entry point
│   │   ├── routes/
│   │   │   ├── patients.py         # Patient CRUD endpoints
│   │   │   ├── predictions.py      # Risk prediction endpoints
│   │   │   ├── alerts.py           # Early-warning alert endpoints
│   │   │   └── vault.py            # Vault upload, QR share/revoke, access endpoints
│   │   └── schemas.py              # Request/response validation
│   │
│   └── utils/
│       ├── config.py                # Configuration constants
│       └── logger.py                # Logging utilities
│
├── database/
│   ├── schema.sql                  # Table definitions
│   └── seed_data.sql               # Sample/demo data
│
├── models_store/                   # Serialized trained models (.pkl, .h5)
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_vault.py                # Vault upload, QR generation/validation tests
│   └── test_api.py
│
└── docs/
    ├── architecture.md
    ├── literature_review.md
    └── presentation/                # Phase PPTs and reports
```

---

## 11. Data Pipeline

### 11.1 Data Sources

- **UCI Machine Learning Repository** — structured clinical datasets (e.g., diabetes, heart disease).
- **MIMIC-III Clinical Database** — de-identified ICU patient records for realistic multimodal data.
- **Kaggle Healthcare Datasets** — supplementary datasets for model validation.

### 11.2 Pipeline Stages

1. **Ingestion** — raw patient records (vitals, labs, EHR fields, medications) are loaded from `data/raw/`.
2. **Cleaning** — missing value imputation, outlier detection/removal, deduplication.
3. **Feature Engineering** — derived features (e.g., rate of change in vitals, cumulative abnormality score), encoding of categorical fields.
4. **Normalization / Scaling** — Min-Max normalization or Z-score standardization so no single feature (e.g., blood sugar range 80–300) dominates distance-based models like KNN and K-Means.
5. **Train/Test Split** — 80/20 split, with 5-fold cross-validation used during model development to ensure reliable accuracy estimates.
6. **Feature Store Output** — final processed features are written to `data/processed/` and consumed by the model training notebooks/scripts.

---

## 12. Database Schema

Indicative schema for the relational store (SQLite/MySQL):

```sql
-- Patients
CREATE TABLE patients (
    patient_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    age             INTEGER,
    gender          TEXT,
    admission_date  DATETIME,
    discharge_date  DATETIME
);

-- Vitals (time-series)
CREATE TABLE vitals (
    vital_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER REFERENCES patients(patient_id),
    recorded_at     DATETIME,
    heart_rate      REAL,
    blood_pressure_sys REAL,
    blood_pressure_dia REAL,
    temperature     REAL,
    oxygen_saturation REAL
);

-- Lab Results
CREATE TABLE lab_results (
    lab_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER REFERENCES patients(patient_id),
    recorded_at     DATETIME,
    glucose         REAL,
    creatinine      REAL,
    cholesterol     REAL,
    wbc_count       REAL
);

-- Medications
CREATE TABLE medications (
    medication_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER REFERENCES patients(patient_id),
    drug_name       TEXT,
    dosage          TEXT,
    start_date      DATETIME,
    end_date        DATETIME
);

-- Predictions (model output log)
CREATE TABLE predictions (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER REFERENCES patients(patient_id),
    predicted_at    DATETIME,
    risk_tier       TEXT,          -- Low / Medium / High / Critical
    risk_score      REAL,
    model_used      TEXT,          -- KNN / K-Means / Decision Tree / Neural Net / Ensemble
    explanation     TEXT           -- Rule path or similar-patient summary
);

-- Outcomes (ground truth, used for retraining/evaluation)
CREATE TABLE outcomes (
    outcome_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER REFERENCES patients(patient_id),
    diagnosis       TEXT,
    readmitted_30d  BOOLEAN,
    outcome_date    DATETIME
);

-- ===================== Medical Vault & QR Sharing =====================

-- Medical Vault: patient-owned medical history records
CREATE TABLE medical_vault (
    vault_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER REFERENCES patients(patient_id),
    category        TEXT,      -- diagnosis / allergy / medication / lab_report /
                                -- imaging / surgery / vaccination
    title           TEXT,
    description     TEXT,
    file_path       TEXT,      -- for uploaded documents/scans (nullable)
    structured_data TEXT,      -- JSON for structured fields (nullable)
    uploaded_at     DATETIME
);

-- Vault Access Tokens: QR-code-backed, scoped, time-boxed access grants
CREATE TABLE vault_access_tokens (
    token_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER REFERENCES patients(patient_id),
    scope_json      TEXT,      -- e.g. ["allergy", "medication"] or ["*"] for full history
    issued_at       DATETIME,
    expires_at      DATETIME,
    max_uses        INTEGER,   -- e.g. 1 for single-use, NULL for unlimited until expiry
    use_count        INTEGER DEFAULT 0,
    revoked         BOOLEAN DEFAULT 0,
    qr_payload      TEXT       -- signed JWT encoded into the QR code
);

-- Vault Access Log: full audit trail of every scan/access
CREATE TABLE vault_access_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    token_id        INTEGER REFERENCES vault_access_tokens(token_id),
    accessed_by     TEXT,      -- physician/hospital identifier or name
    accessed_at     DATETIME,
    device_info     TEXT       -- optional metadata (IP/device) for audit purposes
);
```

---

## 13. API Design

Indicative REST endpoints exposed by the Flask backend:

### 13.1 Core Prediction & Dashboard Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/patients` | List all patients |
| `GET` | `/api/patients/<id>` | Get a single patient's full record |
| `POST` | `/api/patients` | Add a new patient record |
| `GET` | `/api/patients/<id>/vitals` | Get time-series vitals for a patient |
| `GET` | `/api/patients/<id>/predict` | Run the ML pipeline and return risk tier + recommendation |
| `GET` | `/api/patients/<id>/similar` | Return the K most similar historical patients (KNN) |
| `GET` | `/api/alerts` | List current early-warning alerts across all patients |
| `GET` | `/api/dashboard/summary` | Aggregate statistics for the risk-stratification dashboard |

**Sample response — `GET /api/patients/42/predict`:**

```json
{
  "patient_id": 42,
  "risk_tier": "High",
  "risk_score": 0.78,
  "model_used": "Ensemble",
  "explanation": {
    "decision_tree_path": [
      "blood_sugar > 110",
      "bmi > 27",
      "blood_pressure > 130/80"
    ],
    "similar_patients": [101, 87, 56, 29, 14],
    "recommended_protocol": "Protocol C (75% success rate in similar cases)"
  },
  "predicted_at": "2026-10-06T10:15:00Z"
}
```

### 13.2 Medical Vault & QR Sharing Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/vault/upload` | Patient uploads a new medical history record/document |
| `GET` | `/api/vault/<patient_id>` | Patient views their own full vault (authenticated as patient) |
| `POST` | `/api/vault/<patient_id>/share` | Generate a new QR code/token with chosen scope + expiry |
| `POST` | `/api/vault/revoke/<token_id>` | Patient revokes a previously issued token |
| `GET` | `/api/vault/access/<token>` | Physician/hospital scans QR → validated → returns scoped vault data (and logs the access) |
| `GET` | `/api/vault/<patient_id>/access-log` | Patient views the full audit trail of who accessed their data and when |

**Sample request — `POST /api/vault/42/share`:**

```json
{
  "scope": ["allergy", "medication", "diagnosis"],
  "expires_in_minutes": 1440,
  "max_uses": 1
}
```

**Sample response:**

```json
{
  "token_id": 7731,
  "qr_payload": "eyJhbGciOiJIUzI1NiIs...<signed JWT>",
  "qr_image_url": "/api/vault/qr/7731.png",
  "expires_at": "2026-10-07T10:15:00Z"
}
```

**Sample response — `GET /api/vault/access/<token>` (physician scans QR):**

```json
{
  "patient_id": 42,
  "patient_name": "Redacted for audit view",
  "shared_scope": ["allergy", "medication", "diagnosis"],
  "records": [
    { "category": "allergy", "title": "Penicillin allergy", "description": "Severe reaction, documented 2023" },
    { "category": "medication", "title": "Metformin 500mg", "description": "Twice daily, ongoing" },
    { "category": "diagnosis", "title": "Type 2 Diabetes", "description": "Diagnosed 2021" }
  ],
  "access_logged": true
}
```

---

## 14. Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/<org>/medihaven.git
cd medihaven

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize the database
sqlite3 database/medihaven.db < database/schema.sql
sqlite3 database/medihaven.db < database/seed_data.sql

# 5. Configure environment variables
cp .env.example .env
# edit .env with database path, Flask secret key, JWT signing key, etc.
```

**`requirements.txt` (indicative):**

```
flask
scikit-learn
tensorflow
pandas
numpy
matplotlib
seaborn
sqlalchemy
python-dotenv
pytest
qrcode
pyzbar
opencv-python
pyjwt
```

---

## 15. How to Run

```bash
# Train the models (writes serialized models to models_store/)
python src/models/kmeans_model.py
python src/models/decision_tree_model.py
python src/models/knn_model.py
python src/models/neural_network_model.py

# Start the Flask API server
python src/api/app.py
# Server runs on http://localhost:5000

# Run the test suite
pytest tests/
```

---

## 16. Evaluation Metrics & Targets

Models are evaluated using 5-fold cross-validation with the following target benchmarks, informed by comparable published literature (see [Section 19](#19-related-work--references)):

| Metric | Target |
|---|---|
| Accuracy | 91% |
| Recall | 88% |
| Precision | 89% |
| F1-Score | 89% |
| Early Prediction Window | 7–14 days |
| Reduction in Adverse Events | 34% |
| Improvement in Treatment Success | 18% |

> **Note:** These are realistic, achievable targets drawn from comparable literature — not guaranteed outcomes at this stage of the project. Final figures will be reported after model training and validation in Phase 3.

**Evaluation methodology:**

- **Confusion Matrix** — tracks true positives, true negatives, false positives, and false negatives for each risk tier classification.
- **Cross-Validation** — 5-fold split ensures the reported accuracy is not an artifact of a single lucky train/test split.
- **Precision vs. Recall trade-off** — tuned according to clinical priority (for early-warning alerts, recall is favored to avoid missing real deterioration; for treatment recommendations, precision is favored to avoid unnecessary interventions).

The Medical Vault feature is evaluated separately on non-ML criteria: QR token validation correctness, access scope enforcement, token expiry/revocation reliability, and completeness of the access-log audit trail.

---

## 17. Project Roadmap (Phases)

| Phase | Focus | Deliverables |
|---|---|---|
| **P1 — Problem Definition** | Problem statement, objectives, scope, user stories, literature survey | This README, Phase 1 presentation, literature table |
| **P2 — Architecture & Design** | System architecture, database schema, API design, model selection finalized, Vault/QR flow design | Architecture diagrams, ER diagrams, API spec |
| **P3 — Implementation** | Data pipeline, model training, backend development, Vault upload + QR generation/scanning | Trained models, working Flask API, preprocessing pipeline, functional Vault & QR sharing |
| **P4 — Integration & Testing** | End-to-end integration, dashboard, validation, testing | Fully integrated system, test reports, final evaluation metrics |

---

## 18. Course Mapping

This project is mapped exclusively to the following core Computer Engineering courses:

| Course Code | Course Name | Application in MediHaven |
|---|---|---|
| **CE204** | Database Management Systems | Patient record storage, EHR management, query optimization, schema design, Vault & access-token schema |
| **AS202** | Python Programming for Data Science | Model building (NumPy, Pandas), data preprocessing, visualization |
| **CE205** | Statistical Methods in Computer Science | Accuracy, precision, recall, F1-Score, cross-validation, confusion matrix analysis |
| **CE207** | Design and Analysis of Algorithms | Algorithm complexity analysis (KNN O(n·d), K-Means O(n·k·d·i), Decision Tree O(n log n)) |
| **CE202** | Data Structures | Efficient patient data storage, similarity search structures, priority queues for alerts, token/log storage |

---

## 19. Related Work / References

| # | Paper | Method | Key Result | Gap MediHaven Addresses |
|---|---|---|---|---|
| 1 | K-Means and Decision Tree for Disease Prediction Using Data Mining (2026) — *ejournal.unibabwi.ac.id* | Clustering + Classification | 0.5556 Silhouette Score on 4,633 patients | No real-time monitoring |
| 2 | Integrating Decision Tree and K-Means Clustering in Heart Disease Diagnosis — *academia.edu* | Compared DT variants (C4.5, CART, C5.0) | Improved centroid selection accuracy | No treatment recommendation |
| 3 | Collaboration of Clustering and Classification for Heart Stroke Severity Prediction (2024) — *ScienceDirect* | K-Means + DT / Naive Bayes / Deep Learning | 90% accuracy with Decision Trees | No temporal vitals tracking |
| 4 | Toward Reliable Diabetes Prediction: Data Engineering & ML (2024) — *SAGE Journals* | Data preprocessing pipeline for diabetes prediction | Improved preprocessing reliability | Diabetes-specific, not generalized |
| 5 | Prediction Model for CVD in Diabetic Patients (2024) — *Nature Scientific Reports* | Random Forest, 12,809 patients | AUROC 0.83 (validated cohort) | No resource optimization |
| 6 | A Comprehensive Review of ML for Heart Disease Prediction (2024) — *PMC* | Survey: Decision Tree, Random Forest, Neural Networks | Comparative performance analysis | No unified multimodal system |

**Takeaway:** Existing work targets single-disease prediction or diagnosis alone. MediHaven's contribution is unifying risk stratification, multi-disease prediction, treatment recommendation, and patient-owned portable medical history into a single interpretable system.

---

## 20. Limitations & Future Scope

**Current limitations (Phase 1 scope):**

- No real-time streaming ingestion yet — current design assumes batch/periodic data updates.
- Neural Network component requires a larger labeled dataset to reach target accuracy; may initially run with reduced confidence weighting.
- Authentication/authorization for the physician and patient portals is defined at a conceptual level only; a production-grade identity system (e.g., OAuth2) is deferred to a later phase.
- Emergency-access mode for the Medical Vault (critical info accessible without active patient consent in life-threatening situations) is identified as a need but requires further ethical and consent-design discussion before implementation — not assumed as built in this phase.

**Future scope:**

- Integration with hospital IoT devices for true real-time vitals streaming.
- Mobile-native app (beyond responsive web) for ward-round usage and for patients to manage their Vault and QR codes on the go.
- Expansion to additional disease categories beyond the initial target set (diabetes, cardiovascular disease, sepsis).
- Federated learning exploration to train across multiple hospitals without centralizing sensitive patient data.
- Physical "MediHaven card" (printed QR) as a wallet-sized emergency-access backup for patients without a smartphone on hand.
- Expiring, single-use QR codes for walk-in/one-time consultations versus longer-lived tokens for ongoing specialist relationships.

---

## 21. Team & Acknowledgements

| Role | Name |
|---|---|
| Faculty Guide | Prof. Swapnali Kurhade |
| Team Member | Nirupam Gupta |
| Team Member | Vidhi Patel |
| Team Member | Param Thakkar |

**Institution:** Sardar Patel Institute of Technology (SPIT), Department of Computer Engineering, Mumbai.

This project is developed as part of the TE Sem V Mini Project I curriculum requirement, mapped exclusively to the core courses listed in [Section 18](#18-course-mapping).

---

*This README reflects the Phase 1 (Problem Definition) scope of MediHaven. Sections on implementation, API, database schema, and the Medical Vault/QR sharing feature describe the intended design and will be updated as the project progresses through Phases 2–4.*
