# MediHaven: Master Project Build Plan

> **Integrated Multimodal Clinical Intelligence System for Patient Risk Assessment and Treatment Optimization**  
> *Department of Computer Engineering, SPIT | Academic Year 2026–27*  
> Reference: [README.md](README.md)

---

## 1. Executive Summary & Strategy

This master build plan translates the architectural specifications, user stories, clinical requirements, and academic course mappings from `README.md` into an actionable, progressive 8-phase engineering roadmap.

The project is structured into **8 modular phases**. Each phase is independently testable, delivers verified artifacts, and directly prepares the foundational assets for subsequent phases:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        8-PHASE BUILD ARCHITECTURE                      │
└────────────────────────────────────────────────────────────────────────┘
  Phase 1: Environment, Scaffolding & Repository Foundation
           (Including CSS Design Tokens & Base UI Layout)
    │
    ▼
  Phase 2: Relational Database Architecture & Seeding Engine
    │
    ├──────────────────────────────────────────────────────┐
    ▼                                                      ▼
  Phase 3: Clinical Data Engineering              Phase 5: Medical Vault &
  & Feature Pipeline                              Cryptographic QR Engine
    │                                                      │
    ▼                                                      │
  Phase 4: ML Intelligence & Explainability Core           │
  (K-Means, Decision Tree, KNN, Neural Net, Ensemble)     │
    │                                                      │
    └──────────────────────┬───────────────────────────────┘
                           ▼
  Phase 6: RESTful API Layer & Application Services (Flask)
                           │
                           ▼
  Phase 7: Interactive Presentation Layer & Web Portals
  (Physician Clinical Triage, Patient Vault Portal, QR Scanner)
                           │
                           ▼
  Phase 8: End-to-End Testing, Validation & Clinical Benchmarking
```

---

## 2. UI/UX Design System & Anti-AI-Aesthetic Specifications

To ensure MediHaven feels like a **world-class clinical platform** (comparable to modern Epic MyChart, Apple Health Records, and Linear) rather than an AI-generated prototype, the UI is governed by strict, human-designed principles:

### 2.1 Anti-AI Design Rules
1. **No Cliché AI Gradients & Glowing Blobs**: Eliminate gratuitous purple-to-cyan neon gradients, random glowing orbs, and cartoonish rainbow shadows.
2. **Clinical Restraint & Intentional Palette**: Colors are calibrated to hospital communication standards:
   * **Primary Surface / Slate**: Deep Obsidian Slate (`#0B0F17`) or Crisp Medical White (`#FFFFFF`).
   * **Card Background**: Refined Slate (`#111827`) with 1px hairline borders (`#1E293B` / `#E2E8F0`).
   * **Clinical Accent**: Surgical Cobalt / Slate Teal (`#0284C7` / `#0D9488`) — authoritative and sterile.
   * **Stable / Low Risk**: Clinical Sage Green (`#10B981` / text `#059669` / pill `#ECFDF5`).
   * **Monitor / Medium Risk**: Medical Ochre / Amber (`#F59E0B` / text `#D97706` / pill `#FFFBEB`).
   * **Urgent / Deterioration**: Clinical Crimson (`#EF4444` / text `#DC2626` / pill `#FEF2F2`).
   * **Vault & Cryptography**: Deep Navy Indigo (`#4338CA` / `#312E81`).
3. **Clinical Data Density**: Physicians need high-density, structured information — not massive cartoonish cards with 40px padding. Tables feature crisp dividers, aligned decimal columns, and tabular numerals.
4. **Typography**:
   * Clean, authoritative sans-serif: `Plus Jakarta Sans` for headers, `Inter` for clinical data.
   * Tabular figures (`font-variant-numeric: tabular-nums`) for vitals, blood pressure, lab values, and timestamps to prevent layout shifts.
   * Monospace (`JetBrains Mono`) reserved strictly for token IDs, cryptographic hashes, and Decision Tree rule paths.
5. **Tactile Micro-Interactions**:
   * Hardware-accelerated transitions (150–250ms).
   * Crisp button depression (`active: scale(0.98)`).
   * High-contrast focus rings (`outline: 2px solid #0284C7`) for keyboard accessibility.
   * Never rely on color alone: every risk state pairs color with an icon + explicit text badge.

---

## 3. Master Phase Breakdown

### Phase 1: Environment, Scaffolding & Repository Foundation
* **Goal**: Establish a clean, standardized development environment, directory layout, dependency manifest, logging infrastructure, configuration system, and foundational CSS design tokens.
* **Core Artifacts**:
  * Directory layout (`data/`, `src/`, `database/`, `models_store/`, `tests/`, `templates/`, `static/css/`, `static/js/`).
  * Pinned `requirements.txt` (Flask, Scikit-learn, TensorFlow, PyJWT, qrcode, pyzbar, opencv-python, pytest).
  * `src/utils/config.py` for centralized configuration, paths, and `.env` parsing.
  * `src/utils/logger.py` for structured clinical and security audit logging.
  * `static/css/tokens.css`: Foundational design tokens (colors, typography, spacing, shadows, borders).
  * `templates/base.html`: Accessible base HTML5 shell with semantic landmarks.
  * Baseline smoke tests verifying imports and environment health.
* **Downstream Integration**: Provides shared imports, constants, paths, logging, and UI foundations for all future phases.

### Phase 2: Relational Database Architecture & Seeding Engine
* **Goal**: Construct the complete relational database schema and a clinically realistic 100+ patient seeding engine.
* **Core Artifacts**:
  * `database/schema.sql`: Clinical tables (`patients`, `vitals`, `lab_results`, `medications`, `predictions`, `outcomes`) and Vault tables (`medical_vault`, `vault_access_tokens`, `vault_access_log`).
  * `src/database/db.py`: Connection manager, transaction helper, and foreign-key enforcement.
  * `database/seed_data.py`: Synthesizes 100+ clinically coherent patients across 4 risk tiers, with 7–14 day longitudinal vitals, lab biomarkers, medications, and sample vault records.
* **Downstream Integration**: Provides persistence for the data pipeline (Phase 3), the vault service (Phase 5), and API testing (Phase 6).

### Phase 3: Clinical Data Engineering & Feature Pipeline
* **Goal**: Ingest raw multimodal clinical data, handle missing values, engineer temporal vitals features, scale biomarkers, and produce normalized feature store splits.
* **Core Artifacts**:
  * `src/data/loader.py`: Ingests and joins demographics, longitudinal vitals, and lab results.
  * `src/data/cleaner.py`: Missing value imputation (forward-fill for vitals, median for labs) and physiological outlier clipping.
  * `src/data/feature_engineering.py`: Computes Mean Arterial Pressure (MAP), Shock Index, vitals trend slopes over 72h, glycemic volatility, and cumulative abnormality index.
  * `src/data/scaler.py`: Normalizes features (StandardScaler/RobustScaler) and serializes `models_store/scaler.joblib`.
  * `data/processed/train.parquet` and `data/processed/test.parquet`.
* **Downstream Integration**: Trains the 4 machine learning models in Phase 4 and normalizes incoming live patient data during API inference.

### Phase 4: Machine Learning Intelligence & Explainability Engine
* **Goal**: Train, validate, and serialize the four complementary models and the ensemble reconciler to deliver transparent risk tiering, rule paths, and treatment recommendations.
* **Core Artifacts**:
  * `src/models/kmeans_model.py`: Unsupervised patient clustering into 4 risk tiers (Healthy, At-Risk, High, Critical).
  * `src/models/decision_tree_model.py`: Interpretable classifier generating exact human-readable if-then branch paths.
  * `src/models/knn_model.py`: Similarity matching ($K=5$) over normalized vectors to retrieve historical precedents and protocol success rates.
  * `src/models/neural_network_model.py`: Multi-layer Perceptron capturing complex non-linear multimodal interactions (e.g., sepsis deterioration).
  * `src/models/ensemble.py`: Reconciles model outputs into a unified risk score, risk tier, rule path, similar patient matches, and treatment recommendation.
  * `models_store/`: Serialized model artifacts (`.joblib`, `.h5`).
* **Downstream Integration**: Loaded into memory by the Flask API (Phase 6) to power real-time inference and automated early warning alerts.

### Phase 5: Patient Medical Vault & Cryptographic QR Subsystem
* **Goal**: Build the zero-trust, patient-owned Medical Vault and dynamic QR code sharing system with signed JWT tokens, strict scope restriction, and audit logging.
* **Core Artifacts**:
  * `src/vault/vault_service.py`: CRUD operations for medical history (diagnoses, allergies, medications, surgeries, lab PDFs).
  * `src/vault/qr_generator.py`: Generates HMAC-SHA256 signed JWTs with explicit claims (`patient_id`, `scope`, `exp`, `max_uses`) and renders visual QR code images (Base64 PNGs).
  * `src/vault/qr_scanner.py`: Decodes QR codes via `pyzbar`/`opencv`, validates signature, expiration, revocation, and use quota.
  * `src/vault/access_controller.py`: Filters vault records so provider receives strictly the authorized scope.
  * `src/vault/access_logger.py`: Immutable logging to `vault_access_log` for every scan.
* **Downstream Integration**: Supplies the business logic and security foundation for `/api/vault/*` endpoints in Phase 6.

### Phase 6: RESTful API Layer & Application Services (Flask Backend)
* **Goal**: Expose all clinical intelligence, patient data, alerting, and vault services through a structured, documented REST API.
* **Core Artifacts**:
  * `src/api/app.py`: Flask app factory with error handling, CORS, and model loading at startup.
  * `src/api/routes/patients.py`: Patient CRUD, profile fetching, vitals time-series endpoints.
  * `src/api/routes/predictions.py`: Live inference (`/predict`), similar patients (`/similar`), and aggregate dashboard summary.
  * `src/api/routes/alerts.py`: Automated early-warning alert feed (7–14 day risk window) and doctor acknowledgement.
  * `src/api/routes/vault.py`: Vault record upload, QR generation, token revocation, QR access/verification, and patient audit log.
* **Downstream Integration**: Serves as the single backend engine for the frontend web portals in Phase 7.

### Phase 7: Interactive Presentation Layer & Web Portals
* **Goal**: Build modern, responsive, clinical-grade web interfaces for physicians, patients, and hospital triage adhering to the anti-AI clinical design system.
* **Core Artifacts**:
  * Comprehensive Styling (`static/css/styles.css`): Built on `tokens.css`, high-density clinical data tables, hairline borders, responsive drawer navigation, accessible contrast.
  * Physician Clinical Dashboard (`templates/dashboard.html`, `static/js/dashboard.js`):
    * Patient triage grid with risk filters (*All*, *Critical*, *High*, *Medium*, *Low*).
    * Interactive longitudinal vitals graphs with normal reference ranges (Chart.js).
    * Decision Tree rule path visualizer (step-by-step breadcrumb trail).
    * KNN similar case comparisons and protocol recommendations with success percentage badges.
    * 7–14 day early warning notification banner.
  * Patient Medical Vault Portal (`templates/patient_vault.html`, `static/js/vault.js`):
    * Categorized medical history manager (allergies, medications, lab reports, surgeries).
    * Interactive QR generator modal with scope checkboxes and expiration selectors.
    * Active shared passes list with 1-click **Revoke Access** button.
    * Access audit log showing exact timestamps and provider names.
  * Provider QR Scanner View (`templates/scanner.html`, `static/js/scanner.js`):
    * In-browser camera viewfinder and file uploader for triage and ER doctors.
    * Scoped read-only summary display with tamper-evident cryptographic verification badge.
* **Downstream Integration**: Provides the interactive interface evaluated in Phase 8 and presented to academic faculty.

### Phase 8: End-to-End Testing, Validation & Clinical Benchmarking
* **Goal**: Rigorously test all components and evaluate model metrics against the project targets.
* **Core Artifacts**:
  * Automated Test Suite (`tests/`): Unit and integration tests covering data pipelines, models, vault security, and API endpoints.
  * Security Verification: Tests validating token forgery rejection, expired token rejection, revoked token blocking, and scope leakage prevention.
  * Benchmark Evaluation (`src/evaluation/benchmark.py`): 5-fold cross-validation verifying target metrics:
    * Accuracy $\ge 91\%$
    * Precision $\ge 89\%$
    * Recall $\ge 88\%$
    * F1-Score $\ge 89\%$
  * Single-command demo runner (`run_demo.py`).
* **Downstream Integration**: Generates the final test reports and performance figures for academic submission and evaluation.

---

## 4. Execution Dependency Matrix

| Phase | Direct Prerequisites | Output Artifacts | Primary Verification Check |
|---|---|---|---|
| **Phase 1** | None | Directory tree, `requirements.txt`, `config.py`, `logger.py`, `tokens.css`, `base.html` | Environment smoke tests pass |
| **Phase 2** | Phase 1 | `schema.sql`, `db.py`, `seed_data.py`, `medihaven.db` | 100+ patients seeded; schema integrity verified |
| **Phase 3** | Phase 2 | `loader.py`, `cleaner.py`, `feature_engineering.py`, scaled datasets | Clean feature matrix exported; 0 nulls |
| **Phase 4** | Phase 3 | K-Means, Decision Tree, KNN, Neural Net, Ensemble | Models trained & serialized; inference returns valid dict |
| **Phase 5** | Phase 2 | `vault_service.py`, `qr_generator.py`, `qr_scanner.py`, `access_logger.py` | Security tests pass (tamper, expiry, scope isolation) |
| **Phase 6** | Phase 4, Phase 5 | Flask API (`app.py`, routes, schemas) | All REST endpoints return valid status & payload |
| **Phase 7** | Phase 6 | Physician Dashboard, Patient Vault UI, Provider Scanner | Full interactive browser flows operate end-to-end |
| **Phase 8** | Phase 7 | Automated test suite, 5-fold CV report, `run_demo.py` | Target metrics met (91% Acc, 89% F1); 100% test pass |

---

*This document serves as the master engineering reference for MediHaven across all phases.*
