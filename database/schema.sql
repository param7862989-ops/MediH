-- ==============================================================================
-- MediHaven Master Database Schema
-- Multi-domain Relational Store: Clinical Intelligence & Patient Medical Vault
-- Optimized for SQLite (Development) and MySQL (Production Deployment)
-- ==============================================================================

-- Enable Foreign Key constraints for SQLite sessions
PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------------------------
-- 1. CLINICAL INTELLIGENCE DOMAIN
-- ------------------------------------------------------------------------------

-- Master Patients Table
CREATE TABLE IF NOT EXISTS patients (
    patient_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    mrn             TEXT UNIQUE NOT NULL,               -- Medical Record Number (e.g., MRN-2026-0042)
    full_name       TEXT NOT NULL,
    age             INTEGER NOT NULL CHECK(age >= 0 AND age <= 130),
    gender          TEXT NOT NULL CHECK(gender IN ('M', 'F', 'Other')),
    admission_date  DATETIME NOT NULL,
    discharge_date  DATETIME,                           -- Nullable if currently admitted
    status          TEXT NOT NULL DEFAULT 'Admitted' 
                    CHECK(status IN ('Admitted', 'Observation', 'ICU', 'Discharged')),
    ward            TEXT NOT NULL,                      -- e.g., General Ward A, Cardiac Care, ICU, Pulmonology
    bed_number      TEXT NOT NULL                       -- e.g., Bed-104, ICU-08
);

-- Longitudinal Vitals (High-frequency time-series)
CREATE TABLE IF NOT EXISTS vitals (
    vital_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id         INTEGER NOT NULL,
    recorded_at        DATETIME NOT NULL,
    heart_rate         REAL NOT NULL CHECK(heart_rate >= 20 AND heart_rate <= 260),            -- bpm
    blood_pressure_sys REAL NOT NULL CHECK(blood_pressure_sys >= 40 AND blood_pressure_sys <= 300), -- mmHg
    blood_pressure_dia REAL NOT NULL CHECK(blood_pressure_dia >= 20 AND blood_pressure_dia <= 200), -- mmHg
    temperature        REAL NOT NULL CHECK(temperature >= 30.0 AND temperature <= 45.0),       -- Celsius
    oxygen_saturation  REAL NOT NULL CHECK(oxygen_saturation >= 50.0 AND oxygen_saturation <= 100.0), -- % SpO2
    respiratory_rate   REAL NOT NULL CHECK(respiratory_rate >= 4 AND respiratory_rate <= 70),  -- breaths/min
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

-- Laboratory Biomarkers & Diagnostic Tests
CREATE TABLE IF NOT EXISTS lab_results (
    lab_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    recorded_at     DATETIME NOT NULL,
    glucose         REAL NOT NULL CHECK(glucose >= 10 AND glucose <= 1000),      -- mg/dL
    creatinine      REAL NOT NULL CHECK(creatinine >= 0.1 AND creatinine <= 25),  -- mg/dL
    cholesterol     REAL CHECK(cholesterol >= 50 AND cholesterol <= 600),        -- mg/dL
    wbc_count       REAL NOT NULL CHECK(wbc_count >= 0.5 AND wbc_count <= 100),  -- x10^3/uL
    bun             REAL CHECK(bun >= 1 AND bun <= 200),                         -- Blood Urea Nitrogen (mg/dL)
    hemoglobin      REAL CHECK(hemoglobin >= 2.0 AND hemoglobin <= 25.0),        -- g/dL
    platelets       REAL CHECK(platelets >= 10 AND platelets <= 1500),           -- x10^3/uL
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

-- Active and Historical Medications
CREATE TABLE IF NOT EXISTS medications (
    medication_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    drug_name       TEXT NOT NULL,
    dosage          TEXT NOT NULL,                      -- e.g., '500mg', '10mg/day'
    route           TEXT NOT NULL DEFAULT 'Oral' CHECK(route IN ('Oral', 'IV', 'SubQ', 'Inhalation', 'Topical')),
    frequency       TEXT NOT NULL DEFAULT 'QD',         -- QD, BID, TID, QID, PRN
    start_date      DATETIME NOT NULL,
    end_date        DATETIME,                           -- Nullable if ongoing
    is_active       BOOLEAN NOT NULL DEFAULT 1,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

-- Model Prediction History (Audit Log for Machine Learning Inferences)
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id           INTEGER NOT NULL,
    predicted_at         DATETIME NOT NULL,
    risk_tier            TEXT NOT NULL CHECK(risk_tier IN ('Low', 'Medium', 'High', 'Critical')),
    risk_score           REAL NOT NULL CHECK(risk_score >= 0.0 AND risk_score <= 1.0),
    model_used           TEXT NOT NULL,                 -- e.g., 'Ensemble', 'DecisionTree', 'KNN', 'KMeans'
    decision_tree_path   TEXT,                          -- JSON array of human-readable decision rules
    similar_patient_ids  TEXT,                          -- JSON array of top-K matched patient IDs
    recommended_protocol TEXT,                          -- e.g., 'Protocol C (Cardioprotective bundle)'
    readmission_30d_risk REAL CHECK(readmission_30d_risk >= 0.0 AND readmission_30d_risk <= 1.0),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

-- Ground Truth Outcomes (Used for model training, cross-validation, and performance metrics)
CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          INTEGER NOT NULL,
    primary_diagnosis   TEXT NOT NULL,
    readmitted_30d      BOOLEAN NOT NULL DEFAULT 0,
    icu_transfer        BOOLEAN NOT NULL DEFAULT 0,
    outcome_date        DATETIME NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

-- ------------------------------------------------------------------------------
-- 2. PATIENT MEDICAL VAULT & QR-BASED ACCESS SHARING DOMAIN
-- ------------------------------------------------------------------------------

-- Patient-Owned Medical Vault Records
CREATE TABLE IF NOT EXISTS medical_vault (
    vault_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    category        TEXT NOT NULL CHECK(category IN (
                        'diagnosis', 'allergy', 'medication', 
                        'lab_report', 'imaging', 'surgery', 'vaccination'
                    )),
    title           TEXT NOT NULL,
    description     TEXT,
    file_path       TEXT,                               -- Optional path to uploaded document/scan
    structured_data TEXT,                               -- JSON metadata for category-specific attributes
    uploaded_at     DATETIME NOT NULL,
    is_sensitive    BOOLEAN NOT NULL DEFAULT 0,         -- Flags sensitive items requiring explicit scope
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

-- Vault Access Tokens (QR-backed, scoped, cryptographically signed access grants)
CREATE TABLE IF NOT EXISTS vault_access_tokens (
    token_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    token_hash      TEXT UNIQUE NOT NULL,               -- Unique token lookup hash
    scope_json      TEXT NOT NULL,                      -- JSON array e.g., '["allergy", "medication"]' or '["*"]'
    issued_at       DATETIME NOT NULL,
    expires_at      DATETIME NOT NULL,
    max_uses        INTEGER,                            -- Nullable, 1 for single-use, NULL for unlimited
    use_count       INTEGER NOT NULL DEFAULT 0,
    revoked         BOOLEAN NOT NULL DEFAULT 0,
    revoked_at      DATETIME,
    qr_payload      TEXT NOT NULL,                      -- Signed JWT encoded into the QR code
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

-- Immutable Vault Access Audit Log (Full audit trail of every scan/access attempt)
CREATE TABLE IF NOT EXISTS vault_access_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    token_id        INTEGER NOT NULL,
    patient_id      INTEGER NOT NULL,
    accessed_by     TEXT NOT NULL,                      -- Doctor name, clinic ID, or scanner terminal ID
    access_status   TEXT NOT NULL CHECK(access_status IN ('GRANTED', 'EXPIRED', 'REVOKED', 'SCOPE_MISMATCH')),
    accessed_at     DATETIME NOT NULL,
    ip_address      TEXT,
    user_agent      TEXT,
    accessed_scope  TEXT,                               -- Categories retrieved in this access event
    FOREIGN KEY (token_id) REFERENCES vault_access_tokens(token_id) ON DELETE CASCADE,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

-- ------------------------------------------------------------------------------
-- 3. COMPOSITE INDEXES (Optimized for High-Throughput & Low Latency)
-- ------------------------------------------------------------------------------

-- Longitudinal Vitals & Labs Time-series Queries (Dashboard sparklines & 72h trend charts)
CREATE INDEX IF NOT EXISTS idx_vitals_patient_recorded ON vitals(patient_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_labs_patient_recorded ON lab_results(patient_id, recorded_at DESC);

-- Patient lookup by Status and Ward (Triage list filters)
CREATE INDEX IF NOT EXISTS idx_patients_status_ward ON patients(status, ward);

-- Scoped Vault Retrieval (Fast category filtering when doctor scans QR)
CREATE INDEX IF NOT EXISTS idx_vault_patient_category ON medical_vault(patient_id, category);

-- Active QR Token Validation (Instant lookup for valid, unexpired, unrevoked tokens)
CREATE INDEX IF NOT EXISTS idx_tokens_active ON vault_access_tokens(token_hash, revoked, expires_at);
CREATE INDEX IF NOT EXISTS idx_tokens_patient ON vault_access_tokens(patient_id, revoked);

-- Audit Log Inspection (Chronological access trail for patients)
CREATE INDEX IF NOT EXISTS idx_access_log_patient ON vault_access_log(patient_id, accessed_at DESC);
CREATE INDEX IF NOT EXISTS idx_access_log_token ON vault_access_log(token_id, accessed_at DESC);
