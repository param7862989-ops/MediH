# Phase 3 — Research-Mode Integration and Security Hardening

MediHaven runs in one of two explicitly selected modes. Nothing switches modes automatically.

| | Synthetic mode (default) | Research mode (`DATA_MODE=mimic`) |
| :--- | :--- | :--- |
| Selection | `DATA_MODE` unset or `synthetic` | `DATA_MODE=mimic` (any other value is a startup error) |
| Startup checks | Warns about weak secrets | **Refuses to start** with default/placeholder/short secrets, a non-loopback `FLASK_HOST`, or the debugger |
| Seeding / training / DB recreation | `run_demo.py` self-heals as before | Never (all entry points refuse; the app never creates the demo database) |
| Synthetic ensemble | Loaded | Never loaded |
| Dashboard, triage, alerts, admin telemetry, narratives | Unchanged | Disabled — APIs return `503 UNAVAILABLE_IN_RESEARCH_MODE`; `/dashboard` and `/admin` redirect to `/research` |
| Patient vault, QR sharing, scanner | Unchanged | Unchanged — still the synthetic demo; MIMIC data never enters them |
| Research view | `/research` → 404, `/api/research/*` → 409 | `/research` and `/api/research/*` for physician/admin |
| External AI (Gemini) | De-identified prompts, TLS verified | Never called |

## 1. Configuration and safe startup

```powershell
# Synthetic demo (unchanged)
.\.venv\Scripts\python.exe run_demo.py

# Research mode: requires strong secrets in .env (never the public defaults from the source code)
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(32))"   # run twice
#   put the two values in .env as FLASK_SECRET_KEY=... and JWT_SECRET_KEY=...
#   (a new JWT_SECRET_KEY invalidates previously issued QR passes)
$env:DATA_MODE = "mimic"
$env:MIMIC_OUTPUT_DIR = "C:\research\medihaven-output"
.\.venv\Scripts\python.exe run_demo.py
Remove-Item Env:DATA_MODE   # back to synthetic mode
```

Prerequisites for research inference: the Phase 1 dataset (v1.2.0) and Phase 2 run
(`medihaven_t1_phase2_baselines_v2.0.0`) in `MIMIC_OUTPUT_DIR`. Without them the app still starts,
but research inference stays disabled and the reason is shown. The demo vault in research mode
needs the existing demo database; it is never created in research mode.

Optional settings: `RESEARCH_ARTIFACT_MANIFEST_SHA256` (pin), `CORS_ORIGINS`,
`SESSION_COOKIE_SECURE`, `MAX_REQUEST_MB` (see `.env.example`).

## 2. Research view and API

- `GET /research` — research dashboard in the same style as the synthetic triage dashboard (KPI tiles,
  Chart.js charts, tables), built from cohort-level aggregates only: eligible stays, deaths and death
  rate, model AUROC and average precision with 95% CIs, cohort flow, measurement coverage, quality
  filtering, anomaly sensitivity analysis, fold-to-fold AUROC, feature weights of the final models,
  a metrics table, confusion matrices and an interactive what-if estimator. No individual patient is
  shown. The page script inserts text with `textContent` only and uses no browser storage.
- `GET /api/research/dashboard` — the aggregate data behind those charts (clinical roles).
- `GET /api/research/status`, `GET /api/research/model-card` — the same information as JSON.
- `POST /api/research/predict` — "what-if" risk estimates for manually entered summaries (contract below).

### What-if input contract (`src/training/whatif.py`)

Example: `{"features": {"heart_rate_count": 6, "heart_rate_min": 95, "heart_rate_max": 128,
"heart_rate_mean": 112, "heart_rate_last": 120, "lactate_count": 1, "lactate_last": 4.1, "age": 71}}`

- **Variables you do not supply are not measured**: `<var>_missing = 1`, `<var>_count = 0`, summaries
  empty — the same encoding Phase 1 uses for unmeasured stays (verified against Phase 1's aggregation
  in tests). The model's training-median imputer fills those empty summaries exactly as during training.
  Nothing is assumed to have been measured.
- **Measured variables need `<var>_count` (whole number ≥ 1).** With count 1, give at least one of
  min/max/mean/last; all given values must be equal and the rest are set to that single reading. With
  count ≥ 2, all four summaries are required and must satisfy min ≤ mean ≤ max and min ≤ last ≤ max.
  `<var>_missing` may be omitted (0) or 0. Values must be within the feature spec's plausible range
  (e.g. temperature in °C, 25–45).
- `bp_source_invasive` (0/1) is required when sbp or dbp is measured and must be omitted otherwise;
  `map_derived` (0/1) likewise for map. `age` (18–120) and `sex_male` (0/1) are optional; if omitted they
  are imputed with the training median and listed in the response.
- Rejected with 422: forbidden names (identifiers, labels, anomaly flag, index time, risk tiers…) →
  `FORBIDDEN_FEATURE`; unknown names → `UNKNOWN_FEATURE`; non-numeric, non-finite, out-of-range,
  non-binary indicators or invalid counts → `INVALID_VALUE`; missing counts/summaries/flags →
  `INCOMPLETE_VARIABLE`; contradictions (e.g. a positive count with `_missing = 1`, values with
  `_count = 0`, unequal values for a single reading) → `CONTRADICTORY_INPUT`.
- The response's `input_summary` lists measured and unmeasured variables, imputed demographics and the
  number of median-imputed features.
- **Limitations:** manually entered summaries are not checked against any patient record, the models
  were trained on 107 demo stays, and estimates are research outputs only — not clinical advice.

The application never reads the patient-level research dataset. It loads only the verified model
artifacts and aggregate metrics, so no MIMIC record or identifier can reach a response, log, page or
browser storage. Per-stay predictions over MIMIC patients are deliberately not offered (see §6).

## 3. Artifact verification (fail closed)

`src/training/artifacts.py` runs before any model is used:

1. `MIMIC_OUTPUT_DIR` is outside the repository and OneDrive.
2. `run_manifest.json` is valid JSON of the expected shape, matches the optional pin, and names run
   `medihaven_t1_phase2_baselines` v2.0.0, dataset v1.2.0, task `t1_inhospital_mortality_after_icu24h`
   v3, feature spec `mimic-iv-t1-features-v1` and **the installed scikit-learn version** (exact match;
   otherwise reinstall the recorded version or retrain).
3. The training dataset hash equals the current Phase 1 dataset manifest.
4. Every required file (metrics, feature config, the reference baseline and the four models) is listed and stays inside the run
   folder. **Each file is read once, its SHA-256 is checked on those bytes, and only those exact bytes
   are parsed or deserialized** — there is no second read between check and load. joblib uses pickle,
   which can execute code, so unverified bytes are never deserialized. A scikit-learn version warning
   raised while loading is treated as a failure.
5. Each model bundle declares exactly the 74 model inputs (the feature spec minus the 14 measurement
   counts, re-checked by the Phase 2 forbidden-name guard; any other exclusion fails verification), the same dataset/task versions and hash, and a fitted pipeline whose inputs are those
   features. `metrics.json` must have the structure the research endpoints read.

Any failure disables research inference with a reason code while the rest of the app keeps running:
`NOT_CONFIGURED`, `UNSAFE_LOCATION`, `ARTIFACTS_MISSING`, `MANIFEST_PIN_MISMATCH`, `MANIFEST_INVALID`,
`VERSION_MISMATCH`, `SKLEARN_VERSION_MISMATCH`, `DATASET_MISMATCH`, `MANIFEST_INCOMPLETE`,
`UNSAFE_MANIFEST_ENTRY`, `ARTIFACT_MISSING`, `ARTIFACT_UNREADABLE`, `ARTIFACT_HASH_MISMATCH`,
`FEATURE_SCHEMA_MISMATCH`, `FEATURE_CONFIG_INVALID`, `FORBIDDEN_FEATURE`, `BUNDLE_UNLOADABLE`,
`BUNDLE_INVALID`, `METRICS_INVALID`, or `VERIFICATION_ERROR` for anything unexpected.

**Residual risk — manifest pin.** Without `RESEARCH_ARTIFACT_MANIFEST_SHA256`, hashes detect corruption
and mismatches but not deliberate replacement: anyone who can write to `MIMIC_OUTPUT_DIR` could replace
both a model and the manifest and get code executed when research mode starts. The pin is optional (the
training workflow regenerates the manifest on every run), so research mode logs a `SECURITY` warning and
the research page shows a note when it is unset. After each training run, set the pin to the new
manifest's SHA-256 (e.g. `Get-FileHash ...\run_manifest.json -Algorithm SHA256`).

What-if inputs are validated: identifiers, labels, the anomaly flag, index time, risk tiers and any
other forbidden name → `422 FORBIDDEN_FEATURE`; unknown names → `UNKNOWN_FEATURE`; non-numeric,
boolean or non-finite values → `INVALID_VALUE`. Supplied values are never logged.

## 4. External AI (Gemini)

- TLS certificate and host-name verification are always on (the previous `CERT_NONE` setting was removed).
- The API key is sent in the `x-goog-api-key` header, not the URL; it, error bodies and payloads are never logged.
- Prompts contain only coarse model outputs (risk tier, risk rounded to 10%, generic protocol label).
  No name, MRN, age, sex, ward, symptoms, diagnosis or measurements are sent; the patient is
  `[PATIENT]`, replaced by the name locally after the response.
- Research mode never calls an external AI service; narrative endpoints are disabled and the
  summarizer itself refuses external calls (defence in depth).
- Admin telemetry shows at most the last 4 characters of a configured key.

## 5. Other hardening

| Finding | Change |
| :--- | :--- |
| `.env` used the public default `FLASK_SECRET_KEY`/`JWT_SECRET_KEY` (session and QR-pass forgery) | Research mode refuses default/placeholder/short secrets; synthetic mode logs a SECURITY warning. The local secrets were rotated (QR passes issued earlier no longer validate). The test session pins the public development values and never reads local secrets. |
| Undocumented weak alternate passwords (`physician123`, `patient123`) | Removed; exact, constant-time comparison |
| CORS `*` on all APIs | Same-origin by default; explicit allowlist via `CORS_ORIGINS` |
| No request size limit (large JSON / base64 images) | `MAX_CONTENT_LENGTH` (8 MB default) with a JSON 413 response |
| Session cookie flags implicit | `HttpOnly`, `SameSite=Lax`, optional `Secure` |
| `python -m src.api.app` enables the debugger by default | Never in research mode |
| The `flask` CLI chooses the bind address and debugger outside the app factory | Research mode refuses to start under the Flask CLI (detected via `FLASK_RUN_FROM_CLI`, also flagging `FLASK_DEBUG`) and rejects a non-loopback `FLASK_RUN_HOST`. Supported research launchers: `run_demo.py` and `python -m src.api.app`, which enforce `FLASK_HOST` and debugger-off. Other WSGI servers must be bound to loopback by the operator — the app cannot verify their bind address. Synthetic mode is unaffected. |
| The vault profile showed a synthetic risk score in research mode | `GET /api/patients/<id>` returns `latest_prediction: null` in research mode; the vault page shows no risk badge without a prediction |
| SQLite connect creates missing databases | Research mode opens the demo database read-write only if it exists |

## 6. Limitations and open items

- Demo authentication (hard-coded accounts) is not production authentication; research endpoints
  expose only aggregates and what-if estimates.
- Research results come from the MIMIC-IV Demo (11 positive stays): not clinically validated, not for
  patient-care decisions.
- Per-stay MIMIC predictions in the dashboard are not implemented: the dashboard is patient-centric
  and would display patient-level research data. Doing so needs a decision on display identifiers,
  access control and data-use-agreement compliance first.
- Not changed (outside the integration scope): the QR scanner's `accessed_by` name is client-supplied,
  and failed scans are stored with mapped audit statuses.
