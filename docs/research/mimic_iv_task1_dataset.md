# MIMIC-IV Research Dataset — Phase 1 (Task 1)

This guide covers the research data pipeline added in Phase 1. It is separate from the
synthetic demo: it does not change the website, the synthetic database, or the trained
synthetic models, and it never writes MIMIC-derived data into the patient vault.

**Status:** pipeline and fixture tests only. No model is trained on MIMIC-IV data yet, and no
performance claim can be made from this phase.

## 1. Data safety rules

- MIMIC-IV source files and every generated dataset stay **outside this repository and outside
  OneDrive**. The builder refuses such locations (`src/utils/research_paths.py`).
- The builder prints and logs **aggregate counts only** (cohort flow, prevalence, coverage). It
  never prints rows, identifiers, timestamps or individual labels.
- Everything the builder writes stays under the output folder: the dataset, the manifest, the
  research log (`<output-dir>\logs\build_mimic_t1.log`) and DuckDB's temporary spill folder
  (`<output-dir>\.duckdb_tmp`, removed after each build). It writes nothing to the repository's
  `logs\`. (DuckDB's default spill folder is `.tmp` in the current working directory, so the
  reader always overrides it and verifies the setting.) The resolved paths are re-checked at
  runtime and recorded in the manifest under `configuration.runtime_paths`.
- The dataset contains source identifiers (`src_subject_id`, `src_hadm_id`, `src_stay_id`) for
  local joins and patient-level splitting only. Do not share the dataset or publish row-level
  content. Display identifiers for the web UI will be separate surrogate keys (later phase).
- Credentialed MIMIC-IV requires PhysioNet credentialing and its data use agreement; do not send
  records to online services (including LLM APIs).
- `DATA_MODE=mimic` blocks synthetic seeding, synthetic training, synthetic predictions, the web
  app and `run_demo.py`, so research mode can never silently fall back to synthetic data.

## 2. Setup (Windows PowerShell)

1. Download the open-access **MIMIC-IV Clinical Database Demo** from PhysioNet
   (https://physionet.org/content/mimic-iv-demo/) and extract it under a local folder such as
   `C:\research\mimic-iv-demo` (the folder containing `hosp\` and `icu\`, or its parent holding a
   single extracted release, e.g. `C:\research\mimic-iv-demo\mimic-iv-clinical-database-demo-2.2`).
   Optionally verify the download: run `sha256sum -c SHA256SUMS.txt` inside the release folder.
2. Install dependencies into the project environment (adds DuckDB):

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Build the dataset:

   ```powershell
   .\.venv\Scripts\python.exe -m src.tasks.build_mimic_t1 `
       --mimic-dir C:\research\mimic-iv-demo `
       --output-dir C:\research\medihaven-output
   ```

   Or set `MIMIC_IV_DIR` / `MIMIC_OUTPUT_DIR` (environment or `.env`) and run
   `.\.venv\Scripts\python.exe -m src.tasks.build_mimic_t1`. Add `--overwrite` to rebuild an
   existing dataset of the same version.

Output: `<output-dir>\medihaven_t1_inhospital_mortality_icu24h_v1.2.0\dataset.parquet` and
`manifest.json` (task definition, source release and file sizes, item mapping, exclusion
counts, outcome anomalies, prevalence, per-variable quality counts, runtime paths, content hash
and software versions), plus the research log `<output-dir>\logs\build_mimic_t1.log`.

## 3. Task 1 definition (task version 3, dataset version 1.2.0)

| Element | Rule |
| :--- | :--- |
| Unit | First ICU stay per hospital admission (earliest `intime`, ties by `stay_id`) |
| Population | Age at admission >= 18 (`anchor_age + admission year − anchor_year`) |
| Index time | ICU `intime` + exactly 24 h |
| Eligibility | Alive at the index time; ICU `outtime` > index time; hospital `dischtime` > index time (checked before labeling) |
| Observation window | `[intime, index_time]`; `charttime <= index_time`; reliable `storetime <= index_time` |
| Positive (consistent) | `hospital_expire_flag = 1`, `index_time < deathtime <= dischtime`, discharge location `DIED` or not recorded |
| Positive (timestamp anomaly) | `hospital_expire_flag = 1`, discharge location `DIED`, `deathtime > index_time` and `deathtime > dischtime` — labeled positive **and flagged** |
| Negative | `hospital_expire_flag = 0`, no `deathtime`, `dischtime > index_time`, discharge location not `DIED`, no `dod` before the discharge date |
| Ambiguous | Any other disagreement between the outcome fields is excluded and counted by reason (below) |
| Horizon | Remainder of the index hospitalization — **not** a fixed 7- or 30-day outcome |

**Timestamp anomalies.** When the expire flag and the `DIED` disposition both record an
in-hospital death but `deathtime` is later than `dischtime`, the stay is labeled positive because
two independent fields agree on the outcome and only the timestamps conflict. The discrepancy is
never hidden: each such stay has `outcome_anomaly_deathtime_after_discharge = 1` in the dataset
(label metadata — outcome-derived, never a model feature), and the manifest
(`aggregate_counts.outcome_anomalies`), research log and printed summary report the count and the
distribution of the `deathtime − dischtime` gap in buckets (0–1 h, 1–6 h, 6–24 h, > 24 h). No
upper bound is imposed on the gap. The anomaly rule never overrides eligibility: a recorded
`dischtime` at or before the index time still excludes the stay.

**Conflict rule (still excluded).** Ambiguity reasons, checked in this order:
`missing_discharge_time`, `missing_expire_flag`, `expire_flag_without_deathtime`,
`deathtime_without_expire_flag`, `deathtime_after_discharge_without_died_location` (flag 1 and
`deathtime` later than `dischtime`, but the disposition is not `DIED`),
`expire_flag_with_alive_discharge_location`, `discharge_location_died_but_flag_alive`,
`date_of_death_before_alive_discharge`. Alive discharges stay negative: a death after a documented
alive discharge (`dod` on or after the discharge date) is an out-of-hospital death, and a recorded
`deathtime` after an alive discharge (flag 0) is excluded as contradictory.

Version history: v1 accepted deaths up to 24 h after `dischtime` as positive; v2 excluded every
`deathtime` later than `dischtime`; v3 (current) labels them positive only with flag 1 + `DIED`,
as flagged anomalies.

Exclusions are applied in a fixed order; each stay is counted under the first rule it fails
(see `src/tasks/mortality_icu_24h.py`).

## 4. Feature spec `mimic-iv-t1-features-v1`

Item IDs are confirmed against `d_items` / `d_labitems` at build time: an item missing from the
dictionary is excluded with a warning; a dictionary label that contradicts the expected concept
stops the build.

| Variable | Source item IDs | Canonical unit | Plausible range |
| :--- | :--- | :--- | :--- |
| heart_rate | chartevents 220045 | bpm | 10–300 |
| resp_rate | chartevents 220210, 224690 | breaths/min | 2–80 |
| spo2 | chartevents 220277 | % | 50–100 |
| temperature | chartevents 223761 (°F → °C), 223762 (°C) | °C | 25–45 |
| sbp | chartevents 220050 (arterial), 220179 (non-invasive) | mmHg | 30–300 |
| dbp | chartevents 220051 (arterial), 220180 (non-invasive) | mmHg | 10–200 |
| map | chartevents 220052 (arterial), 220181 (non-invasive), else derived | mmHg | 20–250 |
| creatinine | labevents 50912 | mg/dL | 0.1–30 |
| bun | labevents 51006 | mg/dL | 1–300 |
| wbc | labevents 51301, 51300 | K/uL | 0.1–500 |
| hemoglobin | labevents 51222 | g/dL | 2–25 |
| platelets | labevents 51265 | K/uL | 1–2000 |
| lactate | labevents 50813 | mmol/L | 0.1–30 |
| glucose | labevents 50931, 50809 | mg/dL | 10–2000 |

Per variable: `_min`, `_max`, `_mean`, `_last` (latest by charttime), `_count`, `_missing`.
Plus `map_derived`, `bp_source_invasive`, `age`, `sex_male`. Missing values stay missing (NaN);
nothing is imputed with normal values.

Quality rules (in order): window → storetime availability → numeric value → accepted unit (a
missing unit is accepted because the item ID defines it) → unit conversion → plausibility →
blood-pressure source (per stay: invasive if any arterial systolic/diastolic reading exists,
otherwise non-invasive) → MAP (measured, else derived as DBP + (SBP − DBP)/3 from same-time pairs
with SBP > DBP). Labs are attached by person and time window because `labevents` has no
`stay_id` and often no `hadm_id`.

## 5. Validation on the MIMIC-IV Demo 2.2

Verified on the real demo with dataset v1.2.0 / task v3 (aggregate results only):

- All 21 mapped item IDs confirmed by `d_items` / `d_labitems` with no warnings; no unit
  mismatches or missing units.
- 140 ICU stays → 107 eligible stays: 11 positive (7 consistent + 4 timestamp anomalies, all
  with a 6–24 h gap), 96 negative, 0 ambiguous; prevalence 10.3%.
- An independent SQL re-implementation of the cohort and label rules reproduced every count.
- Compared with v1.1.0 (task v2), the only change is the 4 anomaly stays added as flagged
  positives; the other 103 stays have identical labels and features.

The 4 anomalies are 4 of 11 positives, so any later analysis should report a sensitivity check
that excludes rows with `outcome_anomaly_deathtime_after_discharge = 1`.

The demo (100 patients) is only for pipeline validation. Its counts are far too small for any
model performance estimate.

## 6. Tests

Fixture-based (hand-crafted, no real records): `tests/test_mimic_layout_mappings.py`,
`tests/test_mimic_t1_dataset.py`, plus `tests/test_data_protection.py` and
`tests/test_datasource_synthetic.py`. The test session uses temporary paths only and fails if
the persistent demo database or model artifacts change.
