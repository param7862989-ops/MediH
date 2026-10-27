# Phase 2 — Task 1 Model Training and Evaluation (research only)

> These are research baselines trained and evaluated on the **MIMIC-IV Clinical Database Demo**.
> The results validate the pipeline; they are **not** evidence of clinical validity. The models are
> not clinically validated, not production-ready and must not be used for patient-care decisions.

## 1. Task and versions

- **Task:** in-hospital mortality among adult ICU stays, predicted at **ICU intime + 24 hours**
  using only information available at that time (`t1_inhospital_mortality_after_icu24h`, task v3).
- **Dataset:** `medihaven_t1_inhospital_mortality_icu24h` **v1.2.0**, feature spec
  `mimic-iv-t1-features-v1` (see [mimic_iv_task1_dataset.md](mimic_iv_task1_dataset.md)).
- Training is pinned to these versions. `src/training/dataset.py` refuses to run if the manifest
  version, task, feature spec, column layout, labels (0/1, no missing), anomaly rows (must be
  positive), one-row-per-stay rule or the dataset content hash do not match.

## 2. Features and leakage protection

- **Model inputs (74, model set v2):** the feature-spec columns — per-variable min/max/mean/last/missing
  over `[intime, index_time]` (with the Phase 1 storetime rule), `map_derived`,
  `bp_source_invasive`, `age` (known at admission) and `sex_male`. Each has a recorded provenance in
  `feature_config.json`; a feature whose availability at the prediction time cannot be established
  is excluded and listed.
- **Never inputs (excluded by name, not position):** `src_subject_id`, `src_hadm_id`,
  `src_stay_id`, `index_time`, `label_inhospital_mortality`,
  `outcome_anomaly_deathtime_after_discharge`, `dataset_version`, `feature_spec_version`.
- **Name guard:** any column whose name suggests labels, anomalies, mortality, death, `dod`,
  discharge, expire flags, outcomes, identifiers (`src_`, subject, hadm, stay_id, `_id`), index time,
  risk tiers/scores, predictions, ICU outtime, length of stay or versions stops the run
  (`ForbiddenFeatureError`). No synthetic risk tiers, scores or previous predictions are used.
- **Measurement counts are not model inputs (since model set v2.0.0):** the 14 `<variable>_count`
  columns stay in the dataset but are excluded by `select_features` and listed with their reason in
  `feature_config.json`. A count reflects how intensively a patient was monitored (a care process
  partly encoding clinicians' judgement) rather than physiology, and a manually entered what-if
  input cannot supply a realistic one. Missingness remains visible through `<variable>_missing`.
  The what-if contract still uses counts to tell measured from unmeasured variables.
- Note: `bp_source_invasive` likewise reflects monitoring practice, which matters when interpreting
  any model.

## 3. Running

```powershell
# Train and evaluate (reads the v1.2.0 dataset, writes a new versioned folder)
.\.venv\Scripts\python.exe -m src.training.train_mimic_t1 --research-dir C:\research\medihaven-output

# Re-run and replace this version's outputs
.\.venv\Scripts\python.exe -m src.training.train_mimic_t1 --research-dir C:\research\medihaven-output --overwrite

# Tests (synthetic fixtures only; temporary paths)
.\.venv\Scripts\python.exe -m pytest tests/test_training_t1.py -q
```

Options: `--dataset-dir`, `--seed` (default 42), `--folds` (default 5), `--bootstrap` (default 2000).
`--research-dir` defaults to `MIMIC_OUTPUT_DIR`. Locations inside the repository or OneDrive are refused.

Outputs, in `<research-dir>\medihaven_t1_phase2_baselines_v3.0.0\`: `models\*.joblib` (pipelines
refitted on all eligible stays, each bundled with its features, dataset/task versions, dataset content
hash and seed), `metrics.json`, `feature_config.json`, `split_metadata.json`, `report.md` and
`run_manifest.json` (versions, seed, timestamp, software, command and SHA-256 of every output file).
The aggregate log is `<research-dir>\logs\train_mimic_t1.log`. No patient-level predictions or
identifiers are written. Outputs are built in a staging folder and swapped into place only on success.
If that final swap fails (e.g. a Windows file lock), the previous run is restored, the completed new run
is kept in `<run folder>.staging`, and the command exits with an error; no complete run is deleted
during cleanup. The next invocation first restores any run orphaned by an interrupted swap.

## 4. Models

Fixed hyperparameters chosen a priori for a very small dataset; **nothing is tuned** on evaluation data.
All learned preprocessing (median imputation, zero-variance filtering, scaling) is inside each
scikit-learn Pipeline and is therefore fitted on training folds only.

| Model | Configuration |
| :--- | :--- |
| `dummy_prior` | predicts the training prevalence — a **reference line**, not one of the four models |
| `logistic_regression` | median imputation → drop constant → standardize → L2 logistic regression, C = 0.1 |
| `decision_tree` | median imputation → drop constant → **distilled tree**: a random forest (300 trees, depth ≤ 4, ≥ 3 stays per leaf) is fitted on the training rows, and one regression tree (depth ≤ 4, ≥ 5 stays per leaf) learns each row's out-of-bag forest risk. Leaf risks are kept within 0.1–99.9 % |
| `random_forest` | median imputation → drop constant → 300 trees, depth ≤ 4, ≥ 3 stays per leaf |
| `neural_network` | median imputation → drop constant → standardize → MLP, one hidden layer of 16 ReLU units, L2 α = 1.0, L-BFGS |

Model set history (all runs kept on disk):
- **v1.0.0:** 88 inputs including counts; three models plus the reference; a plain decision tree whose
  zero-death leaves output exactly 0 %.
- **v2.0.0:** counts excluded; the neural network added; the decision tree's leaf risks smoothed with
  the m-estimate. The tree still had only a few distinct outputs and AUROC 0.56.
- **v3.0.0 (current):** only the decision tree changed. Grown directly on 11 deaths, a single tree is
  unstable. The distilled tree instead imitates the forest's out-of-bag risk estimates: its rules stay
  readable, its outputs are graded, and its splits are steadier.

Two candidate fixes for the tree were evaluated on the same folds before choosing:
- a tree whose depth, leaf size and class weighting were tuned by nested inner cross-validation
  (AUROC 0.48, rejected);
- the distilled tree (AUROC 0.69, adopted).

All of these changes were decided after seeing earlier results on the same data, so differences between
versions are not independent evidence of improvement, and the v3 tree estimate is slightly optimistic.

K-Means is not used: it is not a supervised mortality predictor.

## 5. Evaluation design and its limitations

- **Split:** subject-grouped, stratified 5-fold cross-validation (`StratifiedGroupKFold`, seed 42).
  All stays of a subject are in the same fold, so no subject is in both training and test data. Every
  stay is predicted once, by a model that never saw that subject; metrics use the **pooled
  out-of-fold predictions**.
- **Why not a holdout set:** with 11 positive stays, a 20% holdout would contain about 2 deaths,
  which cannot support any metric. Cross-validation uses every stay for testing exactly once.
- **Safety checks:** the run stops (no fallback to an ungrouped or unstratified split) if positives or
  negatives are fewer than the folds, or if any training/test part of a fold would hold one class.
- **Status flag:** evaluations with fewer than 100 positive stays are labelled `limited`.
- **Limitations:** no external or temporal validation; tiny positive count; per-fold estimates vary
  widely; the saved models are refitted on all stays and their expected performance is the CV estimate.

## 6. Metrics

- **AUROC**; **average precision** = `sklearn.metrics.average_precision_score` (step-wise; not a
  trapezoidal PR-AUC; its no-skill value is the prevalence); **Brier score**.
- **Thresholded metrics** (sensitivity, specificity, precision, F1, confusion matrix): a stay is
  predicted positive when its risk is ≥ the outcome prevalence of the training folds that produced it.
- **Undefined metrics** (e.g. AUROC with one class, precision with no predicted positives, F1 when
  precision and sensitivity are both 0) are reported as null with the reason — never as 0.
- **95% intervals:** percentile intervals from 2,000 subject-level bootstrap resamples of the pooled
  out-of-fold predictions. They capture case-mix sampling only, not model-refitting variability, so
  they understate the true uncertainty.
- **Prior-only baseline caveat:** pooling folds gives it an AUROC that differs from 0.5 (each fold
  predicts its own training prevalence); its per-fold AUROC is exactly 0.5.

## 7. Timestamp-anomaly sensitivity analysis

The 4 stays flagged `outcome_anomaly_deathtime_after_discharge` (flag 1 + DIED, deathtime after
dischtime) are positive in the primary analysis. The flag is never a model input.

- **A. Evaluation-only exclusion:** primary out-of-fold predictions re-scored without the anomaly
  stays (same folds and models) — isolates the change in the evaluation population.
- **B. Exclusion before splitting and training:** anomaly stays removed first, folds re-drawn over
  the smaller population. B is not fold-for-fold comparable with the primary analysis. If the reduced
  data cannot support the grouped stratified split, B is reported as `not_evaluable`.

## 8. Results on the MIMIC-IV Demo 2.2 (limited evaluation)

Primary population: 107 stays, 86 subjects, 11 positive / 96 negative (10.3%). Folds held 2–3
positive and 19–20 negative test stays each.

Model set v3.0.0 (only the decision tree differs from v2.0.0):

| Model | AUROC [95% CI] | Avg. precision [95% CI] | Brier | Sens. / Spec. at prevalence threshold |
| :--- | :--- | :--- | :--- | :--- |
| dummy_prior (reference) | 0.46 [0.30–0.62] | 0.10 [0.05–0.17] | 0.092 | 1.00 / 0.00 |
| logistic_regression | 0.74 [0.57–0.90] | 0.31 [0.14–0.62] | 0.086 | 0.73 / 0.76 |
| decision_tree (distilled) | 0.69 [0.48–0.87] | 0.25 [0.10–0.54] | 0.088 | 0.64 / 0.67 |
| random_forest | 0.73 [0.53–0.91] | 0.34 [0.14–0.66] | 0.082 | 0.64 / 0.62 |
| neural_network | 0.77 [0.60–0.91] | 0.41 [0.17–0.66] | 0.087 | 0.55 / 0.85 |

Sensitivity analyses (103 stays, 83 subjects, 7 positive / 96 negative):
- **A (evaluation-only):** logistic regression AUROC 0.73, decision tree 0.64, random forest 0.73,
  neural network 0.78.
- **B (retrained, folds with 1–2 positives):** logistic regression 0.40 [0.08–0.76], decision tree 0.67
  [0.43–0.87], random forest 0.71, neural network 0.39 [0.02–0.78].

Earlier decision trees (primary analysis):
- v2 smoothed tree: AUROC 0.56, A 0.52, B 0.61;
- v1 plain tree: AUROC 0.62.

Reading: the intervals overlap heavily; the neural network has the highest pooled AUROC and average
precision but the widest swings between analyses (0.78 in A, 0.39 in B), and logistic regression
collapses the same way in B. The distilled tree is close to its random-forest teacher and stays
stable in B. B shows how fragile any estimate is with 7 positive stays. No model ranking or performance
claim is supported by these numbers.

## 9. What would be needed before performance claims

- The full credentialed MIMIC-IV (tens of thousands of ICU stays), obtained and handled under the
  PhysioNet data use agreement.
- A pre-specified analysis plan; a held-out test set (ideally temporal, e.g. by `anchor_year_group`)
  untouched during development; hyperparameter tuning only within training data (nested CV).
- Calibration assessment (calibration curves, slope/intercept), decision-curve analysis and alert-burden
  estimates at clinically chosen thresholds.
- Subgroup and fairness analyses (age, sex, race, care unit) and missing-data sensitivity analyses.
- External validation on another population (e.g. eICU-CRD) and clinical review before any use.

## 10. Risk levels shown in research mode

The research triage page and the what-if estimator classify each risk on six fixed probability bands.
These are display bands for research, chosen in advance around the cohort death rate of 10.3 %. They are
not clinical thresholds.

| Level | Risk of in-hospital death |
| :--- | :--- |
| Very low | < 2.5 % |
| Low | 2.5–7.5 % |
| Medium | 7.5–15 % |
| High | 15–30 % |
| Very high | 30–60 % |
| Extreme | ≥ 60 % |

The overall level is the mean of the four models' risks. Each model's own percentage and level is shown
beside it.
