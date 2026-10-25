"""Phase 2: real-data model training and evaluation for MIMIC-IV Task 1 (research only).

Separate from the synthetic demo models in src/models. Reads the Phase 1 dataset from the
research output directory and writes every artifact back there (never into the repository).

- dataset.py          load + validate the Task 1 dataset (version 1.2.0, task version 3)
- features.py         permitted feature selection, provenance and forbidden-column guards
- splits.py           subject-grouped, stratified cross-validation folds with safety checks
- models.py           fixed-hyperparameter tabular baselines (scikit-learn Pipelines)
- metrics.py          imbalanced binary metrics with explicit "undefined" handling, bootstrap CIs
- evaluation.py       cross-validated evaluation and the anomaly sensitivity analyses
- train_mimic_t1.py   command-line entry point that writes models, metrics, report and manifest
"""
