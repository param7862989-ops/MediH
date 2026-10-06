"""Validation and Clinical Benchmarking Tests (Phase 8).

Validates:
1. Benchmark engine execution and 5-fold cross validation.
2. Compliance with academic project accuracy and latency thresholds:
   - Accuracy >= 91%
   - Precision >= 89%
   - Recall >= 88%
   - F1-Score >= 89%
   - Latency <= 25ms
3. Serialization of JSON and Markdown benchmark reports.
4. Demo runner pre-flight system integrity checks.
"""

import json
import pytest
from pathlib import Path

from src.evaluation.benchmark import run_clinical_benchmark, TARGET_METRICS
from src.utils.config import Config
from run_demo import run_preflight_checks


def test_clinical_benchmarking_targets_met():
    """Verify that empirical 5-fold cross-validation meets all project targets."""
    report = run_clinical_benchmark(n_splits=5, random_state=42, export_artifacts=True)

    assert report["status"] == "PASS", f"Benchmark failed target checks: {report['compliance']}"

    cv = report["cross_validation_5fold"]
    h = report["holdout_test_set"]

    # Target metrics assertions
    assert cv["accuracy_mean"] >= TARGET_METRICS["accuracy"], (
        f"Accuracy {cv['accuracy_mean']:.3f} fell below target {TARGET_METRICS['accuracy']}"
    )
    assert cv["precision_mean"] >= TARGET_METRICS["precision"], (
        f"Precision {cv['precision_mean']:.3f} fell below target {TARGET_METRICS['precision']}"
    )
    assert cv["recall_mean"] >= TARGET_METRICS["recall"], (
        f"Recall {cv['recall_mean']:.3f} fell below target {TARGET_METRICS['recall']}"
    )
    assert cv["f1_score_mean"] >= TARGET_METRICS["f1_score"], (
        f"F1-Score {cv['f1_score_mean']:.3f} fell below target {TARGET_METRICS['f1_score']}"
    )
    assert h["latency_mean_ms"] <= TARGET_METRICS["max_latency_ms"], (
        f"Latency {h['latency_mean_ms']:.2f}ms exceeded limit {TARGET_METRICS['max_latency_ms']}ms"
    )


def test_benchmark_report_artifacts_exported():
    """Verify that JSON and Markdown benchmark reports exist and contain valid structures."""
    json_report = Config.MODELS_STORE_DIR / "benchmark_report.json"
    md_report = Config.MODELS_STORE_DIR / "benchmark_report.md"

    assert json_report.exists(), f"Missing: {json_report}"
    assert md_report.exists(), f"Missing: {md_report}"

    with open(json_report, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "cross_validation_5fold" in data
    assert "holdout_test_set" in data
    assert "compliance" in data
    assert data["status"] == "PASS"

    md_text = md_report.read_text(encoding="utf-8")
    assert "MediHaven Clinical Benchmarking" in md_text
    assert "Confusion Matrix" in md_text


def test_demo_runner_preflight_checks():
    """Verify that run_demo.py pre-flight integrity check returns True."""
    assert run_preflight_checks() is True
