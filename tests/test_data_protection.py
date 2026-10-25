"""Data-protection tests: test isolation, research-mode guards and research-path safety.

Validates:
1. Tests run against temporary database/model/processed-data paths only.
2. Destructive operations (recreate, reseed, re-run the feature pipeline) leave the persistent
   demo database and model artifacts byte-identical.
3. DATA_MODE parsing (synthetic default) and the research-path environment settings.
4. DATA_MODE=mimic blocks synthetic auto-seeding, auto-training, synthetic predictions,
   the web app and the demo launcher.
5. Research input/output paths inside the repository or OneDrive are rejected.
"""

import random
from pathlib import Path

import pytest

from src.utils.config import Config, env_path, optional_env_path, parse_data_mode
from src.utils.data_mode import ResearchModeNotSupportedError
from src.utils.research_paths import UnsafeResearchPathError, require_safe_research_path, unsafe_reasons
from tests.isolation_utils import PROJECT_ROOT, describe_changes, snapshot_persistent_artifacts


# ------------------------------------------------------------------------------
# 1-2. Test isolation
# ------------------------------------------------------------------------------
def test_config_paths_point_to_temporary_test_root(test_root):
    for path in (Config.DATABASE_PATH, Config.MODELS_STORE_DIR, Config.PROCESSED_DATA_DIR,
                 Config.SCALER_PATH, Config.KNN_MODEL_PATH, Config.LOG_FILE, Config.AUDIT_LOG_FILE):
        assert test_root in Path(path).parents, f"{path} is not inside the temporary test root"
        assert PROJECT_ROOT not in Path(path).parents
    assert Config.DATA_MODE == "synthetic"
    assert Config.MIMIC_IV_DIR is None and Config.MIMIC_OUTPUT_DIR is None


def test_destructive_operations_leave_persistent_artifacts_unchanged(persistent_snapshot_before):
    from database.seed_data import seed_database
    from src.data.pipeline import run_batch_pipeline
    from src.database.db import init_db

    init_db(force_recreate=True)
    random.seed(42)
    seed_database(100)
    run_batch_pipeline(test_size=0.2, random_state=42, save_artifacts=True)

    assert describe_changes(persistent_snapshot_before, snapshot_persistent_artifacts()) == []


def test_snapshot_comparison_detects_changes():
    """Negative control for the end-of-session guard: created, deleted and modified files are reported."""
    before = {"a.db": "111", "b.joblib": "222", "c.json": None}
    after = {"a.db": "999", "b.joblib": None, "c.json": "333"}
    assert describe_changes(before, after) == ["modified: a.db", "deleted: b.joblib", "created: c.json"]


# ------------------------------------------------------------------------------
# 3. Configuration parsing
# ------------------------------------------------------------------------------
def test_parse_data_mode_defaults_and_validation():
    assert parse_data_mode(None) == "synthetic"
    assert parse_data_mode("") == "synthetic"
    assert parse_data_mode(" MIMIC ") == "mimic"
    with pytest.raises(ValueError, match="Invalid DATA_MODE"):
        parse_data_mode("real")


def test_env_path_helpers(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDIHAVEN_TEST_DIR", str(tmp_path))
    assert env_path("MEDIHAVEN_TEST_DIR", Path("unused")) == tmp_path
    monkeypatch.setenv("MEDIHAVEN_TEST_DIR", "relative/dir")
    assert env_path("MEDIHAVEN_TEST_DIR", Path("unused")) == Config.BASE_DIR / "relative" / "dir"
    monkeypatch.setenv("MEDIHAVEN_TEST_DIR", "")
    assert env_path("MEDIHAVEN_TEST_DIR", tmp_path / "default") == tmp_path / "default"
    assert optional_env_path("MEDIHAVEN_TEST_DIR") is None
    monkeypatch.delenv("MEDIHAVEN_TEST_DIR")
    assert optional_env_path("MEDIHAVEN_TEST_DIR") is None


# ------------------------------------------------------------------------------
# 4. Research-mode guards
# ------------------------------------------------------------------------------
@pytest.fixture
def research_mode(monkeypatch):
    monkeypatch.setattr(Config, "DATA_MODE", "mimic")


def _patient_count() -> int:
    from src.database.db import execute_query
    return execute_query("SELECT COUNT(*) AS n FROM patients")[0]["n"]


def test_research_mode_blocks_synthetic_seeding(research_mode):
    from database.seed_data import seed_database
    before = _patient_count()
    with pytest.raises(ResearchModeNotSupportedError, match="DATA_MODE=mimic"):
        seed_database(100)
    assert _patient_count() == before


def test_research_mode_blocks_database_recreation(research_mode):
    from src.database.db import init_db
    with pytest.raises(ResearchModeNotSupportedError):
        init_db(force_recreate=True)
    assert Config.DATABASE_PATH.exists()


def test_research_mode_blocks_synthetic_pipeline_and_training(research_mode):
    from src.data.pipeline import run_batch_pipeline
    from src.models.ensemble import EnsembleClinicalPredictor
    with pytest.raises(ResearchModeNotSupportedError):
        run_batch_pipeline(save_artifacts=False)
    with pytest.raises(ResearchModeNotSupportedError):
        EnsembleClinicalPredictor.train_from_data()
    with pytest.raises(ResearchModeNotSupportedError):
        EnsembleClinicalPredictor().train_all(None)


def test_research_mode_blocks_synthetic_predictions(research_mode):
    from src.models.ensemble import EnsembleClinicalPredictor
    with pytest.raises(ResearchModeNotSupportedError):
        EnsembleClinicalPredictor().predict_patient(1)


def test_research_mode_blocks_web_app(research_mode):
    from src.api.app import create_app
    with pytest.raises(ResearchModeNotSupportedError, match="web application"):
        create_app({"TESTING": True})


def test_research_mode_demo_launcher_fails_safely(research_mode, monkeypatch, capsys):
    import run_demo

    def must_not_run():
        raise AssertionError("pre-flight self-healing must not run in research mode")

    monkeypatch.setattr(run_demo, "run_preflight_checks", must_not_run)
    with pytest.raises(SystemExit) as exit_info:
        run_demo.main()
    assert exit_info.value.code == 2
    assert "DATA_MODE=mimic" in capsys.readouterr().err

    monkeypatch.undo()
    monkeypatch.setattr(Config, "DATA_MODE", "mimic")
    with pytest.raises(ResearchModeNotSupportedError):
        run_demo.run_preflight_checks()


def test_init_db_refuses_research_directories(monkeypatch, tmp_path):
    from src.database.db import init_db
    monkeypatch.setattr(Config, "MIMIC_OUTPUT_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="research data directory"):
        init_db(db_path=tmp_path / "app.db")
    assert not (tmp_path / "app.db").exists()


# ------------------------------------------------------------------------------
# 5. Research path safety
# ------------------------------------------------------------------------------
def test_research_paths_inside_repository_are_rejected():
    with pytest.raises(UnsafeResearchPathError, match="Git repository"):
        require_safe_research_path(PROJECT_ROOT / "research_output", "the research output directory")


def test_research_paths_inside_onedrive_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("OneDrive", str(tmp_path / "cloud"))
    with pytest.raises(UnsafeResearchPathError, match="OneDrive"):
        require_safe_research_path(tmp_path / "cloud" / "mimic", "the MIMIC-IV input directory")
    assert any("OneDrive" in r for r in unsafe_reasons(tmp_path / "OneDrive - Contoso" / "out"))


def test_local_research_paths_are_accepted(tmp_path):
    assert require_safe_research_path(tmp_path / "out", "the research output directory") == (tmp_path / "out").resolve()
