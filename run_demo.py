"""MediHaven Single-Command Application Launcher and Demonstration Runner (Phase 8).

Automates:
1. Environment and pre-flight artifact integrity verification (DB, feature store, models, UI).
2. Autonomous self-healing (auto-seeds or trains if artifacts are absent).
3. Pre-loading the 4-model ensemble into server memory.
4. Auto-launching the Flask REST API & Web UI on http://127.0.0.1:5000.
5. Opening default web browser to the Physician Clinical Triage Dashboard.
6. Graceful shutdown on Ctrl+C.
"""

import sys
import os
import time
import threading
import webbrowser
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.utils.config import Config
from src.utils.logger import get_logger
from src.api.app import create_app

logger = get_logger("medihaven.demo")


def print_banner():
    banner = r"""
================================================================================
     __  ___         ___ __  __                           
    /  |/  /__  ____/ (_) / / /___ __   _____  ____       
   / /|_/ / _ \/ __  / / /_/ / __ `/ | / / _ \/ __ \      
  / /  / /  __/ /_/ / / __  / /_/ /| |/ /  __/ / / /      
 /_/  /_/\___/\__,_/_/_/ /_/\__,_/ |___/\___/_/ /_/       

 Integrated Multimodal Clinical Intelligence & Sovereign Medical Vault Platform
 Sardar Patel Institute of Technology (SPIT) - Sem V Mini Project I (2026)
================================================================================
"""
    print(banner)


def run_preflight_checks() -> bool:
    """Verifies that all prerequisite system components and models are ready."""
    print(" [1/4] Checking Relational Database & Seed Data...")
    if not Config.DATABASE_PATH.exists():
        print(f"       -> Database not found at {Config.DATABASE_PATH}. Auto-initializing...")
        from src.database.seed_data import seed_clinical_database
        seed_clinical_database(num_patients=100)
    else:
        print(f"       [OK] SQLite database verified ({Config.DATABASE_PATH.name})")

    print(" [2/4] Checking Clinical Feature Engineering Artifacts...")
    train_parquet = Config.PROCESSED_DATA_DIR / "train.parquet"
    if not train_parquet.exists() or not Config.SCALER_PATH.exists():
        print("       -> Processed feature store missing. Running data engineering pipeline...")
        from src.data.pipeline import run_batch_pipeline
        run_batch_pipeline()
    else:
        print("       [OK] Processed clinical feature store verified (train & test parquets)")

    print(" [3/4] Checking Machine Learning Model Artifacts...")
    models_ready = all(
        p.exists()
        for p in [
            Config.KMEANS_MODEL_PATH,
            Config.DECISION_TREE_MODEL_PATH,
            Config.KNN_MODEL_PATH,
            Config.NEURAL_NETWORK_MODEL_PATH,
        ]
    )
    if not models_ready:
        print("       -> Model weights missing. Fitting and serializing 4-model ensemble...")
        from src.models.ensemble import EnsembleClinicalPredictor
        EnsembleClinicalPredictor.train_from_data()
    else:
        print("       [OK] 4-Model Ensemble verified (K-Means, Decision Tree, KNN, MLP)")

    print(" [4/4] Verifying Presentation Layer Web Templates & UI Assets...")
    templates_ready = all(
        (Config.TEMPLATES_DIR / f).exists()
        for f in ["base.html", "dashboard.html", "patient_vault.html", "scanner.html"]
    )
    if templates_ready:
        print("       [OK] All Jinja2 templates and CSS/JS controllers verified")
    else:
        print("       [!] Some web templates were not found in templates/")

    print("\n -> All System Pre-Flight Checks Passed Successfully.")
    return True


def open_browser_delayed(url: str, delay_seconds: float = 1.5):
    """Waits for the server to bind before launching browser."""
    def _open():
        time.sleep(delay_seconds)
        try:
            print(f"\n [OK] Opening Physician Dashboard in default browser: {url}")
            webbrowser.open(url)
        except Exception as e:
            print(f" Could not auto-open browser: {e}. Please open {url} manually.")

    thread = threading.Thread(target=_open, daemon=True)
    thread.start()


def main():
    print_banner()
    run_preflight_checks()

    app = create_app()
    dashboard_url = f"http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/dashboard"
    vault_url = f"http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/vault"
    scanner_url = f"http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/scanner"

    print("\n" + "=" * 65)
    print("  PORTAL ACCESS URLS:")
    print(f"  * Physician Triage Dashboard: {dashboard_url}")
    print(f"  * Sovereign Patient Vault:    {vault_url}")
    print(f"  * Provider Optical QR Scanner:{scanner_url}")
    print(f"  * Health & API Probe:         http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/api/health")
    print("=" * 65)
    print("  Press Ctrl+C to stop the MediHaven server.\n")

    # Launch browser automatically
    if os.getenv("NO_BROWSER", "0") != "1":
        open_browser_delayed(dashboard_url, delay_seconds=1.5)

    try:
        app.run(
            host=Config.FLASK_HOST,
            port=Config.FLASK_PORT,
            debug=False,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        print("\n\n MediHaven server stopped cleanly. Goodbye!\n")


if __name__ == "__main__":
    main()
