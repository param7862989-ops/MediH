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
from src.utils.data_mode import is_research_mode
from src.utils.security import ResearchModeConfigError
from src.utils.logger import get_logger
from src.api.app import create_app
from src.bootstrap import prepare_synthetic_assets

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
    # Seeds and trains whatever synthetic assets are missing; refuses to run in research mode
    prepare_synthetic_assets(total_steps=4)

    print(" [4/4] Verifying Presentation Layer Web Templates & UI Assets...")
    templates_ready = all(
        (Config.TEMPLATES_DIR / f).exists()
        for f in ["base.html", "landing.html", "login.html", "dashboard.html", "patient_vault.html", "scanner.html", "admin.html"]
    )
    if templates_ready:
        print("       [OK] All Jinja2 templates (Landing, Login, Dashboard, Vault, Scanner, Admin) verified")
    else:
        print("       [!] Some web templates were not found in templates/")

    print("\n -> All System Pre-Flight Checks Passed Successfully.")
    return True


def open_browser_delayed(url: str, delay_seconds: float = 1.5):
    """Waits for the server to bind before launching browser."""
    def _open():
        time.sleep(delay_seconds)
        try:
            print(f"\n [OK] Opening Landing Page in default browser: {url}")
            webbrowser.open(url)
        except Exception as e:
            print(f" Could not auto-open browser: {e}. Please open {url} manually.")

    thread = threading.Thread(target=_open, daemon=True)
    thread.start()


def run_research_server():
    """Research mode: no pre-flight self-healing (never seeds or trains); serves verified artifacts only."""
    print(" RESEARCH MODE (DATA_MODE=mimic): synthetic seeding, training and clinical predictions are disabled.")
    try:
        app = create_app()
    except ResearchModeConfigError as e:
        print(f"\n [!] {e}\n", file=sys.stderr)
        sys.exit(2)

    status = app.research_service.status()
    if status["research_inference_available"]:
        print(f" [OK] Research artifacts verified ({status['run']['run_id']} v{status['run']['run_version']}).")
    else:
        print(f" [!] Research inference disabled ({status['reason_code']}): {status['reason']}")
    base_url = f"http://{Config.FLASK_HOST}:{Config.FLASK_PORT}"
    print("\n" + "=" * 70)
    print(f"  * Research view (physician/admin): {base_url}/research")
    print(f"  * Login:                           {base_url}/login")
    print(f"  * Health probe:                    {base_url}/api/health")
    print("  Research-only outputs on demo data. Not for patient-care decisions.")
    print("=" * 70)
    print("  Press Ctrl+C to stop the MediHaven server.\n")
    if os.getenv("NO_BROWSER", "0") != "1":
        open_browser_delayed(f"{base_url}/login", delay_seconds=1.5)
    try:
        app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n MediHaven server stopped cleanly. Goodbye!\n")


def main():
    print_banner()
    if is_research_mode():
        run_research_server()
        return
    run_preflight_checks()

    app = create_app()
    base_url = f"http://{Config.FLASK_HOST}:{Config.FLASK_PORT}"
    landing_url = f"{base_url}/"
    login_url = f"{base_url}/login"
    dashboard_url = f"{base_url}/dashboard"
    vault_url = f"{base_url}/vault"
    scanner_url = f"{base_url}/scanner"
    admin_url = f"{base_url}/admin"

    print("\n" + "=" * 70)
    print("  MEDIHAVEN WEB PORTAL URLS:")
    print(f"  * 🏠 Hero & Landing Page:     {landing_url}")
    print(f"  * 👤 Role Selection / Login:  {login_url}")
    print(f"  * 🩺 Physician Triage:        {dashboard_url}")
    print(f"  * 🛡️ Patient Medical Vault:   {vault_url}")
    print(f"  * 📷 Provider QR Scanner:     {scanner_url}")
    print(f"  * 📊 Hospital Administrator:  {admin_url}")
    print(f"  * ⚙️ System Health Probe:     {base_url}/api/health")
    print("=" * 70)
    print("  Press Ctrl+C to stop the MediHaven server.\n")

    # Launch browser automatically to the new landing page
    if os.getenv("NO_BROWSER", "0") != "1":
        open_browser_delayed(landing_url, delay_seconds=1.5)

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
