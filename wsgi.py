"""Production WSGI entrypoint for the MediHaven synthetic demo.

    waitress-serve --host=0.0.0.0 --port=8000 wsgi:app

Defaults FLASK_ENV to production, so the app refuses to start with weak secrets, never enables the
debugger and marks the session cookie Secure (set SESSION_COOKIE_SECURE=0 only for plain-HTTP testing).
Missing synthetic assets (database, feature store, models) are created once on first start.

Research mode (DATA_MODE=mimic) is refused here: MIMIC-IV research artifacts are for this machine only
and must never be served to a network. Run research mode with run_demo.py on loopback instead.
See docs/deployment.md.
"""

import os

# Must be set before src.utils.config is imported (it reads the environment once, at import)
os.environ.setdefault("FLASK_ENV", "production")

from src.utils.data_mode import is_research_mode  # noqa: E402

if is_research_mode():
    raise SystemExit("wsgi.py serves the synthetic demo only. Research mode (DATA_MODE=mimic) must stay on "
                     "loopback: start it with run_demo.py. See docs/deployment.md.")

from src.api.app import create_app  # noqa: E402
from src.bootstrap import prepare_synthetic_assets  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

prepare_synthetic_assets(report=get_logger("medihaven.bootstrap").info)
app = create_app()
