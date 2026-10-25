"""Startup security checks.

Research mode (DATA_MODE=mimic) fails closed on weak configuration: public default or placeholder
secrets, a debugger, or a non-loopback bind address (research artifacts must stay on this machine).
Synthetic mode reports the same secret weaknesses as warnings in development and refuses to start
with them in production (FLASK_ENV=production).

Bind address and debugger: run_demo.py and ``python -m src.api.app`` pass Config.FLASK_HOST to
app.run() and force the debugger off in research mode, so those settings are enforced. The ``flask``
CLI chooses the bind address and debugger outside the application factory (command-line options
that are not reliably visible to it), so research mode refuses to start under the Flask CLI rather
than claim checks it cannot make. Other WSGI servers must be bound to loopback by whoever runs them.
"""

import ipaddress
import os
from typing import List

from src.utils.config import Config

PLACEHOLDER_SECRETS = {
    "replace-with-a-secure-random-secret-key",
    "replace-with-a-secure-jwt-signing-key",
    "changeme", "secret", "",
}
MIN_SECRET_LENGTH = 32
SECRET_HELP = 'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'


class ResearchModeConfigError(RuntimeError):
    """Raised when research mode would start with an unsafe configuration."""


class ProductionConfigError(RuntimeError):
    """Raised when a production deployment (FLASK_ENV=production) would start with weak secrets."""


def weak_secret_reasons() -> List[str]:
    """Lists weaknesses of the session and QR-token signing secrets (never their values)."""
    reasons = []
    for name, value, default in (
        ("FLASK_SECRET_KEY", Config.FLASK_SECRET_KEY, Config.DEFAULT_FLASK_SECRET_KEY),
        ("JWT_SECRET_KEY", Config.JWT_SECRET_KEY, Config.DEFAULT_JWT_SECRET_KEY),
    ):
        if value == default:
            reasons.append(f"{name} is the public default from the source code")
        elif value.strip().lower() in PLACEHOLDER_SECRETS:
            reasons.append(f"{name} is a placeholder value")
        elif len(value) < MIN_SECRET_LENGTH:
            reasons.append(f"{name} is shorter than {MIN_SECRET_LENGTH} characters")
    return reasons


def is_loopback_host(host: str) -> bool:
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip()).is_loopback
    except ValueError:
        return False


def launched_via_flask_cli() -> bool:
    """Flask's CLI sets FLASK_RUN_FROM_CLI before it loads the application."""
    return os.environ.get("FLASK_RUN_FROM_CLI") == "true"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def research_config_problems(host: str = None, debug: bool = False) -> List[str]:
    problems = weak_secret_reasons()
    host = Config.FLASK_HOST if host is None else host
    if not is_loopback_host(host):
        problems.append(f"FLASK_HOST={host} is not a loopback address; research mode only serves this machine")
    run_host = os.environ.get("FLASK_RUN_HOST", "").strip()
    if run_host and not is_loopback_host(run_host):
        problems.append(f"FLASK_RUN_HOST={run_host} is not a loopback address")
    if launched_via_flask_cli():
        problems.append("research mode cannot start under the 'flask' CLI, whose bind address and debugger are "
                        "chosen outside the application and cannot be verified; use run_demo.py or "
                        "python -m src.api.app")
        if _env_flag("FLASK_DEBUG"):
            problems.append("FLASK_DEBUG enables the debugger for Flask CLI launches; it must be off in research mode")
    if debug:
        problems.append("the Flask debugger must be off in research mode")
    return problems


def require_research_safe_config(host: str = None, debug: bool = False) -> None:
    problems = research_config_problems(host, debug)
    if problems:
        raise ResearchModeConfigError(
            "Research mode refuses to start: " + "; ".join(problems) + f". {SECRET_HELP}"
        )


def require_production_safe_config() -> None:
    """Production fails closed on public default, placeholder or short signing secrets."""
    problems = weak_secret_reasons()
    if problems:
        raise ProductionConfigError(
            "Production (FLASK_ENV=production) refuses to start: " + "; ".join(problems)
            + ". Anyone could forge sessions and QR passes. " + SECRET_HELP
        )
