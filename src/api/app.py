"""Master Flask Application Factory and Web Entry Point for MediHaven.

Configures REST API Blueprints, in-memory machine learning model caching,
CORS security, standardized error handlers, and web portal view routes.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_cors import CORS

from src.utils.config import Config
from src.utils.logger import get_logger
from src.api.schemas import error_response, success_response
from src.api.routes.patients import patients_bp
from src.api.routes.predictions import predictions_bp
from src.api.routes.alerts import alerts_bp
from src.api.routes.vault import vault_bp
from src.api.auth import auth_bp, ROLE_DASHBOARD_MAP
from src.models.ensemble import EnsembleClinicalPredictor

logger = get_logger("medihaven.api.app")


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Creates and configures a MediHaven Flask application instance.

    Args:
        test_config: Optional configuration dictionary override for testing.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(
        __name__,
        template_folder=str(Config.TEMPLATES_DIR),
        static_folder=str(Config.STATIC_DIR),
    )

    # Core Application Configuration
    app.config.from_mapping(
        SECRET_KEY=Config.FLASK_SECRET_KEY,
        JSON_SORT_KEYS=False,
    )

    if test_config:
        app.config.update(test_config)

    # Enable Cross-Origin Resource Sharing (CORS) for all REST endpoints
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Pre-load trained clinical intelligence model into application memory
    try:
        app.ensemble_predictor = EnsembleClinicalPredictor.load()
        logger.info("Ensemble Clinical Intelligence model loaded and cached in app memory.")
    except Exception as e:
        logger.warning(f"Ensemble model could not be pre-loaded at startup: {e}")
        app.ensemble_predictor = None

    # Register API Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(patients_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(alerts_bp)
    app.register_blueprint(vault_bp)

    @app.context_processor
    def inject_user():
        from flask import session
        return {"current_user": session.get("user")}

    def enforce_role(allowed_roles: list):
        """Enforces role-based route protection for web portals."""
        from flask import session, current_app
        # Testing bypass only if not explicitly testing auth enforcement
        if current_app.config.get("TESTING") and not request.headers.get("X-Enforce-Auth"):
            return None

        user = session.get("user")
        if not user:
            return redirect(url_for("view_login", next=request.path, error="Please sign in to access this portal."))

        user_role = user.get("role")
        if user_role not in allowed_roles:
            own_dest = ROLE_DASHBOARD_MAP.get(user_role, "/")
            return redirect(own_dest)

        return None

    # --------------------------------------------------------------------------
    # Web Portal View Routes (Jinja2 Templates & Presentation Shell)
    # --------------------------------------------------------------------------
    @app.route("/", methods=["GET"])
    def index():
        """Landing Page & Clinical Overview."""
        try:
            return render_template("landing.html", active_page="landing")
        except Exception:
            return render_template("base.html", active_page="landing")

    @app.route("/login", methods=["GET"])
    def view_login():
        """Team Role Selection & Login Gateway."""
        from flask import session
        user = session.get("user")
        if user and not request.args.get("switch"):
            target = ROLE_DASHBOARD_MAP.get(user.get("role"), "/")
            return redirect(target)

        try:
            return render_template("login.html", active_page="login")
        except Exception:
            return render_template("base.html", active_page="login")

    @app.route("/logout", methods=["GET"])
    def view_logout():
        """Logs out the active user session and returns to login."""
        from flask import session
        session.clear()
        return redirect(url_for("view_login"))

    @app.route("/dashboard", methods=["GET"])
    def view_dashboard():
        """Physician Clinical Triage Dashboard (Restricted to Physician role)."""
        guard = enforce_role(["physician"])
        if guard:
            return guard

        try:
            return render_template("dashboard.html", active_page="dashboard")
        except Exception:
            return render_template("base.html", active_page="dashboard")

    @app.route("/vault", methods=["GET"])
    def view_vault():
        """Patient Medical Vault Portal (Restricted to Patient role)."""
        guard = enforce_role(["patient"])
        if guard:
            return guard

        try:
            return render_template("patient_vault.html", active_page="vault")
        except Exception:
            return render_template("base.html", active_page="vault")

    @app.route("/scanner", methods=["GET"])
    def view_scanner():
        """Provider Optical QR Scanner View (Restricted to Physician & Admin)."""
        guard = enforce_role(["physician", "admin"])
        if guard:
            return guard

        try:
            return render_template("scanner.html", active_page="scanner")
        except Exception:
            return render_template("base.html", active_page="scanner")

    @app.route("/admin", methods=["GET"])
    def view_admin():
        """Hospital Administrator Operations Overview (Restricted to Admin role)."""
        guard = enforce_role(["admin"])
        if guard:
            return guard

        try:
            return render_template("admin.html", active_page="admin")
        except Exception:
            return render_template("base.html", active_page="admin")

    @app.route("/favicon.ico", methods=["GET"])
    def favicon():
        """Favicon route to prevent 404 logs in browsers."""
        return ("", 204)

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """System health and operational telemetry probe."""
        model_online = getattr(app, "ensemble_predictor", None) is not None
        return success_response(
            data={
                "status": "OPERATIONAL",
                "service": "MediHaven Multimodal Clinical Intelligence System",
                "version": "1.0.0",
                "models_online": model_online,
                "timestamp": datetime.now().isoformat(),
            }
        )

    # --------------------------------------------------------------------------
    # Centralized HTTP Error Handlers
    # --------------------------------------------------------------------------
    @app.errorhandler(400)
    def bad_request(e):
        return error_response(message="Malformed request or invalid parameters.", code="BAD_REQUEST", status_code=400)

    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return error_response(message=f"Endpoint '{request.path}' not found.", code="NOT_FOUND", status_code=404)
        return render_template("base.html", active_page=""), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return error_response(message="HTTP method not allowed for this route.", code="METHOD_NOT_ALLOWED", status_code=405)

    @app.errorhandler(500)
    def internal_server_error(e):
        logger.error(f"Internal server error on {request.path}: {e}", exc_info=True)
        return error_response(message="Internal clinical service error.", code="INTERNAL_SERVER_ERROR", status_code=500)

    logger.info("MediHaven Flask application initialized successfully.")
    return app


if __name__ == "__main__":
    app = create_app()
    print(f"\n========================================================")
    print(f" MediHaven REST API Server Running")
    print(f" Web URL: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
    print(f" Health:  http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/api/health")
    print(f" Triage:  http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/dashboard")
    print(f"========================================================\n")
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG,
    )
