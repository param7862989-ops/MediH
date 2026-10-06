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
    app.register_blueprint(patients_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(alerts_bp)
    app.register_blueprint(vault_bp)

    # --------------------------------------------------------------------------
    # Web Portal View Routes (Jinja2 Templates & Presentation Shell)
    # --------------------------------------------------------------------------
    @app.route("/", methods=["GET"])
    def index():
        """Root redirect to the Physician Triage Dashboard."""
        return redirect(url_for("view_dashboard"))

    @app.route("/dashboard", methods=["GET"])
    def view_dashboard():
        """Physician Clinical Triage Dashboard."""
        try:
            return render_template("dashboard.html", active_page="dashboard")
        except Exception:
            # Fallback to base shell if dashboard.html is pending Phase 7 completion
            return render_template("base.html", active_page="dashboard")

    @app.route("/vault", methods=["GET"])
    def view_vault():
        """Patient Medical Vault Portal."""
        try:
            return render_template("patient_vault.html", active_page="vault")
        except Exception:
            return render_template("base.html", active_page="vault")

    @app.route("/scanner", methods=["GET"])
    def view_scanner():
        """Provider Optical QR Scanner View."""
        try:
            return render_template("scanner.html", active_page="scanner")
        except Exception:
            return render_template("base.html", active_page="scanner")

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
