"""Flask application factory and main server entrypoint for Wattsmart.

Environment-based configuration, production-grade error handling, CORS,
and rate-limiting protection.
"""

import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from app.api import api_bp
from wattsmart.config import (
    APP_ENV,
    CORS_ORIGINS,
    DEBUG,
    FIGURES_DIR,
    HOST,
    PORT,
    PROJECT_ROOT,
    SECRET_KEY,
    SHAP_SUMMARY_PATH,
)
from wattsmart.predict import Predictor

# Configure structured application logger
logging.basicConfig(
    level=logging.INFO if not DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("wattsmart.app")

# Simple in-memory sliding window rate limiter
RATE_LIMIT_WINDOW_SEC = 60
RATE_LIMIT_MAX_REQUESTS = 120
_ip_request_timestamps: Dict[str, list] = defaultdict(list)


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure an instance of the Wattsmart Flask application.

    Args:
        test_config: Optional configuration overrides (useful for testing).

    Returns:
        Configured Flask instance.
    """
    template_dir = PROJECT_ROOT / "app" / "templates"
    static_dir = PROJECT_ROOT / "app" / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )

    # Core configuration from environment
    app.config.update(
        SECRET_KEY=SECRET_KEY,
        ENV=APP_ENV,
        DEBUG=DEBUG,
        TESTING=False,
    )

    if test_config:
        app.config.update(test_config)

    # Restrict CORS to configured origins
    CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})

    # Register blueprints
    app.register_blueprint(api_bp)

    # Pre-load predictor singleton in memory
    try:
        Predictor.get_instance()
    except Exception as exc:
        logger.warning("Predictor deferred initialization: %s", exc)

    # --- Middleware: Rate Limiting & Access Logging ---

    @app.before_request
    def rate_limiter_and_log():
        # Skip rate limit for static files
        if request.path.startswith("/static") or request.path == "/health":
            return None

        client_ip = request.headers.get(
            "X-Forwarded-For", request.remote_addr or "unknown"
        )
        now = time.time()
        # Clean timestamps older than window
        timestamps = _ip_request_timestamps[client_ip]
        _ip_request_timestamps[client_ip] = [
            ts for ts in timestamps if now - ts < RATE_LIMIT_WINDOW_SEC
        ]

        if len(_ip_request_timestamps[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
            logger.warning(
                "Rate limit exceeded for client %s on %s", client_ip, request.path
            )
            return (
                jsonify(
                    {
                        "error": "Too Many Requests",
                        "message": f"Rate limit of {RATE_LIMIT_MAX_REQUESTS} requests per minute exceeded. Please retry shortly.",
                    }
                ),
                429,
            )

        _ip_request_timestamps[client_ip].append(now)
        logger.info("%s %s from %s", request.method, request.path, client_ip)
        return None

    # --- Health & Diagnostic Endpoints ---

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint for Docker / orchestration liveness probes."""
        predictor = Predictor.get_instance()
        is_loaded = predictor.model is not None
        status = "healthy" if is_loaded else "degraded"
        status_code = 200 if is_loaded else 503
        return (
            jsonify(
                {
                    "status": status,
                    "version": "2.0.0",
                    "model_loaded": is_loaded,
                    "timestamp": time.time(),
                }
            ),
            status_code,
        )

    @app.route("/models/shap-summary", methods=["GET"])
    def serve_shap_summary():
        """Serve SHAP summary image."""
        if SHAP_SUMMARY_PATH.exists():
            return send_from_directory(
                str(SHAP_SUMMARY_PATH.parent), SHAP_SUMMARY_PATH.name
            )
        return jsonify({"error": "SHAP summary not found"}), 404

    @app.route("/docs/figures/<path:filename>", methods=["GET"])
    def serve_figure(filename):
        """Serve EDA and model figures."""
        return send_from_directory(str(FIGURES_DIR), filename)

    # --- Frontend Navigation Routes ---

    @app.route("/")
    def index():
        return render_template("predict.html", active_page="predict")

    @app.route("/forecast")
    def forecast_page():
        return render_template("forecast.html", active_page="forecast")

    @app.route("/whatif")
    def whatif_page():
        return render_template("whatif.html", active_page="whatif")

    @app.route("/insights")
    def insights_page():
        return render_template("insights.html", active_page="insights")

    @app.route("/model")
    def model_page():
        return render_template("model.html", active_page="model")

    # --- Error Handlers ---

    @app.errorhandler(400)
    def handle_bad_request(err):
        if request.path.startswith("/api"):
            return jsonify({"error": "Bad Request", "message": str(err)}), 400
        return render_template("error.html", code=400, message="Bad Request"), 400

    @app.errorhandler(404)
    def handle_not_found(err):
        if request.path.startswith("/api"):
            return (
                jsonify(
                    {
                        "error": "Not Found",
                        "message": "The requested resource was not found.",
                    }
                ),
                404,
            )
        return render_template("error.html", code=404, message="Page Not Found"), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(err):
        if request.path.startswith("/api"):
            return (
                jsonify(
                    {
                        "error": "Method Not Allowed",
                        "message": "HTTP method not supported.",
                    }
                ),
                405,
            )
        return (
            render_template("error.html", code=405, message="Method Not Allowed"),
            405,
        )

    @app.errorhandler(500)
    def handle_internal_server_error(err):
        logger.error("Internal Server Error on %s: %s", request.path, err)
        if request.path.startswith("/api"):
            return (
                jsonify(
                    {
                        "error": "Internal Server Error",
                        "message": "An unexpected error occurred.",
                    }
                ),
                500,
            )
        return (
            render_template("error.html", code=500, message="Internal Server Error"),
            500,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    logger.info(
        "Starting Wattsmart server on %s:%d (env: %s, debug: %s)",
        HOST,
        PORT,
        APP_ENV,
        DEBUG,
    )
    app.run(host=HOST, port=PORT, debug=DEBUG)
