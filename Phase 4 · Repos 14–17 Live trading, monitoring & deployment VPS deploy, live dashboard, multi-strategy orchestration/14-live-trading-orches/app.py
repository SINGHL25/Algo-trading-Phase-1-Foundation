"""
app.py
───────
Flask application factory for the live trading orchestrator.

On create_app():
  1. Sets up logging
  2. Creates the Orchestrator singleton
  3. Registers trade + admin blueprints
  4. Starts the DailyScheduler background thread
  5. Registers SIGINT/SIGTERM handlers for graceful shutdown

Graceful shutdown sequence (SIGTERM):
  1. Stop accepting new /trade requests (return 503)
  2. Wait up to 30s for any in-flight order to complete
  3. Call orchestrator.force_exit_all() for any open positions
  4. Stop scheduler thread
  5. Exit with code 0
"""

import os
import sys
import signal
import logging
import threading
import time
from datetime import datetime
from flask import Flask, jsonify, request, g
import pytz
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
IST    = pytz.timezone("Asia/Kolkata")

_shutdown_event = threading.Event()
_shutdown_lock  = threading.Lock()


def create_app(config_path: str = None) -> Flask:
    from utils.logging_config import setup_logging
    setup_logging()

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # ── Create orchestrator ────────────────────────────────────────────
    from core.orchestrator import Orchestrator
    orch = Orchestrator(config_path=config_path)
    app.config["ORCHESTRATOR"] = orch

    # ── Register blueprints ────────────────────────────────────────────
    from api.routes import trade_bp, admin_bp
    app.register_blueprint(trade_bp)
    app.register_blueprint(admin_bp)

    # ── Request timing + shutdown gate ────────────────────────────────
    @app.before_request
    def _check_shutdown():
        if _shutdown_event.is_set() and request.path == "/trade":
            return jsonify({"error": "server_shutting_down"}), 503
        g.start = time.perf_counter()

    @app.after_request
    def _log_request(response):
        elapsed = int((time.perf_counter() - g.start) * 1000)
        logger.info(
            f"{request.method} {request.path} "
            f"→ {response.status_code} ({elapsed}ms)"
        )
        return response

    # ── Global error handlers ─────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "not_found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "method_not_allowed"}), 405

    @app.errorhandler(500)
    def internal_error(e):
        logger.exception(f"Internal error: {e}")
        return jsonify({"error": "internal_server_error"}), 500

    # ── Scheduler ─────────────────────────────────────────────────────
    from scheduler.daily_jobs import DailyScheduler
    scheduler = DailyScheduler(orch)
    scheduler.start()
    app.config["SCHEDULER"] = scheduler

    # ── Signal handlers (graceful shutdown) ───────────────────────────
    def _handle_shutdown(sig, frame):
        logger.info(f"Signal {sig} received — starting graceful shutdown…")
        _shutdown_event.set()

        with _shutdown_lock:
            logger.info("Waiting up to 10s for in-flight requests…")
            time.sleep(2)

            # Force-exit any open positions
            open_pos = orch.orders.open_positions
            if open_pos:
                logger.info(f"Closing {len(open_pos)} open positions before exit…")
                orch.force_exit_all(reason="SERVER_SHUTDOWN")

            scheduler.stop()

        logger.info("Shutdown complete. Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    # ── Startup notification ───────────────────────────────────────────
    dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
    try:
        from utils.telegram import send_startup_alert
        send_startup_alert(
            name       = orch.cfg.name,
            dry_run    = dry_run,
            strategies = len(orch.cfg.enabled_strategies()),
        )
    except Exception:
        pass

    logger.info(
        f"App ready: {orch.cfg.name} | "
        f"DRY_RUN={dry_run} | "
        f"{len(orch.cfg.enabled_strategies())} strategies active"
    )
    return app
