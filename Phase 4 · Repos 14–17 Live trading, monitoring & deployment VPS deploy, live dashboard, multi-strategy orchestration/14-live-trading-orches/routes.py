"""
api/routes.py
──────────────
Flask blueprints for the orchestrator API.

Endpoints:
  POST /trade            — receive a signal from repo 10 webhook server or direct
  GET  /status           — full system status (risk, positions, strategies)
  GET  /positions        — open positions
  GET  /positions/<id>   — single position detail
  POST /positions/<id>/close  — manually close a position
  POST /config/reload    — hot-reload strategies.yaml without restart
  POST /admin/halt       — emergency halt (set daily loss = limit)
  POST /admin/resume     — resume after manual halt
  POST /admin/token/update  — update access token without restart
  GET  /health           — liveness probe
"""

import os
import hmac
import hashlib
import logging
from flask import Blueprint, request, jsonify, current_app
import pytz
from dotenv import load_dotenv, set_key
from pathlib import Path

load_dotenv()
logger         = logging.getLogger(__name__)
IST            = pytz.timezone("Asia/Kolkata")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
ENV_FILE       = Path(".env")

trade_bp = Blueprint("trade",  __name__)
admin_bp = Blueprint("admin",  __name__)


def _orch():
    """Get orchestrator from Flask app context."""
    return current_app.config["ORCHESTRATOR"]


# ── Auth helper ────────────────────────────────────────────────────────

def _verify_hmac(raw_body: bytes, signature: str) -> bool:
    if not WEBHOOK_SECRET:
        return True   # auth disabled in dev
    expected = hmac.new(WEBHOOK_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    incoming = signature.removeprefix("sha256=").strip()
    return hmac.compare_digest(expected, incoming)


# ── Trade route ────────────────────────────────────────────────────────

@trade_bp.route("/trade", methods=["POST"])
def receive_trade():
    """
    Receive a trading signal and process through the full orchestrator pipeline.

    Accepts signals from:
      - repo 10 (webhook-flask-server) with ml_filter=PASSED header
      - repo 19 (ML filter server) forwarded signals
      - Direct Pine Script alerts (with HMAC auth)
      - Manual curl for testing
    """
    raw_body = request.get_data()

    # HMAC validation (same pattern as repo 10)
    signature = (
        request.headers.get("X-Signature", "") or
        request.headers.get("X-Hub-Signature-256", "")
    )
    if WEBHOOK_SECRET and not _verify_hmac(raw_body, signature):
        logger.warning(f"Auth failed from {request.remote_addr}")
        return jsonify({"error": "invalid_signature"}), 401

    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "invalid_json"}), 400

    orch   = _orch()
    result = orch.process_signal(payload)

    if result.ok:
        return jsonify({
            "status":      "PROCESSED",
            "position_id": result.position_id,
            "dry_run":     result.dry_run,
            "message":     result.message,
        }), 200
    else:
        return jsonify({
            "status":     "BLOCKED",
            "reason":     result.message,
            "blocked_by": result.blocked_by,
        }), 200   # 200 so upstream doesn't retry


# ── Status & positions ─────────────────────────────────────────────────

@admin_bp.route("/status", methods=["GET"])
def status():
    return jsonify(_orch().status()), 200


@admin_bp.route("/health", methods=["GET"])
def health():
    from datetime import datetime
    return jsonify({
        "status":    "ok",
        "timestamp": datetime.now(IST).isoformat(),
        "dry_run":   os.getenv("DRY_RUN", "true").lower() == "true",
    }), 200


@admin_bp.route("/positions", methods=["GET"])
def list_positions():
    orch  = _orch()
    open_ = orch.orders.open_positions
    return jsonify({
        "count":     len(open_),
        "positions": [p.to_dict() for p in open_],
    }), 200


@admin_bp.route("/positions/<position_id>", methods=["GET"])
def get_position(position_id: str):
    pos = _orch().orders.get_position(position_id)
    if not pos:
        return jsonify({"error": "not_found"}), 404
    return jsonify(pos.to_dict()), 200


@admin_bp.route("/positions/<position_id>/close", methods=["POST"])
def close_position(position_id: str):
    data        = request.get_json(silent=True) or {}
    exit_price  = float(data.get("exit_price", 0.0))
    reason      = data.get("reason", "MANUAL_CLOSE")

    orch = _orch()
    pos  = orch.orders.close_position(position_id, reason=reason, exit_price=exit_price)
    if pos is None:
        return jsonify({"error": "position_not_found_or_already_closed"}), 404

    orch.risk.record_close(position_id, pos.pnl)
    return jsonify({
        "status":  "closed",
        "pnl":     pos.pnl,
        "position": pos.to_dict(),
    }), 200


# ── Config ─────────────────────────────────────────────────────────────

@admin_bp.route("/config/reload", methods=["POST"])
def reload_config():
    """Hot-reload strategies.yaml without restarting the server."""
    result = _orch().loader.reload()
    if result["ok"]:
        return jsonify(result), 200
    return jsonify(result), 500


@admin_bp.route("/config", methods=["GET"])
def get_config():
    cfg = _orch().cfg
    return jsonify({
        "name":       cfg.name,
        "version":    cfg.version,
        "capital":    {"total": cfg.capital.total_inr, "effective": cfg.capital.effective_inr},
        "strategies": [
            {
                "id":      s.id,
                "name":    s.name,
                "enabled": s.enabled,
                "dry_run": s.dry_run,
                "symbol":  s.instrument.symbol,
            }
            for s in cfg.strategies
        ],
    }), 200


# ── Admin controls ──────────────────────────────────────────────────────

@admin_bp.route("/admin/halt", methods=["POST"])
def emergency_halt():
    """Manually halt all trading (force the daily loss trigger)."""
    orch = _orch()
    orch.risk._halted      = True
    orch.risk._halt_reason = "Manual emergency halt via /admin/halt"
    logger.warning("EMERGENCY HALT activated via API")
    return jsonify({"status": "halted", "message": "All trading halted."}), 200


@admin_bp.route("/admin/resume", methods=["POST"])
def resume_trading():
    """Resume trading after a manual halt."""
    orch = _orch()
    orch.risk._halted      = False
    orch.risk._halt_reason = ""
    logger.info("Trading resumed via /admin/resume")
    return jsonify({"status": "resumed"}), 200


@admin_bp.route("/admin/token/update", methods=["POST"])
def update_token():
    """
    Update the Kite access token without restarting the server.
    Called after running utils/generate_token.py.

    Body: {"access_token": "new_token_here"}
    """
    data  = request.get_json(silent=True) or {}
    token = data.get("access_token", "").strip()

    if not token or len(token) < 10:
        return jsonify({"error": "invalid_token"}), 400

    # Update env
    os.environ["KITE_ACCESS_TOKEN"] = token
    if ENV_FILE.exists():
        set_key(str(ENV_FILE), "KITE_ACCESS_TOKEN", token)

    # Reset Kite client singleton
    try:
        from core.kite_client import KiteClient
        KiteClient._instance = None
        logger.info("Access token updated and Kite client reset.")
        return jsonify({"status": "ok", "message": "Token updated."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/admin/force-exit", methods=["POST"])
def admin_force_exit():
    """Manual force-exit all positions (emergency use)."""
    reason  = request.get_json(silent=True, force=True) or {}
    reason  = reason.get("reason", "MANUAL_FORCE_EXIT")
    closed  = _orch().force_exit_all(reason=reason)
    total_pnl = sum(p.pnl for p in closed)
    return jsonify({
        "closed":    len(closed),
        "total_pnl": total_pnl,
        "positions": [p.to_dict() for p in closed],
    }), 200
