"""
api/signal_server.py
─────────────────────
Flask webhook server that:
  1. Receives TradingView Pine Script alert webhooks
  2. Fetches the last N candles of live OHLCV data from Kite
  3. Computes features for the current bar
  4. Runs the XGBoost classifier
  5. Forwards the signal to the trading orchestrator ONLY if proba >= MIN_PROBA

Endpoint: POST /signal
Auth:      HMAC-SHA256 signature in X-Signature header (same as TradingView webhook secret)

Expected JSON payload (from Pine Script alert):
  {
    "action":     "BUY",           // or SELL
    "symbol":     "NIFTY 50",
    "exchange":   "NSE",
    "strategy_id":"ORB_15min",
    "price":      21500.0,
    "timestamp":  "2024-01-25T09:30:00+05:30"
  }

Response on PASS:
  { "decision": "PASS", "confidence": 0.73, "signal": {...} }

Response on BLOCK:
  { "decision": "BLOCK", "confidence": 0.41, "reason": "below_threshold" }

Integration with orchestrator (repo 14):
  Set ORCHESTRATOR_URL=http://localhost:5000/trade in .env
  This server POSTs the filtered signal forward on PASS.
"""

import os
import hmac
import hashlib
import logging
import json
import time
import requests
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
import pytz
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

logger = logging.getLogger(__name__)
IST    = pytz.timezone("Asia/Kolkata")

API_PORT         = int(os.getenv("API_PORT",    "5001"))
API_SECRET       = os.getenv("API_SECRET",      "")
MIN_PROBA        = float(os.getenv("MIN_PROBA", "0.60"))
MODEL_PATH       = os.getenv("MODEL_PATH",       "data/models/xgb_signal_filter.joblib")
FEATURE_WINDOW   = int(os.getenv("FEATURE_WINDOW", "50"))
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "")
SYMBOL           = os.getenv("SYMBOL", "NSE:NIFTY 50")
TIMEFRAME        = os.getenv("TIMEFRAME", "15minute")

LOG_FILE = Path("data/signal_log.jsonl")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# Lazy-loaded model and feature engineer (loaded on first request)
_clf = None
_fe  = None


def _load_model():
    global _clf
    if _clf is None:
        from model.xgb_classifier import SignalClassifier
        _clf = SignalClassifier.load(MODEL_PATH)
        logger.info(f"Model loaded: {MODEL_PATH}")
    return _clf


def _load_fe():
    global _fe
    if _fe is None:
        from features.feature_engineer import FeatureEngineer
        _fe = FeatureEngineer()
    return _fe


# ──────────────────────────────────────────────────────────────────────
# Main webhook endpoint
# ──────────────────────────────────────────────────────────────────────

@app.route("/signal", methods=["POST"])
def receive_signal():
    """Main endpoint — receives TradingView webhook, filters, forwards."""

    # ── Auth check ────────────────────────────────────────────────────
    if API_SECRET:
        sig_header = request.headers.get("X-Signature", "")
        body       = request.get_data()
        expected   = hmac.new(
            API_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig_header, expected):
            logger.warning("Webhook auth failed — bad signature.")
            return jsonify({"error": "unauthorized"}), 401

    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400

    if not payload:
        return jsonify({"error": "empty payload"}), 400

    action    = payload.get("action", "").upper()
    symbol    = payload.get("symbol", SYMBOL)
    timestamp = payload.get("timestamp", datetime.now(IST).isoformat())

    logger.info(f"Signal received: {action} {symbol} @ {timestamp}")

    # ── Fetch live features ───────────────────────────────────────────
    try:
        features_row = _get_live_features(symbol, action)
    except Exception as e:
        logger.error(f"Feature fetch failed: {e}")
        _log_signal(payload, "ERROR", 0.0, str(e))
        return jsonify({"decision": "BLOCK", "reason": f"feature_error: {e}"}), 500

    # ── Run ML filter ─────────────────────────────────────────────────
    try:
        clf       = _load_model()
        passed, confidence = clf.filter_signal(features_row)
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        _log_signal(payload, "ERROR", 0.0, str(e))
        return jsonify({"decision": "BLOCK", "reason": f"model_error: {e}"}), 500

    decision = "PASS" if passed else "BLOCK"
    reason   = "above_threshold" if passed else "below_threshold"

    logger.info(
        f"  Decision: {decision} | Confidence: {confidence:.3f} | "
        f"Threshold: {MIN_PROBA}"
    )

    # ── Forward to orchestrator if passed ────────────────────────────
    if passed and ORCHESTRATOR_URL:
        _forward_signal(payload, confidence)

    _log_signal(payload, decision, confidence, reason)

    return jsonify({
        "decision":   decision,
        "confidence": confidence,
        "threshold":  MIN_PROBA,
        "reason":     reason,
        "signal":     payload,
    })


# ── Health check ──────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    model_loaded = Path(MODEL_PATH).exists()
    return jsonify({
        "status":       "ok",
        "model_loaded": model_loaded,
        "min_proba":    MIN_PROBA,
        "timestamp":    datetime.now(IST).isoformat(),
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    try:
        clf = _load_model()
        return jsonify({
            "feature_count": len(clf.feature_names_),
            "min_proba":     clf.min_proba,
            "model_path":    MODEL_PATH,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────────────
# Live feature computation
# ──────────────────────────────────────────────────────────────────────

def _get_live_features(symbol: str, direction: str) -> pd.Series:
    """
    Fetch the last FEATURE_WINDOW candles from Kite and compute features
    for the most recent (current) bar.
    """
    from core.data_fetcher import DataFetcher

    fetcher = DataFetcher()
    # Fetch enough bars for the warmup period (200 for EMA200)
    n_days  = max(FEATURE_WINDOW * 3, 10)
    ohlcv   = fetcher.fetch(
        symbol   = f"NSE:{symbol.replace('NSE:', '')}",
        interval = TIMEFRAME,
        days     = n_days,
        use_cache= False,    # always fresh for live use
    )

    fe       = _load_fe()
    features = fe.transform(ohlcv, dropna=True)

    if features.empty:
        raise ValueError("Feature matrix is empty after transform.")

    latest = features.iloc[-1]

    # Keep only model feature columns (exclude datetime, close)
    clf           = _load_model()
    feature_cols  = clf.feature_names_
    return latest[feature_cols]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _forward_signal(payload: dict, confidence: float):
    """POST the filtered signal to the downstream orchestrator."""
    enriched = {**payload, "ml_confidence": confidence, "ml_filter": "PASSED"}
    try:
        resp = requests.post(ORCHESTRATOR_URL, json=enriched, timeout=5)
        logger.info(f"Forwarded to orchestrator: {resp.status_code}")
    except Exception as e:
        logger.error(f"Orchestrator forward failed: {e}")


def _log_signal(payload: dict, decision: str, confidence: float, reason: str):
    """Append signal outcome to JSONL log."""
    record = {
        "timestamp":  datetime.now(IST).isoformat(),
        "decision":   decision,
        "confidence": confidence,
        "reason":     reason,
        **{k: v for k, v in payload.items() if isinstance(v, (str, int, float, bool))},
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning(f"Signal log write failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)-8s  %(message)s",
    )
    logger.info(f"Starting ML Signal Filter API on port {API_PORT}")
    app.run(host="0.0.0.0", port=API_PORT, debug=False)
