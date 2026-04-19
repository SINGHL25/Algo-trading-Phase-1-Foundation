"""
core/orchestrator.py
─────────────────────
The central brain of the live trading system.

Wires together:
  ConfigLoader → StrategyConfig
  RiskManager  → pre-trade approval + position sizing
  OrderManager → Kite order placement + lifecycle
  MLFilter     → optional signal quality gate (repo 19)
  Telegram     → real-time notifications

Signal processing flow:
  1. Incoming webhook payload arrives (from repo 10 or directly)
  2. Resolve strategy_id → StrategyConfig
  3. Validate signal (symbol, action, price)
  4. Run ML filter if enabled
  5. Run RiskManager.approve() — all pre-trade checks
  6. Call OrderManager.open_position()
  7. Record position in RiskManager
  8. Send Telegram notification
  9. Return ProcessResult to caller

Close flow:
  1. Incoming SELL/EXIT signal — or scheduled force-exit
  2. Find matching open position
  3. Call OrderManager.close_position()
  4. Call RiskManager.record_close() with P&L
  5. Notify Telegram

The Orchestrator is a singleton created at Flask app startup
and shared across all request handlers and scheduler threads.
"""

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pytz
import requests

from config.loader    import ConfigLoader, StrategyConfig
from risk.risk_manager import RiskManager
from execution.order_manager import OrderManager, Position
from utils.telegram  import send_order_alert, send_error_alert, send_info_alert

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
IST    = pytz.timezone("Asia/Kolkata")


@dataclass
class ProcessResult:
    ok:          bool
    message:     str
    position_id: Optional[str] = None
    position:    Optional[Position] = None
    blocked_by:  Optional[str] = None   # risk check name that blocked
    dry_run:     bool = False


class Orchestrator:
    """
    Singleton orchestrator. Created once at app startup.
    All Flask routes and the scheduler share this instance.
    """

    def __init__(self, config_path: str = None):
        self.loader  = ConfigLoader(config_path or os.getenv("STRATEGY_CONFIG", "config/strategies.yaml"))
        self.risk    = RiskManager(self.cfg)
        self.orders  = OrderManager()
        logger.info(
            f"Orchestrator initialised: {self.cfg.name} | "
            f"DRY_RUN={os.getenv('DRY_RUN','true')} | "
            f"{len(self.cfg.enabled_strategies())} strategies active"
        )

    @property
    def cfg(self):
        return self.loader.config

    # ── Signal processing ─────────────────────────────────────────────

    def process_signal(self, payload: dict) -> ProcessResult:
        """
        Process an incoming trading signal from any source.

        Expected payload keys:
          strategy_id, action, symbol, price (optional), quantity (override, optional),
          exchange (optional override), order_type (optional override)

        Returns ProcessResult.
        """
        strategy_id = payload.get("strategy_id", "").upper()
        action      = payload.get("action", "").upper()
        symbol      = payload.get("symbol", "")
        price       = float(payload.get("price", 0.0))
        now         = datetime.now(IST)

        logger.info(
            f"Signal: strategy={strategy_id} action={action} "
            f"symbol={symbol} price={price}"
        )

        # ── Resolve strategy ──────────────────────────────────────────
        strategy = self.cfg.get_strategy(strategy_id)
        if not strategy:
            return self._block(
                f"Unknown strategy_id '{strategy_id}'. "
                f"Available: {[s.id for s in self.cfg.strategies]}"
            )

        if not strategy.enabled:
            return self._block(
                f"Strategy '{strategy_id}' is disabled in strategies.yaml"
            )

        # ── Resolve symbol ────────────────────────────────────────────
        tradingsymbol = payload.get("tradingsymbol") or strategy.instrument.symbol
        exchange      = payload.get("exchange")      or strategy.instrument.exchange

        # ── Handle exit signals ───────────────────────────────────────
        if action in ("SELL", "EXIT_LONG", "EXIT"):
            return self._handle_exit(strategy, tradingsymbol, price)

        if action not in ("BUY", "LONG"):
            return self._block(f"Unrecognised action '{action}'")

        # ── ML filter ─────────────────────────────────────────────────
        if self.cfg.ml_filter.enabled:
            ml_result = self._call_ml_filter(payload, strategy_id)
            if not ml_result["pass"]:
                return self._block(
                    f"ML filter blocked: confidence={ml_result.get('confidence', 0):.2f} "
                    f"< threshold={self.cfg.ml_filter.min_confidence}",
                    blocked_by="ml_filter",
                )
            logger.info(f"ML filter passed: confidence={ml_result.get('confidence', 0):.2f}")

        # ── Risk approval ─────────────────────────────────────────────
        decision = self.risk.approve(
            strategy = strategy,
            action   = action,
            symbol   = tradingsymbol,
            price    = price or 1.0,
            now      = now,
        )

        if not decision.approved:
            logger.info(f"Risk blocked: {decision.reason}")
            return ProcessResult(
                ok=False, message=decision.reason,
                blocked_by=decision.check_name,
            )

        # Override quantity if provided in payload (explicit override)
        quantity = int(payload.get("quantity", decision.quantity)) or decision.quantity

        # ── Place order ───────────────────────────────────────────────
        order_type   = payload.get("order_type") or strategy.entry.order_type
        limit_price  = float(payload.get("limit_price",  0.0))
        trigger_price= float(payload.get("trigger_price",0.0))

        position = self.orders.open_position(
            strategy_id      = strategy_id,
            tradingsymbol    = tradingsymbol,
            exchange         = exchange,
            product          = strategy.instrument.product,
            action           = action,
            quantity         = quantity,
            order_type       = order_type,
            signal_price     = price,
            sl_pct           = strategy.exit.stop_loss_pct,
            tp_pct           = strategy.exit.take_profit_pct,
            max_slippage_pct = strategy.entry.max_slippage_pct,
            limit_price      = limit_price,
            trigger_price    = trigger_price,
        )

        if position is None:
            msg = f"Order placement failed for {tradingsymbol}"
            send_error_alert(msg)
            return ProcessResult(ok=False, message=msg)

        # ── Record in risk manager ─────────────────────────────────────
        self.risk.record_open(
            position_id = position.position_id,
            strategy_id = strategy_id,
            symbol      = tradingsymbol,
            qty         = quantity,
            entry_price = position.entry_price,
            action      = action,
        )

        # ── Notify ────────────────────────────────────────────────────
        send_order_alert(
            action       = action,
            symbol       = f"{exchange}:{tradingsymbol}",
            qty          = quantity,
            entry_price  = position.entry_price,
            sl_price     = position.sl_price,
            tp_price     = position.tp_price,
            strategy_id  = strategy_id,
            dry_run      = position.is_dry_run,
        )

        return ProcessResult(
            ok          = True,
            message     = f"Position opened: {position.position_id}",
            position_id = position.position_id,
            position    = position,
            dry_run     = position.is_dry_run,
        )

    # ── Exit handling ─────────────────────────────────────────────────

    def _handle_exit(
        self, strategy: StrategyConfig, symbol: str, price: float
    ) -> ProcessResult:
        """Find open positions for this strategy+symbol and close them."""
        matching = [
            p for p in self.orders.open_positions
            if p.strategy_id == strategy.id
            and p.tradingsymbol == symbol
        ]
        if not matching:
            return ProcessResult(
                ok=False,
                message=f"No open position to exit for {strategy.id} {symbol}",
            )

        closed = []
        for pos in matching:
            result = self.orders.close_position(pos.position_id, reason="SIGNAL_EXIT",
                                                 exit_price=price)
            if result:
                self.risk.record_close(pos.position_id, result.pnl)
                closed.append(result)
                send_order_alert(
                    action      = "EXIT",
                    symbol      = symbol,
                    qty         = result.quantity,
                    entry_price = result.exit_price,
                    sl_price    = 0,
                    tp_price    = 0,
                    strategy_id = strategy.id,
                    dry_run     = result.is_dry_run,
                    pnl         = result.pnl,
                )

        return ProcessResult(
            ok      = True,
            message = f"Closed {len(closed)} position(s) for {strategy.id}",
        )

    # ── EOD force exit ────────────────────────────────────────────────

    def force_exit_all(self, reason: str = "FORCE_EXIT_EOD") -> list[Position]:
        """Called by scheduler at force_exit time."""
        logger.info(f"Force-exiting all positions: {reason}")
        closed = self.orders.close_all(reason=reason)
        for pos in closed:
            self.risk.record_close(pos.position_id, pos.pnl)

        total_pnl = sum(p.pnl for p in closed)
        send_info_alert(
            f"🔔 EOD Force Exit\n"
            f"Closed {len(closed)} positions\n"
            f"Total P&L: ₹{total_pnl:,.2f}"
        )
        return closed

    def graceful_shutdown(self):
        """
        Called at graceful_shutdown time (15:20).
        Notifies of pending positions — actual close happens at force_exit.
        """
        open_pos = self.orders.open_positions
        if not open_pos:
            return
        lines = [f"⏰ *Graceful shutdown in 5 min*\n{len(open_pos)} open positions:"]
        for p in open_pos:
            lines.append(f"  • {p.strategy_id} {p.tradingsymbol} qty={p.quantity}")
        send_info_alert("\n".join(lines))
        logger.info(f"Graceful shutdown triggered: {len(open_pos)} positions pending close")

    # ── ML filter ─────────────────────────────────────────────────────

    def _call_ml_filter(self, payload: dict, strategy_id: str) -> dict:
        ml = self.cfg.ml_filter
        try:
            resp = requests.post(
                ml.url,
                json    = {**payload, "strategy_id": strategy_id},
                timeout = ml.timeout_s,
            )
            data       = resp.json()
            confidence = float(data.get("confidence", 0.0))
            decision   = data.get("decision", "BLOCK")
            return {
                "pass":       decision == "PASS" and confidence >= ml.min_confidence,
                "confidence": confidence,
                "decision":   decision,
            }
        except requests.Timeout:
            logger.warning(f"ML filter timeout after {ml.timeout_s}s")
            return {"pass": ml.on_timeout == "pass", "confidence": 0.0}
        except Exception as e:
            logger.error(f"ML filter error: {e}")
            return {"pass": ml.on_timeout == "pass", "confidence": 0.0}

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _block(message: str, blocked_by: str = "orchestrator") -> ProcessResult:
        logger.info(f"Signal blocked: {message}")
        return ProcessResult(ok=False, message=message, blocked_by=blocked_by)

    def status(self) -> dict:
        return {
            "name":       self.cfg.name,
            "version":    self.cfg.version,
            "dry_run":    os.getenv("DRY_RUN", "true").lower() == "true",
            "risk":       self.risk.summary(),
            "orders":     self.orders.summary(),
            "strategies": [
                {"id": s.id, "name": s.name, "enabled": s.enabled, "dry_run": s.dry_run}
                for s in self.cfg.strategies
            ],
        }
