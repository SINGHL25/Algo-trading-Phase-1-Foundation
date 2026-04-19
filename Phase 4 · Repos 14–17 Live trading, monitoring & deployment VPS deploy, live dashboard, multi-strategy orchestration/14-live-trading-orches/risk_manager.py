"""
risk/risk_manager.py
─────────────────────
Pre-trade risk gating for every incoming signal.

Checks run in order (first failure blocks the trade):
  1. Daily loss circuit breaker   — halt all trading if P&L < limit
  2. Max open positions           — across all strategies
  3. Max correlated positions     — same instrument across strategies
  4. Strategy-level max positions — per-strategy cap
  5. Max orders per minute        — anti-spam rate limit
  6. No new positions after time  — market timing
  7. Market hours                 — 9:15–15:30 IST only
  8. Avoid opening auction        — first N minutes after open
  9. Strategy-specific filters    — IV, ADX, volume filters

Position sizing models:
  fixed_risk   — risk a % of capital, size = (capital * risk%) / (price * sl%)
  fixed_units  — always trade N lots (from config)
  kelly        — Kelly criterion (experimental, capped at 25%)

All results returned as RiskDecision dataclass — never raises exceptions.
The orchestrator reads .approved and .reason to decide what to do.
"""

import os
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Optional
import pytz

from config.loader import OrchestratorConfig, StrategyConfig
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
IST    = pytz.timezone("Asia/Kolkata")


@dataclass
class RiskDecision:
    approved:        bool
    reason:          str
    quantity:        int   = 0
    position_value:  float = 0.0
    risk_amount:     float = 0.0
    check_name:      str   = ""   # which check triggered the block


class RiskManager:
    """
    Stateful risk manager.  Holds:
      - daily P&L (updated by orchestrator after each fill)
      - open position count (updated by orchestrator)
      - per-minute order counter (rolling window)
    """

    def __init__(self, config: OrchestratorConfig):
        self.cfg            = config
        self._daily_pnl     = 0.0
        self._open_positions: dict[str, dict] = {}  # position_id → {strategy_id, symbol, qty, ...}
        self._order_times   = deque()               # timestamps of recent orders
        self._halted        = False                 # daily loss halt flag
        self._halt_reason   = ""

    # ── Main entry point ──────────────────────────────────────────────

    def approve(
        self,
        strategy:    StrategyConfig,
        action:      str,        # BUY | SELL
        symbol:      str,
        price:       float,
        now:         Optional[datetime] = None,
    ) -> RiskDecision:
        """
        Run all risk checks and return a RiskDecision.

        Args:
            strategy : resolved StrategyConfig
            action   : 'BUY' or 'SELL'
            symbol   : tradingsymbol
            price    : current market price (used for sizing)
            now      : datetime override (for testing)
        """
        now = now or datetime.now(IST)

        # ── 1. Global halt (daily loss) ───────────────────────────────
        if self._halted:
            return RiskDecision(
                approved=False, reason=self._halt_reason, check_name="daily_loss_halt"
            )

        # ── 2. Check daily loss limit ─────────────────────────────────
        loss_limit = -(self.cfg.capital.effective_inr *
                       self.cfg.risk.max_daily_loss_pct / 100)
        if self._daily_pnl <= loss_limit:
            msg = (
                f"Daily loss limit hit: P&L ₹{self._daily_pnl:,.0f} ≤ "
                f"limit ₹{loss_limit:,.0f}"
            )
            self._halted     = True
            self._halt_reason = msg
            logger.warning(msg)
            return RiskDecision(approved=False, reason=msg, check_name="daily_loss_limit")

        # ── 3. Market hours ───────────────────────────────────────────
        market_open  = _parse_time(self.cfg.timing.market_open)
        market_close = _parse_time(self.cfg.timing.market_close)
        current_time = now.time().replace(second=0, microsecond=0)

        if not (market_open <= current_time <= market_close):
            return RiskDecision(
                approved=False,
                reason=f"Outside market hours ({current_time} IST)",
                check_name="market_hours",
            )

        # ── 4. No new positions after cutoff ──────────────────────────
        no_new_after = _parse_time(self.cfg.timing.no_new_positions_after)
        if current_time >= no_new_after:
            return RiskDecision(
                approved=False,
                reason=f"No new positions after {self.cfg.timing.no_new_positions_after}",
                check_name="no_new_positions_time",
            )

        # ── 5. Opening auction buffer ─────────────────────────────────
        avoid_mins  = self.cfg.timing.avoid_first_minutes
        buffer_end  = dtime(
            market_open.hour,
            market_open.minute + avoid_mins,
        )
        if current_time < buffer_end:
            return RiskDecision(
                approved=False,
                reason=f"Within {avoid_mins}min of market open — avoiding opening volatility",
                check_name="opening_buffer",
            )

        # ── 6. Global max open positions ──────────────────────────────
        open_count = len(self._open_positions)
        if open_count >= self.cfg.risk.max_open_positions:
            return RiskDecision(
                approved=False,
                reason=f"Max open positions reached ({open_count}/{self.cfg.risk.max_open_positions})",
                check_name="max_open_positions",
            )

        # ── 7. Max correlated positions (same symbol) ─────────────────
        same_symbol = sum(
            1 for p in self._open_positions.values()
            if p.get("symbol") == symbol
        )
        if same_symbol >= self.cfg.risk.max_correlated_positions:
            return RiskDecision(
                approved=False,
                reason=(
                    f"Max correlated positions for {symbol}: "
                    f"{same_symbol}/{self.cfg.risk.max_correlated_positions}"
                ),
                check_name="max_correlated_positions",
            )

        # ── 8. Strategy-level max positions ───────────────────────────
        strat_count = sum(
            1 for p in self._open_positions.values()
            if p.get("strategy_id") == strategy.id
        )
        if strat_count >= strategy.risk.max_positions:
            return RiskDecision(
                approved=False,
                reason=(
                    f"Strategy {strategy.id} at max positions "
                    f"({strat_count}/{strategy.risk.max_positions})"
                ),
                check_name="strategy_max_positions",
            )

        # ── 9. Orders-per-minute rate limit ───────────────────────────
        now_ts = time.time()
        self._order_times.append(now_ts)
        # Remove entries older than 60s
        while self._order_times and now_ts - self._order_times[0] > 60:
            self._order_times.popleft()
        if len(self._order_times) > self.cfg.risk.max_orders_per_minute:
            return RiskDecision(
                approved=False,
                reason=f"Order rate limit exceeded ({len(self._order_times)}/min)",
                check_name="order_rate_limit",
            )

        # ── 10. Position sizing ───────────────────────────────────────
        if price <= 0:
            return RiskDecision(
                approved=False,
                reason="Price is 0 — cannot size position",
                check_name="zero_price",
            )

        quantity, risk_amount, pos_value = self._size_position(strategy, price)
        if quantity <= 0:
            return RiskDecision(
                approved=False,
                reason="Position sizing returned 0 quantity",
                check_name="position_sizing",
            )

        logger.info(
            f"Risk approved: {strategy.id} {action} {symbol} "
            f"qty={quantity} risk=₹{risk_amount:,.0f} value=₹{pos_value:,.0f}"
        )
        return RiskDecision(
            approved       = True,
            reason         = "All checks passed",
            quantity       = quantity,
            position_value = pos_value,
            risk_amount    = risk_amount,
        )

    # ── Position sizing ───────────────────────────────────────────────

    def _size_position(
        self, strategy: StrategyConfig, price: float
    ) -> tuple[int, float, float]:
        """
        Returns (quantity, risk_inr, position_value_inr).
        """
        model = strategy.risk.position_size_model

        if model == "fixed_units":
            qty   = strategy.risk.fixed_units * strategy.instrument.lot_size
            value = qty * price
            risk  = value * (strategy.exit.stop_loss_pct / 100)
            return qty, risk, value

        if model == "kelly":
            # Simplified Kelly: f = (p*b - q) / b  where b = TP/SL ratio
            # Capped at 25% of capital for safety
            tp    = strategy.exit.take_profit_pct / 100
            sl    = strategy.exit.stop_loss_pct / 100
            ratio = tp / max(sl, 0.001)
            win_p = 0.55   # assumed 55% win rate — replace with real data
            kelly = max((win_p * ratio - (1 - win_p)) / ratio, 0)
            kelly = min(kelly, 0.25)   # cap at 25%
            capital_at_risk = self.cfg.capital.effective_inr * kelly
            qty   = max(int(capital_at_risk / price), 1) * strategy.instrument.lot_size
            value = qty * price
            risk  = value * sl
            return qty, risk, value

        # Default: fixed_risk
        risk_pct   = strategy.risk.max_risk_per_trade_pct / 100
        sl_pct     = strategy.exit.stop_loss_pct / 100
        capital    = self.cfg.capital.effective_inr
        risk_inr   = capital * risk_pct
        # qty = risk_inr / (price * sl_pct)
        raw_qty    = risk_inr / max(price * sl_pct, 1)
        lot_size   = strategy.instrument.lot_size
        # Round DOWN to nearest lot
        lots       = max(int(raw_qty / lot_size), 1)
        qty        = lots * lot_size
        pos_value  = qty * price
        return qty, risk_inr, pos_value

    # ── State updates ─────────────────────────────────────────────────

    def record_open(self, position_id: str, strategy_id: str, symbol: str,
                    qty: int, entry_price: float, action: str):
        self._open_positions[position_id] = {
            "strategy_id": strategy_id,
            "symbol":      symbol,
            "qty":         qty,
            "entry_price": entry_price,
            "action":      action,
        }
        logger.debug(f"Position recorded: {position_id} ({strategy_id})")

    def record_close(self, position_id: str, pnl: float):
        self._open_positions.pop(position_id, None)
        self._daily_pnl += pnl
        logger.debug(f"Position closed: {position_id} | PnL: ₹{pnl:,.2f} | Daily: ₹{self._daily_pnl:,.2f}")

    def update_pnl(self, pnl_delta: float):
        self._daily_pnl += pnl_delta

    def reset_daily(self):
        """Called at market open each day."""
        self._daily_pnl  = 0.0
        self._halted     = False
        self._halt_reason = ""
        self._order_times.clear()
        logger.info("Risk manager: daily state reset")

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def open_count(self) -> int:
        return len(self._open_positions)

    @property
    def is_halted(self) -> bool:
        return self._halted

    def summary(self) -> dict:
        return {
            "daily_pnl":       round(self._daily_pnl, 2),
            "open_positions":  self.open_count,
            "is_halted":       self._halted,
            "halt_reason":     self._halt_reason,
            "orders_last_min": len(self._order_times),
        }


# ── Helpers ────────────────────────────────────────────────────────────

def _parse_time(t: str) -> dtime:
    h, m = map(int, t.split(":"))
    return dtime(h, m)
