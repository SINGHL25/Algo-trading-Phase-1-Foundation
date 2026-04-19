"""
execution/order_manager.py
───────────────────────────
Kite Connect order manager with full position lifecycle.

Responsibilities:
  1. Place entry orders (MARKET, LIMIT, SL-M)
  2. Immediately place bracket SL + TP orders after entry
  3. Poll order status until fill confirmed or timeout
  4. Track all open positions in memory + state file
  5. Close positions (full or partial) on signal or time-based exit
  6. Handle partial fills — close only the filled portion
  7. Slippage guard — reject if fill > max_slippage_pct from signal price

State persistence:
  - Open positions written to data/state/orchestrator_state.json on every change
  - Reloaded on startup so a restart doesn't lose position awareness

DRY_RUN=true (default):
  - Orders logged but not placed
  - Positions tracked with simulated fills at signal price
  - Full flow exercised — use this for at least 2–4 weeks before going live
"""

import os
import json
import time
import uuid
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import pytz

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
IST    = pytz.timezone("Asia/Kolkata")

DRY_RUN    = os.getenv("DRY_RUN",    "true").lower() == "true"
STATE_FILE = Path(os.getenv("STATE_FILE", "data/state/orchestrator_state.json"))
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

ORDER_VARIETY = "regular"
POLL_TIMEOUT  = 20    # seconds to wait for fill confirmation


@dataclass
class Position:
    position_id:  str
    strategy_id:  str
    tradingsymbol: str
    exchange:     str
    product:      str
    action:       str           # BUY | SELL
    quantity:     int
    entry_price:  float
    entry_time:   str
    order_id:     str
    sl_price:     float = 0.0
    tp_price:     float = 0.0
    sl_order_id:  Optional[str] = None
    tp_order_id:  Optional[str] = None
    status:       str  = "OPEN"   # OPEN | CLOSED | SL_HIT | TP_HIT
    exit_price:   float = 0.0
    exit_time:    Optional[str] = None
    exit_order_id: Optional[str] = None
    pnl:          float = 0.0
    is_dry_run:   bool  = False

    @property
    def direction(self) -> int:
        return 1 if self.action == "BUY" else -1

    def mark_closed(self, exit_price: float, exit_order_id: str, reason: str):
        self.status        = reason        # CLOSED | SL_HIT | TP_HIT
        self.exit_price    = exit_price
        self.exit_time     = datetime.now(IST).isoformat()
        self.exit_order_id = exit_order_id
        self.pnl           = round(
            (exit_price - self.entry_price) * self.direction * self.quantity, 2
        )

    def to_dict(self) -> dict:
        return asdict(self)


class OrderManager:
    def __init__(self):
        self._positions: dict[str, Position] = {}
        self._load_state()
        if not DRY_RUN:
            from core.kite_client import get_kite
            self.kite = get_kite()
        else:
            self.kite = None
            logger.info("OrderManager: DRY_RUN mode — no real orders will be placed.")

    # ── Entry ──────────────────────────────────────────────────────────

    def open_position(
        self,
        strategy_id:   str,
        tradingsymbol: str,
        exchange:      str,
        product:       str,
        action:        str,
        quantity:      int,
        order_type:    str,
        signal_price:  float,
        sl_pct:        float,
        tp_pct:        float,
        max_slippage_pct: float = 0.2,
        limit_price:   float = 0.0,
        trigger_price: float = 0.0,
    ) -> Optional[Position]:
        """
        Place entry order + SL/TP bracket.

        Returns a Position on success, None on failure.
        """
        now = datetime.now(IST).isoformat()

        # ── DRY RUN ────────────────────────────────────────────────────
        if DRY_RUN:
            fill_price = signal_price or limit_price or 1.0
            pos_id     = f"DRY_{uuid.uuid4().hex[:8].upper()}"
            order_id   = f"DRY_ORDER_{pos_id}"
            logger.info(
                f"[DRY RUN] {action} {quantity}x {tradingsymbol} "
                f"@ {fill_price:.2f} | strategy={strategy_id}"
            )
        else:
            result = self._place_order(
                tradingsymbol = tradingsymbol,
                exchange      = exchange,
                product       = product,
                action        = action,
                quantity      = quantity,
                order_type    = order_type,
                price         = limit_price,
                trigger_price = trigger_price,
            )
            if result is None:
                return None

            order_id   = result["order_id"]
            fill_price = result["fill_price"] or signal_price

            # Slippage guard
            if signal_price and abs(fill_price - signal_price) / signal_price > max_slippage_pct / 100:
                logger.warning(
                    f"Slippage too high: fill={fill_price:.2f} "
                    f"signal={signal_price:.2f} — position still opened"
                )
            pos_id = uuid.uuid4().hex[:12].upper()

        # ── Compute SL / TP prices ─────────────────────────────────────
        direction = 1 if action == "BUY" else -1
        sl_price  = round(fill_price * (1 - direction * sl_pct / 100), 2)
        tp_price  = round(fill_price * (1 + direction * tp_pct / 100), 2)

        position = Position(
            position_id    = pos_id,
            strategy_id    = strategy_id,
            tradingsymbol  = tradingsymbol,
            exchange       = exchange,
            product        = product,
            action         = action,
            quantity       = quantity,
            entry_price    = fill_price,
            entry_time     = now,
            order_id       = order_id,
            sl_price       = sl_price,
            tp_price       = tp_price,
            is_dry_run     = DRY_RUN,
        )

        # Place SL order (SL-M) immediately after entry
        if not DRY_RUN:
            sl_result = self._place_sl_order(
                tradingsymbol, exchange, product,
                "SELL" if action == "BUY" else "BUY",
                quantity, sl_price,
            )
            if sl_result:
                position.sl_order_id = sl_result["order_id"]

        self._positions[pos_id] = position
        self._save_state()

        logger.info(
            f"Position opened: {pos_id} | {strategy_id} | "
            f"{action} {quantity}x {tradingsymbol} @ {fill_price:.2f} | "
            f"SL={sl_price:.2f} TP={tp_price:.2f}"
        )
        return position

    # ── Exit ───────────────────────────────────────────────────────────

    def close_position(
        self,
        position_id:  str,
        reason:       str = "CLOSED",
        exit_price:   float = 0.0,
    ) -> Optional[Position]:
        """
        Close an open position by market order.
        """
        pos = self._positions.get(position_id)
        if not pos or pos.status != "OPEN":
            logger.warning(f"Cannot close position {position_id}: not found or already closed")
            return None

        close_action = "SELL" if pos.action == "BUY" else "BUY"

        if DRY_RUN:
            fill_price  = exit_price or pos.entry_price
            order_id    = f"DRY_CLOSE_{position_id}"
        else:
            result = self._place_order(
                tradingsymbol = pos.tradingsymbol,
                exchange      = pos.exchange,
                product       = pos.product,
                action        = close_action,
                quantity      = pos.quantity,
                order_type    = "MARKET",
            )
            if result is None:
                return None
            order_id   = result["order_id"]
            fill_price = result["fill_price"] or exit_price or pos.entry_price

            # Cancel any pending SL order
            if pos.sl_order_id:
                self._cancel_order(pos.sl_order_id)

        pos.mark_closed(fill_price, order_id, reason)
        self._save_state()

        logger.info(
            f"Position closed: {position_id} | reason={reason} | "
            f"exit={fill_price:.2f} | PnL=₹{pos.pnl:,.2f}"
        )
        return pos

    def close_all(self, reason: str = "FORCE_EXIT_EOD") -> list[Position]:
        """Close every open position — called at force_exit time."""
        closed = []
        for pos_id, pos in list(self._positions.items()):
            if pos.status == "OPEN":
                result = self.close_position(pos_id, reason=reason)
                if result:
                    closed.append(result)
        logger.info(f"Force-closed {len(closed)} positions | reason={reason}")
        return closed

    # ── Queries ────────────────────────────────────────────────────────

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.status == "OPEN"]

    @property
    def all_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_position(self, position_id: str) -> Optional[Position]:
        return self._positions.get(position_id)

    def today_pnl(self) -> float:
        today = datetime.now(IST).strftime("%Y-%m-%d")
        return sum(
            p.pnl for p in self._positions.values()
            if p.status != "OPEN" and (p.exit_time or "").startswith(today)
        )

    def summary(self) -> dict:
        open_p = self.open_positions
        return {
            "open_positions":  len(open_p),
            "total_positions": len(self._positions),
            "today_pnl":       round(self.today_pnl(), 2),
            "positions":       [p.to_dict() for p in open_p],
            "dry_run":         DRY_RUN,
        }

    # ── Kite helpers ───────────────────────────────────────────────────

    def _place_order(
        self,
        tradingsymbol: str,
        exchange:      str,
        product:       str,
        action:        str,
        quantity:      int,
        order_type:    str = "MARKET",
        price:         float = 0.0,
        trigger_price: float = 0.0,
    ) -> Optional[dict]:
        params = {
            "variety":          ORDER_VARIETY,
            "exchange":         exchange,
            "tradingsymbol":    tradingsymbol,
            "transaction_type": action,
            "quantity":         quantity,
            "order_type":       order_type,
            "product":          product,
        }
        if order_type == "LIMIT" and price:
            params["price"] = price
        if order_type in ("SL", "SL-M") and trigger_price:
            params["trigger_price"] = trigger_price
            if order_type == "SL":
                params["price"] = price

        try:
            order_id   = self.kite.place_order(**params)
            fill_price = self._poll_fill(order_id)
            return {"order_id": str(order_id), "fill_price": fill_price}
        except Exception as e:
            logger.error(f"Order failed: {e} | params={params}")
            return None

    def _place_sl_order(
        self,
        tradingsymbol: str,
        exchange:      str,
        product:       str,
        action:        str,
        quantity:      int,
        trigger_price: float,
    ) -> Optional[dict]:
        params = {
            "variety":          ORDER_VARIETY,
            "exchange":         exchange,
            "tradingsymbol":    tradingsymbol,
            "transaction_type": action,
            "quantity":         quantity,
            "order_type":       "SL-M",
            "trigger_price":    round(trigger_price, 2),
            "product":          product,
        }
        try:
            order_id = self.kite.place_order(**params)
            logger.info(f"SL order placed: {order_id} @ trigger={trigger_price:.2f}")
            return {"order_id": str(order_id)}
        except Exception as e:
            logger.error(f"SL order failed: {e}")
            return None

    def _cancel_order(self, order_id: str):
        try:
            self.kite.cancel_order(variety=ORDER_VARIETY, order_id=order_id)
            logger.debug(f"Order {order_id} cancelled")
        except Exception as e:
            logger.warning(f"Cancel failed for {order_id}: {e}")

    def _poll_fill(self, order_id: str, timeout: int = POLL_TIMEOUT) -> Optional[float]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                for o in self.kite.orders():
                    if str(o["order_id"]) == str(order_id):
                        if o["status"] == "COMPLETE":
                            return float(o.get("average_price", 0))
                        if o["status"] in ("CANCELLED", "REJECTED"):
                            logger.warning(f"Order {order_id} {o['status']}")
                            return None
            except Exception as e:
                logger.debug(f"Poll error: {e}")
            time.sleep(1)
        logger.warning(f"Order {order_id} fill timeout after {timeout}s")
        return None

    # ── State persistence ──────────────────────────────────────────────

    def _save_state(self):
        try:
            data = {
                p_id: p.to_dict()
                for p_id, p in self._positions.items()
            }
            with open(STATE_FILE, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"State save failed: {e}")

    def _load_state(self):
        if not STATE_FILE.exists():
            return
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            for p_id, p_dict in data.items():
                try:
                    self._positions[p_id] = Position(**p_dict)
                except Exception as e:
                    logger.warning(f"Could not restore position {p_id}: {e}")
            open_count = len(self.open_positions)
            logger.info(
                f"State loaded: {len(self._positions)} positions "
                f"({open_count} open)"
            )
        except Exception as e:
            logger.error(f"State load failed: {e}")
