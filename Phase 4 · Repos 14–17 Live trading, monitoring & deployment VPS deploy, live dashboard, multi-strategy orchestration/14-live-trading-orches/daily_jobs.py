"""
scheduler/daily_jobs.py
────────────────────────
Schedule-based jobs that run on a daily clock.

Jobs registered:
  08:55  token_refresh_job     — refresh Kite access token
  09:15  market_open_job       — reset daily state, send morning briefing
  15:00  no_new_positions_job  — log reminder (risk manager enforces this)
  15:20  graceful_shutdown_job — warn about open positions, prepare for close
  15:25  force_exit_job        — market-order close all open positions
  15:30  market_close_job      — EOD summary, persist final state
  15:31  daily_summary_job     — P&L summary to Telegram

All jobs are registered against a `schedule` scheduler running in a
background daemon thread. The scheduler thread is started by create_app()
and stopped by a signal handler on SIGINT/SIGTERM.

Token refresh:
  Kite access tokens expire at midnight IST. The daily_refresh_job opens a
  browser session via Selenium (optional) or prompts for a manual token.
  Set KITE_AUTO_REFRESH=false to skip auto-refresh (manual token update).
"""

import os
import logging
import threading
import time
from datetime import datetime, date
import pytz
import schedule

from utils.telegram import send_info_alert, send_error_alert, send_startup_alert
from dotenv import load_dotenv, set_key
from pathlib import Path

load_dotenv()
logger = logging.getLogger(__name__)
IST    = pytz.timezone("Asia/Kolkata")

KITE_AUTO_REFRESH = os.getenv("KITE_AUTO_REFRESH", "false").lower() == "true"
ENV_FILE          = Path(".env")


class DailyScheduler:
    """
    Wraps the `schedule` library and runs a background thread.
    The orchestrator is injected so jobs can call force_exit, etc.
    """

    def __init__(self, orchestrator):
        self.orch    = orchestrator
        self._thread = None
        self._stop   = threading.Event()
        self._cfg    = orchestrator.cfg

    def start(self):
        """Register all jobs and start the background thread."""
        timing = self._cfg.timing

        schedule.every().day.at(timing.token_refresh).do(self._job(self.token_refresh_job))
        schedule.every().day.at(timing.market_open).do(self._job(self.market_open_job))
        schedule.every().day.at(timing.no_new_positions_after).do(
            self._job(self.no_new_positions_job)
        )
        schedule.every().day.at(timing.graceful_shutdown).do(
            self._job(self.graceful_shutdown_job)
        )
        schedule.every().day.at(timing.force_exit).do(self._job(self.force_exit_job))
        schedule.every().day.at(timing.market_close).do(self._job(self.market_close_job))

        # EOD summary 1 min after close
        close_h, close_m = map(int, timing.market_close.split(":"))
        summary_time = f"{close_h:02d}:{close_m + 1:02d}"
        schedule.every().day.at(summary_time).do(self._job(self.daily_summary_job))

        self._thread = threading.Thread(
            target=self._run_loop,
            name="DailyScheduler",
            daemon=True,
        )
        self._thread.start()

        now = datetime.now(IST)
        logger.info(
            f"Scheduler started at {now.strftime('%H:%M:%S IST')} | "
            f"Jobs: token_refresh={timing.token_refresh}, "
            f"market_open={timing.market_open}, "
            f"force_exit={timing.force_exit}"
        )

    def stop(self):
        """Signal the background thread to stop."""
        self._stop.set()
        schedule.clear()
        logger.info("Scheduler stopped.")

    def _run_loop(self):
        while not self._stop.is_set():
            schedule.run_pending()
            time.sleep(1)

    def _job(self, fn):
        """Wrap a job function so exceptions are caught and alerted."""
        def wrapper():
            try:
                fn()
            except Exception as e:
                logger.exception(f"Scheduler job '{fn.__name__}' failed: {e}")
                send_error_alert(f"Scheduler job `{fn.__name__}` failed:\n`{e}`")
        wrapper.__name__ = fn.__name__
        return wrapper

    # ── Job implementations ───────────────────────────────────────────

    def token_refresh_job(self):
        """
        Refresh the Kite access token.

        If KITE_AUTO_REFRESH=true: attempts Selenium-based auto-login.
        Otherwise:  sends a Telegram prompt to manually update the token.
        """
        logger.info("Token refresh job triggered.")

        if not KITE_AUTO_REFRESH:
            send_info_alert(
                "🔑 *Daily Token Refresh Required*\n"
                "Run: `python utils/generate_token.py`\n"
                "Then restart the server or call:\n"
                "`POST /admin/token/update`"
            )
            return

        # Attempt automated refresh
        try:
            from utils.auto_token import refresh_token_automated
            new_token = refresh_token_automated()
            if new_token:
                os.environ["KITE_ACCESS_TOKEN"] = new_token
                if ENV_FILE.exists():
                    set_key(str(ENV_FILE), "KITE_ACCESS_TOKEN", new_token)

                # Reset Kite client singleton
                from core.kite_client import KiteClient
                KiteClient._instance = None

                logger.info("Access token refreshed successfully.")
                send_info_alert("✅ Kite access token refreshed automatically.")
            else:
                raise RuntimeError("Automated refresh returned empty token.")
        except Exception as e:
            logger.error(f"Auto token refresh failed: {e}")
            send_error_alert(
                f"❌ Token auto-refresh failed: `{e}`\n"
                f"Please refresh manually: `python utils/generate_token.py`"
            )

    def market_open_job(self):
        """Called at 09:15 — reset daily state, send morning briefing."""
        logger.info("Market open — resetting daily state.")
        self.orch.risk.reset_daily()

        cfg      = self.orch.cfg
        open_pos = self.orch.orders.open_positions

        lines = [
            f"🟢 *Market Open — {date.today()}*",
            f"Capital: ₹{cfg.capital.effective_inr:,.0f}",
            f"Active strategies: {len(cfg.enabled_strategies())}",
            f"DRY_RUN: {os.getenv('DRY_RUN', 'true')}",
        ]
        if open_pos:
            lines.append(f"\n⚠️ {len(open_pos)} positions carried from yesterday:")
            for p in open_pos:
                lines.append(f"  • {p.strategy_id} {p.tradingsymbol} qty={p.quantity}")
        send_info_alert("\n".join(lines))

    def no_new_positions_job(self):
        logger.info(f"No new positions after {self.orch.cfg.timing.no_new_positions_after}")
        open_count = len(self.orch.orders.open_positions)
        if open_count > 0:
            send_info_alert(
                f"⏰ No new positions from now.\n"
                f"{open_count} position(s) still open."
            )

    def graceful_shutdown_job(self):
        logger.info("Graceful shutdown triggered.")
        self.orch.graceful_shutdown()

    def force_exit_job(self):
        logger.info("Force exit job triggered.")
        closed = self.orch.force_exit_all(reason="FORCE_EXIT_EOD")
        if not closed:
            logger.info("Force exit: no open positions to close.")

    def market_close_job(self):
        logger.info("Market closed.")
        open_remaining = self.orch.orders.open_positions
        if open_remaining:
            logger.warning(
                f"{len(open_remaining)} positions still open at market close!"
            )
            send_error_alert(
                f"⚠️ {len(open_remaining)} positions open at market close!\n"
                + "\n".join(f"  {p.position_id} {p.tradingsymbol}" for p in open_remaining)
            )

    def daily_summary_job(self):
        """Send EOD P&L summary."""
        risk    = self.orch.risk
        orders  = self.orch.orders
        today   = date.today()

        today_positions = [
            p for p in orders.all_positions
            if p.status != "OPEN"
            and (p.exit_time or "").startswith(str(today))
        ]

        total_pnl  = sum(p.pnl for p in today_positions)
        wins       = sum(1 for p in today_positions if p.pnl > 0)
        losses     = sum(1 for p in today_positions if p.pnl <= 0)

        lines = [
            f"📅 *EOD Summary — {today}*",
            f"Trades: {len(today_positions)} ({wins}W / {losses}L)",
            f"P&L: ₹{total_pnl:,.2f}",
            f"Risk halted: {risk.is_halted}",
        ]
        if today_positions:
            lines.append("")
            for p in today_positions:
                icon = "✅" if p.pnl > 0 else "❌"
                lines.append(
                    f"  {icon} {p.strategy_id} {p.tradingsymbol} "
                    f"→ ₹{p.pnl:,.2f}"
                )
        send_info_alert("\n".join(lines))
        logger.info(f"EOD summary: {len(today_positions)} trades | P&L ₹{total_pnl:,.2f}")
