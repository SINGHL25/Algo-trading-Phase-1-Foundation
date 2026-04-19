"""
tests/test_orchestrator.py
───────────────────────────
Full test suite — no Kite API, no Telegram required.
All external calls are patched.

Run: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, time as dtime
from unittest.mock import patch, MagicMock
import pytz

IST = pytz.timezone("Asia/Kolkata")

# ── Minimal valid strategies.yaml content ─────────────────────────────
MINIMAL_YAML = """
orchestrator:
  name: Test Orchestrator
  version: "1.0"
  capital:
    total_inr: 100000
    reserve_pct: 10
  risk:
    max_risk_per_trade_pct: 2.0
    max_daily_loss_pct: 5.0
    max_open_positions: 3
    max_orders_per_minute: 10
    max_correlated_positions: 2
    position_size_model: fixed_risk
  timing:
    market_open: "09:15"
    market_close: "15:30"
    no_new_positions_after: "15:00"
    graceful_shutdown: "15:20"
    force_exit: "15:25"
    token_refresh: "08:55"
    avoid_first_minutes: 0
    avoid_last_minutes: 10
ml_filter:
  enabled: false
strategies:
  - id: TEST_STRAT
    name: Test Strategy
    enabled: true
    dry_run: true
    instrument:
      symbol: "NIFTY 50"
      exchange: NSE
      product: MIS
      lot_size: 1
    entry:
      order_type: MARKET
      max_slippage_pct: 0.5
    exit:
      take_profit_pct: 1.0
      stop_loss_pct: 0.5
      trailing_stop: false
    risk:
      max_risk_per_trade_pct: 2.0
      max_positions: 2
      position_size_model: fixed_risk
    filters:
      avoid_expiry_day: false
  - id: DISABLED_STRAT
    name: Disabled Strategy
    enabled: false
    dry_run: true
    instrument:
      symbol: "RELIANCE"
      exchange: NSE
      product: CNC
      lot_size: 1
    entry:
      order_type: MARKET
    exit:
      take_profit_pct: 2.0
      stop_loss_pct: 1.0
    risk:
      max_positions: 1
"""


@pytest.fixture
def yaml_file(tmp_path):
    f = tmp_path / "strategies.yaml"
    f.write_text(MINIMAL_YAML)
    return str(f)


@pytest.fixture
def config(yaml_file):
    from config.loader import ConfigLoader
    return ConfigLoader(yaml_file).config


@pytest.fixture
def risk_mgr(config):
    from risk.risk_manager import RiskManager
    return RiskManager(config)


@pytest.fixture
def order_mgr(tmp_path, monkeypatch):
    monkeypatch.setenv("DRY_RUN",    "true")
    monkeypatch.setenv("STATE_FILE", str(tmp_path / "state.json"))
    # Reload module to pick up new env
    import importlib
    import execution.order_manager as om
    importlib.reload(om)
    return om.OrderManager()


@pytest.fixture
def orchestrator(yaml_file, tmp_path, monkeypatch):
    monkeypatch.setenv("DRY_RUN",    "true")
    monkeypatch.setenv("STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID",   "")
    with patch("utils.telegram._send", return_value=True):
        from core.orchestrator import Orchestrator
        return Orchestrator(config_path=yaml_file)


@pytest.fixture
def app_client(yaml_file, tmp_path, monkeypatch):
    monkeypatch.setenv("DRY_RUN",     "true")
    monkeypatch.setenv("STATE_FILE",  str(tmp_path / "state.json"))
    monkeypatch.setenv("WEBHOOK_SECRET", "")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID",   "")
    monkeypatch.setenv("STRATEGY_CONFIG",    yaml_file)
    with patch("utils.telegram._send", return_value=True), \
         patch("scheduler.daily_jobs.send_info_alert",  return_value=True), \
         patch("scheduler.daily_jobs.send_error_alert", return_value=True):
        from app import create_app
        flask_app = create_app(config_path=yaml_file)
        flask_app.config["TESTING"] = True
        # Stop scheduler immediately to avoid background threads in tests
        flask_app.config["SCHEDULER"].stop()
        with flask_app.test_client() as c:
            yield c


# ─────────────────────────────────────────────────────────────────────
# Config loader tests
# ─────────────────────────────────────────────────────────────────────

class TestConfigLoader:
    def test_loads_yaml(self, config):
        assert config.name == "Test Orchestrator"
        assert config.version == "1.0"

    def test_strategies_parsed(self, config):
        assert len(config.strategies) == 2

    def test_enabled_strategies(self, config):
        enabled = config.enabled_strategies()
        assert len(enabled) == 1
        assert enabled[0].id == "TEST_STRAT"

    def test_get_strategy_by_id(self, config):
        s = config.get_strategy("TEST_STRAT")
        assert s is not None
        assert s.instrument.symbol == "NIFTY 50"

    def test_get_unknown_strategy_returns_none(self, config):
        assert config.get_strategy("NONEXISTENT") is None

    def test_capital_effective(self, config):
        # 100_000 * (1 - 10/100) = 90_000
        assert config.capital.effective_inr == 90_000

    def test_timing_parsed(self, config):
        assert config.timing.market_open  == "09:15"
        assert config.timing.force_exit   == "15:25"
        assert config.timing.token_refresh == "08:55"

    def test_dry_run_env_override(self, monkeypatch, yaml_file):
        monkeypatch.setenv("DRY_RUN", "true")
        from config.loader import ConfigLoader
        cfg = ConfigLoader(yaml_file).config
        s   = cfg.get_strategy("TEST_STRAT")
        assert s.dry_run is True

    def test_missing_strategy_id_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("orchestrator:\n  name: x\nstrategies:\n  - name: no_id\n    instrument:\n      symbol: X\n      exchange: NSE\n      product: MIS\n    entry: {}\n    exit: {take_profit_pct: 1, stop_loss_pct: 0.5}\n    risk: {}\n    filters: {}")
        from config.loader import ConfigLoader
        with pytest.raises(ValueError, match="id"):
            ConfigLoader(str(bad)).config

    def test_hot_reload(self, yaml_file):
        from config.loader import ConfigLoader
        loader = ConfigLoader(yaml_file)
        result = loader.reload()
        assert result["ok"] is True
        assert result["strategies"] == 2


# ─────────────────────────────────────────────────────────────────────
# Risk manager tests
# ─────────────────────────────────────────────────────────────────────

class TestRiskManager:
    def _during_hours(self):
        return datetime(2024, 1, 15, 9, 45, tzinfo=IST)

    def test_approve_valid_signal(self, risk_mgr, config):
        strategy = config.get_strategy("TEST_STRAT")
        dec = risk_mgr.approve(strategy, "BUY", "NIFTY 50", 21500, now=self._during_hours())
        assert dec.approved is True
        assert dec.quantity > 0

    def test_blocks_after_no_new_positions_time(self, risk_mgr, config):
        strategy = config.get_strategy("TEST_STRAT")
        late = datetime(2024, 1, 15, 15, 5, tzinfo=IST)
        dec  = risk_mgr.approve(strategy, "BUY", "NIFTY 50", 21500, now=late)
        assert dec.approved is False
        assert "no new positions" in dec.reason.lower()

    def test_blocks_outside_market_hours(self, risk_mgr, config):
        strategy = config.get_strategy("TEST_STRAT")
        eve  = datetime(2024, 1, 15, 18, 0, tzinfo=IST)
        dec  = risk_mgr.approve(strategy, "BUY", "NIFTY 50", 21500, now=eve)
        assert dec.approved is False
        assert "market hours" in dec.reason.lower()

    def test_blocks_when_daily_loss_hit(self, risk_mgr, config):
        # Daily loss limit = 90_000 * 5% = 4_500 INR
        risk_mgr._daily_pnl = -5000
        strategy = config.get_strategy("TEST_STRAT")
        dec = risk_mgr.approve(strategy, "BUY", "NIFTY 50", 21500, now=self._during_hours())
        assert dec.approved is False
        assert "daily loss" in dec.reason.lower()

    def test_blocks_when_halted(self, risk_mgr, config):
        risk_mgr._halted     = True
        risk_mgr._halt_reason = "Manual halt"
        strategy = config.get_strategy("TEST_STRAT")
        dec = risk_mgr.approve(strategy, "BUY", "NIFTY 50", 21500, now=self._during_hours())
        assert dec.approved is False

    def test_blocks_max_open_positions(self, risk_mgr, config):
        # Fill up all 3 slots
        for i in range(3):
            risk_mgr.record_open(f"POS{i}", "TEST_STRAT", "NIFTY 50", 1, 21500, "BUY")
        strategy = config.get_strategy("TEST_STRAT")
        dec = risk_mgr.approve(strategy, "BUY", "NIFTY 50", 21500, now=self._during_hours())
        assert dec.approved is False
        assert "max open positions" in dec.reason.lower()

    def test_blocks_max_strategy_positions(self, risk_mgr, config):
        # TEST_STRAT has max_positions=2 AND max_correlated_positions=2.
        # Use a different symbol per slot so the correlated check passes,
        # letting the strategy-level cap trigger instead.
        risk_mgr.record_open("P1", "TEST_STRAT", "NIFTY 50",   1, 21500, "BUY")
        risk_mgr.record_open("P2", "TEST_STRAT", "NIFTY BANK", 1, 46000, "BUY")
        strategy = config.get_strategy("TEST_STRAT")
        dec = risk_mgr.approve(strategy, "BUY", "RELIANCE", 2500, now=self._during_hours())
        assert dec.approved is False
        assert dec.check_name == "strategy_max_positions"

    def test_blocks_zero_price(self, risk_mgr, config):
        strategy = config.get_strategy("TEST_STRAT")
        dec = risk_mgr.approve(strategy, "BUY", "NIFTY 50", 0.0, now=self._during_hours())
        assert dec.approved is False

    def test_position_sizing_fixed_risk(self, risk_mgr, config):
        strategy = config.get_strategy("TEST_STRAT")
        dec = risk_mgr.approve(strategy, "BUY", "NIFTY 50", 21500, now=self._during_hours())
        # risk = 2% of 90_000 = 1_800 INR
        # sl_pct = 0.5% of 21_500 = 107.5 per unit
        # qty = 1_800 / 107.5 ≈ 16 → round down to nearest lot (lot_size=1)
        assert dec.quantity >= 1

    def test_record_and_close_updates_pnl(self, risk_mgr):
        risk_mgr.record_open("P1", "S1", "NIFTY 50", 50, 21500, "BUY")
        assert risk_mgr.open_count == 1
        risk_mgr.record_close("P1", pnl=500)
        assert risk_mgr.open_count == 0
        assert risk_mgr.daily_pnl == 500

    def test_reset_daily_clears_state(self, risk_mgr):
        risk_mgr._daily_pnl = -3000
        risk_mgr._halted    = True
        risk_mgr.reset_daily()
        assert risk_mgr.daily_pnl == 0
        assert risk_mgr.is_halted is False

    def test_summary_returns_dict(self, risk_mgr):
        s = risk_mgr.summary()
        assert "daily_pnl"      in s
        assert "open_positions" in s
        assert "is_halted"      in s


# ─────────────────────────────────────────────────────────────────────
# Order manager tests (DRY_RUN)
# ─────────────────────────────────────────────────────────────────────

class TestOrderManager:
    def test_open_position_dry_run(self, order_mgr):
        pos = order_mgr.open_position(
            strategy_id="TEST", tradingsymbol="NIFTY 50", exchange="NSE",
            product="MIS", action="BUY", quantity=50, order_type="MARKET",
            signal_price=21500, sl_pct=0.5, tp_pct=1.0,
        )
        assert pos is not None
        assert pos.status == "OPEN"
        assert pos.is_dry_run is True
        assert pos.sl_price < pos.entry_price
        assert pos.tp_price > pos.entry_price

    def test_sl_price_correct_for_buy(self, order_mgr):
        pos = order_mgr.open_position(
            strategy_id="T", tradingsymbol="RELIANCE", exchange="NSE",
            product="CNC", action="BUY", quantity=10, order_type="MARKET",
            signal_price=2500, sl_pct=1.0, tp_pct=2.0,
        )
        assert abs(pos.sl_price - 2500 * 0.99) < 1.0
        assert abs(pos.tp_price - 2500 * 1.02) < 1.0

    def test_sl_price_correct_for_sell(self, order_mgr):
        pos = order_mgr.open_position(
            strategy_id="T", tradingsymbol="RELIANCE", exchange="NSE",
            product="MIS", action="SELL", quantity=10, order_type="MARKET",
            signal_price=2500, sl_pct=1.0, tp_pct=2.0,
        )
        # SL above entry for SELL
        assert pos.sl_price > pos.entry_price
        assert pos.tp_price < pos.entry_price

    def test_close_position_calculates_pnl(self, order_mgr):
        pos = order_mgr.open_position(
            strategy_id="T", tradingsymbol="NIFTY 50", exchange="NSE",
            product="MIS", action="BUY", quantity=50, order_type="MARKET",
            signal_price=21000, sl_pct=0.5, tp_pct=1.0,
        )
        closed = order_mgr.close_position(pos.position_id, exit_price=21210)
        assert closed.status == "CLOSED"
        assert closed.pnl == pytest.approx(50 * (21210 - 21000), abs=1)

    def test_open_positions_filter(self, order_mgr):
        p1 = order_mgr.open_position(
            "T", "NIFTY 50", "NSE", "MIS", "BUY", 50, "MARKET",
            21000, 0.5, 1.0,
        )
        assert len(order_mgr.open_positions) == 1
        order_mgr.close_position(p1.position_id)
        assert len(order_mgr.open_positions) == 0

    def test_close_all(self, order_mgr):
        for _ in range(3):
            order_mgr.open_position(
                "T", "NIFTY 50", "NSE", "MIS", "BUY", 1, "MARKET", 21000, 0.5, 1.0
            )
        closed = order_mgr.close_all("EOD")
        assert len(closed) == 3
        assert len(order_mgr.open_positions) == 0

    def test_state_persistence(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DRY_RUN",    "true")
        state_file = str(tmp_path / "state.json")
        monkeypatch.setenv("STATE_FILE", state_file)
        import importlib, execution.order_manager as om
        importlib.reload(om)
        mgr = om.OrderManager()
        pos = mgr.open_position(
            "T", "NIFTY 50", "NSE", "MIS", "BUY", 50, "MARKET", 21000, 0.5, 1.0
        )
        # Reload and check persistence
        importlib.reload(om)
        mgr2 = om.OrderManager()
        assert any(p.position_id == pos.position_id for p in mgr2.open_positions)


# ─────────────────────────────────────────────────────────────────────
# Orchestrator integration tests
# ─────────────────────────────────────────────────────────────────────

class TestOrchestrator:
    def _signal(self, **overrides):
        base = {
            "strategy_id": "TEST_STRAT",
            "action":      "BUY",
            "symbol":      "NIFTY 50",
            "price":       21500.0,
        }
        return {**base, **overrides}

    def test_valid_signal_processed(self, orchestrator):
        # Override timing to allow entry now
        orchestrator.cfg.timing.avoid_first_minutes = 0
        with patch.object(orchestrator.risk, 'approve',
                         wraps=orchestrator.risk.approve) as mock_approve:
            # Ensure risk approves at any time
            from risk.risk_manager import RiskDecision
            mock_approve.return_value = RiskDecision(
                approved=True, reason="test", quantity=50,
                position_value=21500*50, risk_amount=500,
            )
            result = orchestrator.process_signal(self._signal())
        assert result.ok is True
        assert result.position_id is not None

    def test_unknown_strategy_blocked(self, orchestrator):
        result = orchestrator.process_signal(self._signal(strategy_id="UNKNOWN"))
        assert result.ok is False
        assert "Unknown strategy_id" in result.message

    def test_disabled_strategy_blocked(self, orchestrator):
        result = orchestrator.process_signal(self._signal(strategy_id="DISABLED_STRAT"))
        assert result.ok is False
        assert "disabled" in result.message.lower()

    def test_risk_block_propagates(self, orchestrator):
        # Force halt
        orchestrator.risk._halted = True
        orchestrator.risk._halt_reason = "Test halt"
        result = orchestrator.process_signal(self._signal())
        assert result.ok is False
        assert result.blocked_by in ("daily_loss_halt", "orchestrator")

    def test_force_exit_all(self, yaml_file, tmp_path, monkeypatch):
        monkeypatch.setenv("DRY_RUN",    "true")
        monkeypatch.setenv("STATE_FILE", str(tmp_path / "fresh_state.json"))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
        monkeypatch.setenv("TELEGRAM_CHAT_ID",   "")
        with patch("utils.telegram._send", return_value=True):
            from core.orchestrator import Orchestrator
            fresh = Orchestrator(config_path=yaml_file)

        from execution.order_manager import Position
        # Clear any state left by earlier tests in this module run
        fresh.orders._positions.clear()
        fresh.orders._positions["POS1"] = Position(
            position_id="POS1", strategy_id="TEST_STRAT",
            tradingsymbol="NIFTY 50", exchange="NSE", product="MIS",
            action="BUY", quantity=50, entry_price=21000,
            entry_time="2024-01-15T09:30:00", order_id="DRY_001",
        )
        fresh.risk.record_open("POS1", "TEST_STRAT", "NIFTY 50", 50, 21000, "BUY")
        closed = fresh.force_exit_all("TEST_EXIT")
        assert len(closed) == 1
        assert closed[0].position_id == "POS1"

    def test_status_returns_dict(self, orchestrator):
        s = orchestrator.status()
        assert "name"       in s
        assert "risk"       in s
        assert "orders"     in s
        assert "strategies" in s


# ─────────────────────────────────────────────────────────────────────
# API endpoint tests
# ─────────────────────────────────────────────────────────────────────

class TestAPIEndpoints:
    def test_health_ok(self, app_client):
        resp = app_client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_status_endpoint(self, app_client):
        resp = app_client.get("/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "strategies" in data

    def test_positions_empty(self, app_client):
        resp = app_client.get("/positions")
        assert resp.status_code == 200
        assert resp.get_json()["count"] == 0

    def test_trade_blocked_returns_200(self, app_client):
        """BLOCKED trades return 200 (not an error — TV shouldn't retry)."""
        payload = {"strategy_id": "UNKNOWN", "action": "BUY", "symbol": "NIFTY 50", "price": 100}
        resp    = app_client.post("/trade", json=payload)
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "BLOCKED"

    def test_config_endpoint(self, app_client):
        resp = app_client.get("/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "strategies" in data

    def test_config_reload(self, app_client):
        resp = app_client.post("/config/reload")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_halt_and_resume(self, app_client):
        resp = app_client.post("/admin/halt")
        assert resp.status_code == 200
        orch = app_client.application.config["ORCHESTRATOR"]
        assert orch.risk.is_halted is True

        resp = app_client.post("/admin/resume")
        assert resp.status_code == 200
        assert orch.risk.is_halted is False

    def test_force_exit_endpoint(self, app_client):
        resp = app_client.post("/admin/force-exit", json={"reason": "TEST"})
        assert resp.status_code == 200
        assert "closed" in resp.get_json()

    def test_close_nonexistent_position(self, app_client):
        resp = app_client.post("/positions/NONEXISTENT/close", json={})
        assert resp.status_code == 404

    def test_unknown_route_404(self, app_client):
        assert app_client.get("/does/not/exist").status_code == 404
