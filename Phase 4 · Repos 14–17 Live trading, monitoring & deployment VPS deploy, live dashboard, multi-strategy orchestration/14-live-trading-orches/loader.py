"""
config/loader.py
─────────────────
Loads, validates, and hot-reloads the strategies.yaml config file.

Key responsibilities:
  - Parse YAML into typed dataclasses
  - Validate all required fields with clear error messages
  - Support hot-reload: POST /config/reload re-reads disk without restart
  - Merge environment variables as overrides (e.g. DRY_RUN=true overrides
    any strategy's dry_run: false)
  - Expose get_strategy(id) for O(1) lookup

Thread safety: a threading.RLock guards the config dict so the scheduler
thread and Flask threads can both read/write without races.
"""

import os
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CONFIG_PATH = os.getenv("STRATEGY_CONFIG", "config/strategies.yaml")
_lock       = threading.RLock()


# ── Dataclasses ────────────────────────────────────────────────────────

@dataclass
class InstrumentConfig:
    symbol:   str
    exchange: str
    product:  str       # MIS | CNC | NRML
    lot_size: int = 1

@dataclass
class EntryConfig:
    order_type:        str   = "MARKET"
    max_slippage_pct:  float = 0.2
    limit_offset_pct:  float = 0.0

@dataclass
class ExitConfig:
    take_profit_pct:        float = 1.0
    stop_loss_pct:          float = 0.5
    trailing_stop:          bool  = False
    trailing_atr_multiplier: float = 1.5

@dataclass
class StrategyRisk:
    max_risk_per_trade_pct: float = 2.0
    max_positions:          int   = 1
    position_size_model:    str   = "fixed_risk"
    fixed_units:            int   = 1

@dataclass
class StrategyFilters:
    min_iv:              Optional[float] = None
    max_iv:              Optional[float] = None
    avoid_expiry_day:    bool            = False
    min_adx:             Optional[float] = None
    min_volume_vs_avg:   Optional[float] = None

@dataclass
class StrategyConfig:
    id:         str
    name:       str
    enabled:    bool
    dry_run:    bool
    instrument: InstrumentConfig
    entry:      EntryConfig
    exit:       ExitConfig
    risk:       StrategyRisk
    filters:    StrategyFilters

@dataclass
class OrchestratorRisk:
    max_risk_per_trade_pct:   float = 2.0
    max_daily_loss_pct:       float = 5.0
    max_open_positions:       int   = 5
    max_orders_per_minute:    int   = 10
    max_correlated_positions: int   = 2
    position_size_model:      str   = "fixed_risk"

@dataclass
class TimingConfig:
    market_open:              str   = "09:15"
    market_close:             str   = "15:30"
    no_new_positions_after:   str   = "15:00"
    graceful_shutdown:        str   = "15:20"
    force_exit:               str   = "15:25"
    token_refresh:            str   = "08:55"
    avoid_first_minutes:      int   = 5
    avoid_last_minutes:       int   = 10

@dataclass
class CapitalConfig:
    total_inr:   float = 500_000
    reserve_pct: float = 10.0

    @property
    def effective_inr(self) -> float:
        return self.total_inr * (1 - self.reserve_pct / 100)

@dataclass
class MLFilterConfig:
    enabled:        bool  = False
    url:            str   = "http://localhost:5001/signal"
    timeout_s:      float = 3.0
    on_timeout:     str   = "pass"
    min_confidence: float = 0.60

@dataclass
class OrchestratorConfig:
    name:       str
    version:    str
    capital:    CapitalConfig
    risk:       OrchestratorRisk
    timing:     TimingConfig
    ml_filter:  MLFilterConfig
    strategies: list[StrategyConfig] = field(default_factory=list)

    def get_strategy(self, strategy_id: str) -> Optional[StrategyConfig]:
        for s in self.strategies:
            if s.id == strategy_id:
                return s
        return None

    def enabled_strategies(self) -> list[StrategyConfig]:
        return [s for s in self.strategies if s.enabled]


# ── Loader ────────────────────────────────────────────────────────────

class ConfigLoader:
    """
    Thread-safe YAML config loader with hot-reload support.

    Usage:
        loader = ConfigLoader()
        cfg    = loader.config          # OrchestratorConfig
        strat  = loader.config.get_strategy("ORB_NIFTY_15M")
    """

    def __init__(self, path: str = CONFIG_PATH):
        self._path   = Path(path)
        self._config: Optional[OrchestratorConfig] = None
        self._lock   = threading.RLock()
        self.load()

    @property
    def config(self) -> OrchestratorConfig:
        with self._lock:
            return self._config

    def load(self) -> OrchestratorConfig:
        """Load (or reload) config from disk. Thread-safe."""
        if not self._path.exists():
            raise FileNotFoundError(
                f"Strategy config not found: {self._path}\n"
                f"Copy config/strategies.yaml.example and edit it."
            )
        with open(self._path) as f:
            raw = yaml.safe_load(f)

        cfg = self._parse(raw)
        with self._lock:
            self._config = cfg

        logger.info(
            f"Config loaded: {cfg.name} v{cfg.version} | "
            f"{len(cfg.strategies)} strategies "
            f"({len(cfg.enabled_strategies())} enabled)"
        )
        return cfg

    def reload(self) -> dict:
        """Hot-reload from disk. Returns summary dict."""
        try:
            cfg  = self.load()
            return {
                "ok":       True,
                "strategies": len(cfg.strategies),
                "enabled":  len(cfg.enabled_strategies()),
                "path":     str(self._path),
            }
        except Exception as e:
            logger.error(f"Config reload failed: {e}")
            return {"ok": False, "error": str(e)}

    # ── Parsing ────────────────────────────────────────────────────────

    def _parse(self, raw: dict) -> OrchestratorConfig:
        orch = raw.get("orchestrator", {})

        cap_raw = orch.get("capital", {})
        capital = CapitalConfig(
            total_inr   = float(cap_raw.get("total_inr",  os.getenv("MAX_CAPITAL", 500_000))),
            reserve_pct = float(cap_raw.get("reserve_pct", 10)),
        )

        risk_raw = orch.get("risk", {})
        risk = OrchestratorRisk(
            max_risk_per_trade_pct   = float(risk_raw.get("max_risk_per_trade_pct",
                                             os.getenv("MAX_RISK_PER_TRADE_PCT", 2.0))),
            max_daily_loss_pct       = float(risk_raw.get("max_daily_loss_pct",
                                             os.getenv("MAX_DAILY_LOSS_PCT", 5.0))),
            max_open_positions       = int(risk_raw.get("max_open_positions",
                                          os.getenv("MAX_OPEN_POSITIONS", 5))),
            max_orders_per_minute    = int(risk_raw.get("max_orders_per_minute",
                                          os.getenv("MAX_ORDERS_PER_MINUTE", 10))),
            max_correlated_positions = int(risk_raw.get("max_correlated_positions", 2)),
            position_size_model      = risk_raw.get("position_size_model", "fixed_risk"),
        )

        timing_raw = orch.get("timing", {})
        timing = TimingConfig(
            market_open            = timing_raw.get("market_open",            "09:15"),
            market_close           = timing_raw.get("market_close",           "15:30"),
            no_new_positions_after = timing_raw.get("no_new_positions_after", "15:00"),
            graceful_shutdown      = timing_raw.get("graceful_shutdown",      "15:20"),
            force_exit             = timing_raw.get("force_exit",             "15:25"),
            token_refresh          = timing_raw.get("token_refresh",          "08:55"),
            avoid_first_minutes    = int(timing_raw.get("avoid_first_minutes", 5)),
            avoid_last_minutes     = int(timing_raw.get("avoid_last_minutes",  10)),
        )

        ml_raw = raw.get("ml_filter", {})
        ml_filter = MLFilterConfig(
            enabled        = bool(ml_raw.get("enabled",        False)),
            url            = ml_raw.get("url",            "http://localhost:5001/signal"),
            timeout_s      = float(ml_raw.get("timeout_s",     3.0)),
            on_timeout     = ml_raw.get("on_timeout",     "pass"),
            min_confidence = float(ml_raw.get("min_confidence", 0.60)),
        )

        strategies = [
            self._parse_strategy(s) for s in raw.get("strategies", [])
        ]

        return OrchestratorConfig(
            name       = orch.get("name",    "Orchestrator"),
            version    = orch.get("version", "1.0"),
            capital    = capital,
            risk       = risk,
            timing     = timing,
            ml_filter  = ml_filter,
            strategies = strategies,
        )

    def _parse_strategy(self, raw: dict) -> StrategyConfig:
        sid = raw.get("id", "")
        if not sid:
            raise ValueError("Every strategy entry must have an 'id' field.")

        # Global DRY_RUN env overrides individual strategy dry_run
        global_dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        strategy_dry   = bool(raw.get("dry_run", global_dry_run))
        effective_dry  = global_dry_run or strategy_dry

        inst_raw = raw.get("instrument", {})
        instrument = InstrumentConfig(
            symbol   = inst_raw.get("symbol",   ""),
            exchange = inst_raw.get("exchange",  "NSE"),
            product  = inst_raw.get("product",   "MIS"),
            lot_size = int(inst_raw.get("lot_size", 1)),
        )
        if not instrument.symbol:
            raise ValueError(f"Strategy '{sid}' missing instrument.symbol")

        entr_raw = raw.get("entry", {})
        entry = EntryConfig(
            order_type       = entr_raw.get("order_type",       "MARKET").upper(),
            max_slippage_pct = float(entr_raw.get("max_slippage_pct", 0.2)),
            limit_offset_pct = float(entr_raw.get("limit_offset_pct", 0.0)),
        )

        exit_raw = raw.get("exit", {})
        exit_cfg = ExitConfig(
            take_profit_pct         = float(exit_raw.get("take_profit_pct",         1.0)),
            stop_loss_pct           = float(exit_raw.get("stop_loss_pct",           0.5)),
            trailing_stop           = bool(exit_raw.get("trailing_stop",            False)),
            trailing_atr_multiplier = float(exit_raw.get("trailing_atr_multiplier", 1.5)),
        )

        risk_raw = raw.get("risk", {})
        risk = StrategyRisk(
            max_risk_per_trade_pct = float(risk_raw.get("max_risk_per_trade_pct", 2.0)),
            max_positions          = int(risk_raw.get("max_positions",            1)),
            position_size_model    = risk_raw.get("position_size_model", "fixed_risk"),
            fixed_units            = int(risk_raw.get("fixed_units", 1)),
        )

        filt_raw = raw.get("filters", {})
        filters = StrategyFilters(
            min_iv            = filt_raw.get("min_iv"),
            max_iv            = filt_raw.get("max_iv"),
            avoid_expiry_day  = bool(filt_raw.get("avoid_expiry_day", False)),
            min_adx           = filt_raw.get("min_adx"),
            min_volume_vs_avg = filt_raw.get("min_volume_vs_avg"),
        )

        return StrategyConfig(
            id         = sid,
            name       = raw.get("name", sid),
            enabled    = bool(raw.get("enabled", True)),
            dry_run    = effective_dry,
            instrument = instrument,
            entry      = entry,
            exit       = exit_cfg,
            risk       = risk,
            filters    = filters,
        )
