# 14-live-trading-orchestrator

**Repo 14 of 20** in the [Algo Trading GitHub Series](../README.md)

The central nervous system of the algo trading stack. Wires together the webhook server (repo 10), risk manager, Kite order manager, and ML signal filter (repo 19) into a single production-ready process. Controlled entirely by a YAML config file — no code changes needed to add or disable strategies.

---

## System Architecture

```
TradingView Pine Alert
        │
        ▼
 repo 10 / webhook-flask-server
        │  (HMAC validated, symbol resolved)
        │
        ▼  POST /trade
┌─────────────────────────────────────────────────────────┐
│              14-live-trading-orchestrator                │
│                                                         │
│  process_signal()                                       │
│    │                                                    │
│    ├── 1. Resolve strategy_id → StrategyConfig (YAML)  │
│    ├── 2. ML filter (repo 19) — optional               │
│    ├── 3. RiskManager.approve()                        │
│    │       ├── daily loss circuit breaker              │
│    │       ├── market hours + timing gates             │
│    │       ├── max open positions                      │
│    │       ├── correlated position limit               │
│    │       ├── strategy-level cap                      │
│    │       ├── order rate limit                        │
│    │       └── position sizing (fixed_risk/kelly)      │
│    │                                                   │
│    ├── 4. OrderManager.open_position()                 │
│    │       ├── Place entry order (Kite)                │
│    │       ├── Place SL-M bracket immediately          │
│    │       └── Persist to data/state/                  │
│    │                                                   │
│    └── 5. Telegram notification                        │
│                                                         │
│  DailyScheduler (background thread)                    │
│    08:55 → token refresh                               │
│    09:15 → market open, reset daily state              │
│    15:00 → stop new positions                          │
│    15:20 → graceful shutdown warning                   │
│    15:25 → force-exit all open positions               │
│    15:30 → EOD P&L summary                             │
└─────────────────────────────────────────────────────────┘
        │
        ▼
  Zerodha Kite Connect (live orders)
```

---

## What's in This Repo

| File | Purpose |
|---|---|
| `config/strategies.yaml` | Master config — all strategies, risk params, timing |
| `config/loader.py` | YAML parser → typed dataclasses, hot-reload support |
| `risk/risk_manager.py` | 10-step pre-trade gate + 3 position sizing models |
| `execution/order_manager.py` | Kite order placement, SL bracket, state persistence |
| `core/orchestrator.py` | Central wiring — signal → risk → order → notify |
| `scheduler/daily_jobs.py` | Cron-style daily jobs (token refresh, force exit, EOD) |
| `api/routes.py` | Flask endpoints: /trade, /status, /positions, /admin/* |
| `app.py` | Flask factory with SIGTERM graceful shutdown |
| `run.py` | Dev server entry point |

---

## Setup

### 1. Install

```bash
git clone https://github.com/yourusername/14-live-trading-orchestrator.git
cd 14-live-trading-orchestrator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit: KITE_API_KEY, KITE_ACCESS_TOKEN, WEBHOOK_SECRET, TELEGRAM_*
```

### 3. Edit strategies.yaml

```bash
nano config/strategies.yaml
# Enable/disable strategies, adjust risk params, change symbols
```

### 4. Run tests

```bash
python -m pytest tests/ -v   # 45/45 passing, no Kite/Telegram needed
```

### 5. Start the server

```bash
# Development
python run.py

# Production (gunicorn)
gunicorn -w 2 -b 0.0.0.0:5000 "app:create_app()"
```

---

## Strategy Config (strategies.yaml)

Each strategy entry in `strategies.yaml` is independently controlled:

```yaml
strategies:
  - id: ORB_NIFTY_15M          # matches strategy_id in Pine alert JSON
    name: "Opening Range Breakout"
    enabled: true               # false = signals received but ignored
    dry_run: false              # true = simulate only (overridden by DRY_RUN env)

    instrument:
      symbol: "NIFTY 50"
      exchange: NSE
      product: MIS              # MIS (intraday) | CNC (delivery) | NRML

    exit:
      take_profit_pct: 0.8      # close at +0.8% profit
      stop_loss_pct: 0.4        # close at -0.4% loss
      trailing_stop: true

    risk:
      max_risk_per_trade_pct: 1.5   # size position so max loss = 1.5% of capital
      max_positions: 1
      position_size_model: fixed_risk
```

**Hot-reload** without restart:
```bash
curl -X POST http://localhost:5000/config/reload
```

---

## Risk Manager — 10-Step Gate

Every signal runs through all checks in order. First failure blocks the trade.

| Step | Check | Config Key |
|---|---|---|
| 1 | Global halt (daily loss hit) | `max_daily_loss_pct` |
| 2 | Daily P&L below limit | `max_daily_loss_pct` |
| 3 | Market hours (9:15–15:30) | `market_open/close` |
| 4 | No new positions after time | `no_new_positions_after` |
| 5 | Opening buffer (avoid first N min) | `avoid_first_minutes` |
| 6 | Global max open positions | `max_open_positions` |
| 7 | Max correlated positions (same symbol) | `max_correlated_positions` |
| 8 | Strategy-level max positions | `strategy.risk.max_positions` |
| 9 | Order rate limit (N/min anti-spam) | `max_orders_per_minute` |
| 10 | Position sizing (returns qty > 0) | `position_size_model` |

**Position sizing models:**

`fixed_risk` (recommended): `qty = (capital × risk%) / (price × sl%)`
— sizes each trade so the maximum loss equals the configured % of capital.

`fixed_units`: always trade N lots, regardless of price or volatility.

`kelly`: Kelly criterion capped at 25% — experimental, requires real win-rate data.

---

## Daily Scheduler

Runs in a background daemon thread. All times IST.

| Time | Job | What it does |
|---|---|---|
| 08:55 | `token_refresh_job` | Telegram prompt (or Selenium auto) to refresh Kite token |
| 09:15 | `market_open_job` | Reset daily P&L, send morning briefing to Telegram |
| 15:00 | `no_new_positions_job` | Log reminder (RiskManager enforces this automatically) |
| 15:20 | `graceful_shutdown_job` | Warn about open positions, prepare for forced close |
| 15:25 | `force_exit_job` | Market-order close **all** open positions |
| 15:30 | `market_close_job` | Alert if anything still open |
| 15:31 | `daily_summary_job` | EOD P&L breakdown by strategy to Telegram |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/trade` | Receive signal — full orchestrator pipeline |
| `GET` | `/health` | Liveness probe |
| `GET` | `/status` | Full system status (risk, positions, strategies) |
| `GET` | `/positions` | Open positions |
| `POST` | `/positions/<id>/close` | Manually close a position |
| `GET` | `/config` | Active config summary |
| `POST` | `/config/reload` | Hot-reload strategies.yaml |
| `POST` | `/admin/halt` | Emergency halt (stop all trading) |
| `POST` | `/admin/resume` | Resume after manual halt |
| `POST` | `/admin/token/update` | Update Kite token without restart |
| `POST` | `/admin/force-exit` | Manual force-exit all positions |

---

## Graceful Shutdown (SIGTERM)

```
SIGTERM received
    │
    ├── Stop accepting /trade requests (return 503)
    ├── Wait 2s for in-flight requests
    ├── force_exit_all() on any open positions
    ├── Stop scheduler thread
    └── Exit 0
```

Send SIGTERM via:
```bash
systemctl stop orchestrator      # via systemd
kill -SIGTERM $(cat logs/gunicorn.pid)
```

---

## Integration with Other Repos

**Receives signals from repo 10 (webhook-flask-server):**
```
# In repo 10's .env:
ORCHESTRATOR_URL=http://localhost:5000/trade
```

**Optionally filters via repo 19 (ML signal filter):**
```yaml
# In strategies.yaml:
ml_filter:
  enabled: true
  url: "http://localhost:5001/signal"
  min_confidence: 0.60
  on_timeout: pass   # don't block if ML server is down
```

**Direct Pine Script integration** (without repo 10):
Set your TradingView webhook URL directly to `http://YOUR_SERVER:5000/trade`.

---

## State Persistence

Open positions are written to `data/state/orchestrator_state.json` after every change. On restart, the file is reloaded so the system is aware of any positions opened before the restart. Check the file before restarting if you have open positions.

---

## Disclaimer

Educational use only. Not financial advice. Always paper trade (`DRY_RUN=true`) for several weeks before enabling live trading. The author is not a SEBI-registered investment advisor.
