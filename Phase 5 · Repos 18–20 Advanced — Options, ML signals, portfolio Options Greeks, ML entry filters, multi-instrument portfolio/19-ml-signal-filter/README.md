# 19-ml-signal-filter

**Repo 19 of 20** in the [Algo Trading GitHub Series](../README.md)

XGBoost classifier that learns to filter trading signals before they reach the Zerodha order executor. Trained on 60+ technical features with walk-forward validation — no look-ahead bias.

---

## The Core Idea

Most Pine Script strategies fire signals on every bar that meets the indicator conditions. Many of those signals are noise. This ML layer learns which market conditions historically preceded good trades (TP hit) vs bad trades (SL hit), and blocks the low-quality ones.

```
TradingView Pine Alert (JSON webhook)
          │
          ▼
┌─────────────────────────────────┐
│   ML Signal Filter (this repo)  │
│   P(good trade) = 0.71          │ ──── PASS (≥ 0.60) ──▶ Orchestrator ──▶ Kite
│                                 │
│   P(good trade) = 0.42          │ ──── BLOCK (< 0.60) → logged, not traded
└─────────────────────────────────┘
```

**What the model learns:**
- Is IV expanded or compressed right now?
- Are we trending or mean-reverting?
- Is volume confirming the move?
- Is this signal at a structurally strong or weak price level?
- What time of day is it? (avoid first 15 min / last 30 min)

---

## Architecture

```
19-ml-signal-filter/
├── train.py                      ← end-to-end training pipeline
│
├── core/
│   ├── kite_client.py            ← Kite Connect singleton
│   └── data_fetcher.py           ← OHLCV history fetch + cache
│
├── features/
│   ├── feature_engineer.py       ← 60+ technical indicator features
│   └── label_generator.py        ← forward TP/SL binary labels
│
├── model/
│   └── xgb_classifier.py         ← XGBoost + Optuna HPO
│
├── backtest/
│   └── walk_forward.py           ← N-fold walk-forward validation
│
├── api/
│   └── signal_server.py          ← Flask webhook filter server
│
└── utils/
    └── visualize.py              ← feature importance + PR curve + SHAP
```

---

## Feature Groups (60+ features)

| Group | Count | Examples |
|---|---|---|
| **Trend** | 12 | EMA 9/21/50/200 relative, ADX, SuperTrend direction |
| **Momentum** | 14 | RSI 7/14/21, MACD histogram slope, Stochastic, ROC |
| **Volatility** | 10 | ATR%, BB width, %B, Historical vol, Parkinson vol |
| **Volume** | 8 | Vol vs MA20, OBV z-score, VWAP deviation%, AD oscillator |
| **Candle** | 10 | Body/ATR ratio, wick ratios, inside bar, engulfing, consecutive bars |
| **Time** | 6 | Hour sin/cos, day-of-week sin/cos, minutes to close |

All features are computed with **zero look-ahead bias** — each row uses only data available up to that bar.

---

## Walk-Forward Validation

Standard k-fold cross-validation is wrong for time series (it trains on future data). Walk-forward strictly tests on unseen future periods:

```
Fold 1: Train [Jan–Jun] │ Test [Jul]
Fold 2: Train [Feb–Jul] │ Test [Aug]
Fold 3: Train [Mar–Aug] │ Test [Sep]
...
Fold 8: Train [May–Nov] │ Test [Dec]
```

Each fold runs its own Optuna HPO (20 trials) on an inner validation split. The reported metrics are the **average across all test folds** — the most honest estimate of live performance.

---

## Setup

### 1. Install

```bash
git clone https://github.com/yourusername/19-ml-signal-filter.git
cd 19-ml-signal-filter
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit: KITE_API_KEY, KITE_ACCESS_TOKEN, SYMBOL, TIMEFRAME
```

Key training parameters:

```bash
FORWARD_CANDLES=10       # how many bars to look ahead for TP/SL
PROFIT_TARGET_PCT=0.5    # % gain = good trade label
STOP_LOSS_PCT=0.3        # % loss = bad trade label
N_FOLDS=8                # walk-forward folds
TRAIN_MONTHS=6           # training window per fold
MIN_PROBA=0.60           # classifier threshold (tune this!)
OPTUNA_TRIALS=50         # HPO trials (more = better, slower)
```

### 3. Train the model

```bash
# Full pipeline (recommended, ~20 min)
python train.py

# Fast version — no HPO, no walk-forward
python train.py --no-hpo --no-wf

# Train for SELL signals
python train.py --direction SELL

# Use cached data (skip Kite API call)
python train.py --no-fetch
```

Training output:
```
Step 1/6 — Loading OHLCV data…  21840 rows | 2022-01-03 → 2024-01-25
Step 2/6 — Engineering features…  feature matrix: (21640, 64)
Step 3/6 — Generating labels…  21640 total | 8123 positive (37.5%)
Step 4/6 — Walk-forward validation…
  Fold 1: precision=0.612  recall=0.481  AP=0.589  pass_rate=31.2%
  Fold 2: precision=0.634  recall=0.512  AP=0.611  pass_rate=28.9%
  ...
Step 5/6 — Training final model…
Step 6/6 — Generating charts…

Top 10 features:
   1. rsi14                    4821.3
   2. ema9_21_spread           3912.1
   3. macd_histogram           3544.7
   ...
```

### 4. Run tests

```bash
python -m pytest tests/ -v
```

### 5. Start the filter API

```bash
python api/signal_server.py

# Test it:
curl http://localhost:5001/health
```

### 6. Connect TradingView

In your Pine Script strategy, set the webhook alert URL to:
```
http://YOUR_SERVER:5001/signal
```

Alert JSON message format:
```json
{
  "action":      "{{strategy.order.action}}",
  "symbol":      "{{ticker}}",
  "exchange":    "{{exchange}}",
  "price":       {{close}},
  "strategy_id": "ORB_15min",
  "timestamp":   "{{time}}"
}
```

Set `ORCHESTRATOR_URL` in `.env` to forward passed signals to your execution bot (repo 14).

---

## Tuning the Threshold

`MIN_PROBA` is the most important parameter. Use the PR curve chart to choose:

| MIN_PROBA | Effect |
|---|---|
| 0.50 | Pass most signals. Higher recall, lower precision. More trades. |
| 0.60 | **Recommended starting point.** Balanced precision/recall. |
| 0.70 | Block most signals. Higher precision, lower recall. Fewer but better trades. |
| 0.80 | Very selective. May pass <10% of signals. Use only with large history. |

The calibration chart shows whether the model's predicted probabilities are reliable. A well-calibrated model at 0.65 confidence means ~65% of those signals actually hit the TP. If the model is poorly calibrated, adjust with a `CalibratedClassifierCV` wrapper.

---

## Charts Generated

| File | Description |
|---|---|
| `data/models/feature_importance.png` | XGBoost gain importance — top 20 features |
| `data/models/shap_summary.png` | SHAP beeswarm — feature direction and magnitude |
| `data/models/walk_forward_metrics.png` | Per-fold precision / recall / AP / pass-rate |
| `data/models/pr_curve.png` | Precision-Recall curve with current threshold marked |
| `data/models/calibration.png` | Predicted probability vs actual hit rate |

---

## Integration with Repo 14 (Live Orchestrator)

Set in `.env`:
```bash
ORCHESTRATOR_URL=http://localhost:5000/trade
```

The filter server POSTs passed signals to the orchestrator:
```json
{
  "action":        "BUY",
  "symbol":        "NIFTY 50",
  "ml_confidence": 0.73,
  "ml_filter":     "PASSED",
  ...original signal fields...
}
```

Optionally, in the orchestrator's `/trade` handler, check for `ml_filter == "PASSED"` before executing.

---

## Previous Repo (Required)

- **[01-zerodha-kite-setup](../01-zerodha-kite-setup)** — Kite auth
- **[02-market-data-pipeline](../02-market-data-pipeline)** — OHLCV data caching (used by DataFetcher)

---

## Next Repo

**[20-portfolio-manager](../20-portfolio-manager)** — run multiple strategies simultaneously with correlation-aware position sizing and aggregate risk limits.

---

## Disclaimer

Educational use only. Not financial advice. Past ML performance does not guarantee future trading results. A model that passes 65% precision in backtest may perform differently live due to market regime changes.
