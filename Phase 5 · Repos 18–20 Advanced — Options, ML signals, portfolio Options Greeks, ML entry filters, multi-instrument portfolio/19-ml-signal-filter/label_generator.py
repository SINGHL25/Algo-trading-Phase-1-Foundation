"""
features/label_generator.py
─────────────────────────────
Generates binary labels for training the XGBoost classifier.

Label logic (forward-looking):
  For each signal bar, look N candles ahead:
    - If price moves PROFIT_TARGET_PCT% in the signal direction first → label = 1 (good trade)
    - If price moves STOP_LOSS_PCT% against the signal direction first → label = 0 (bad trade)
    - If neither hit → label = 0 (inconclusive = bad for training data quality)

Signal directions:
    BUY  signal: we need price to rise by target%
    SELL signal: we need price to fall by target%

Why this labelling approach:
  This mimics the actual P&L outcome of placing a trade with a TP and SL.
  It avoids look-ahead bias (we only use future bars, not same-bar outcomes).
  The imbalance in labels (bad trades usually outnumber good) is handled
  by class_weight in XGBoost.

Usage:
    lg     = LabelGenerator()
    labels = lg.label_signals(ohlcv_df, signals_df)
    # signals_df: DataFrame with columns [datetime, direction]
    # direction: 'BUY' or 'SELL'
"""

import os
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

FORWARD_CANDLES  = int(float(os.getenv("FORWARD_CANDLES",  "10")))
PROFIT_TARGET_PCT = float(os.getenv("PROFIT_TARGET_PCT",   "0.5")) / 100.0
STOP_LOSS_PCT     = float(os.getenv("STOP_LOSS_PCT",       "0.3")) / 100.0


class LabelGenerator:
    def __init__(
        self,
        forward_candles:  int   = FORWARD_CANDLES,
        profit_target_pct: float = PROFIT_TARGET_PCT,
        stop_loss_pct:     float = STOP_LOSS_PCT,
    ):
        self.forward_candles   = forward_candles
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct     = stop_loss_pct

    def label_signals(
        self,
        ohlcv: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Assign outcome labels to a set of trading signals.

        Args:
            ohlcv   : OHLCV DataFrame — datetime, open, high, low, close, volume
            signals : DataFrame with columns [datetime, direction]
                      direction ∈ {'BUY', 'SELL'}

        Returns:
            DataFrame: signals joined with label (0 or 1), entry_price,
                       outcome, max_profit_pct, max_loss_pct
        """
        ohlcv   = ohlcv.sort_values("datetime").reset_index(drop=True)
        signals = signals.copy()
        signals["datetime"] = pd.to_datetime(signals["datetime"])

        results = []
        for _, sig in signals.iterrows():
            idx = ohlcv[ohlcv["datetime"] == sig["datetime"]].index
            if idx.empty:
                continue
            bar_idx    = idx[0]
            entry_price = float(ohlcv.loc[bar_idx, "close"])
            direction   = sig["direction"].upper()

            label, outcome, max_profit, max_loss = self._evaluate(
                ohlcv, bar_idx, entry_price, direction
            )

            results.append({
                "datetime":     sig["datetime"],
                "direction":    direction,
                "entry_price":  entry_price,
                "label":        label,
                "outcome":      outcome,
                "max_profit_pct": max_profit,
                "max_loss_pct":   max_loss,
            })

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).reset_index(drop=True)

    def label_all_bars(
        self,
        ohlcv: pd.DataFrame,
        direction: str = "BUY",
    ) -> pd.DataFrame:
        """
        Label every bar as if we had entered long/short there.
        Used for training when we don't have explicit Pine signals —
        we label every bar and let the model learn which bars would have
        worked.

        Returns DataFrame aligned to the ohlcv index with a 'label' column.
        """
        ohlcv  = ohlcv.sort_values("datetime").reset_index(drop=True)
        labels = np.full(len(ohlcv), np.nan)

        for i in range(len(ohlcv) - self.forward_candles):
            price = float(ohlcv.loc[i, "close"])
            lbl, _, _, _ = self._evaluate(ohlcv, i, price, direction)
            labels[i] = lbl

        result = ohlcv[["datetime", "close"]].copy()
        result["label"] = labels
        return result.dropna(subset=["label"]).reset_index(drop=True)

    def _evaluate(
        self,
        ohlcv: pd.DataFrame,
        bar_idx: int,
        entry_price: float,
        direction: str,
    ) -> tuple[int, str, float, float]:
        """
        Look forward from bar_idx and determine outcome.

        Returns:
            (label, outcome_str, max_profit_pct, max_loss_pct)
        """
        tp_price = entry_price * (1 + self.profit_target_pct) if direction == "BUY" \
                   else entry_price * (1 - self.profit_target_pct)
        sl_price = entry_price * (1 - self.stop_loss_pct)     if direction == "BUY" \
                   else entry_price * (1 + self.stop_loss_pct)

        max_profit = 0.0
        max_loss   = 0.0

        end_idx = min(bar_idx + self.forward_candles + 1, len(ohlcv))
        future  = ohlcv.iloc[bar_idx + 1 : end_idx]

        for _, row in future.iterrows():
            h = float(row["high"])
            l = float(row["low"])

            if direction == "BUY":
                profit_pct = (h - entry_price) / entry_price
                loss_pct   = (entry_price - l) / entry_price
            else:
                profit_pct = (entry_price - l) / entry_price
                loss_pct   = (h - entry_price) / entry_price

            max_profit = max(max_profit, profit_pct)
            max_loss   = max(max_loss,   loss_pct)

            # TP hit first
            if direction == "BUY" and h >= tp_price:
                return 1, "TP_HIT", max_profit, max_loss
            if direction == "SELL" and l <= tp_price:
                return 1, "TP_HIT", max_profit, max_loss

            # SL hit first
            if direction == "BUY" and l <= sl_price:
                return 0, "SL_HIT", max_profit, max_loss
            if direction == "SELL" and h >= sl_price:
                return 0, "SL_HIT", max_profit, max_loss

        # Neither hit within forward window → inconclusive
        return 0, "EXPIRED", max_profit, max_loss

    def class_balance(self, labels: pd.Series) -> dict:
        """Return class distribution stats."""
        counts = labels.value_counts()
        total  = len(labels)
        return {
            "total":     total,
            "positive":  int(counts.get(1, 0)),
            "negative":  int(counts.get(0, 0)),
            "pos_pct":   round(counts.get(1, 0) / total * 100, 1),
            "neg_pct":   round(counts.get(0, 0) / total * 100, 1),
        }
