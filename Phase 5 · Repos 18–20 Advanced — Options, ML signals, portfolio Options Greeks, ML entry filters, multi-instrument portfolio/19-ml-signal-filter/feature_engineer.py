"""
features/feature_engineer.py
──────────────────────────────
Transforms raw OHLCV data into an ML feature matrix.

Feature groups (60+ features total):

  TREND (12)
    EMA 9, 21, 50, 200 — raw and relative to close
    EMA 9/21 spread, EMA 21/50 spread
    Price vs EMA200 (bull/bear regime)
    ADX(14) — trend strength
    SuperTrend direction

  MOMENTUM (14)
    RSI 7, 14, 21
    RSI slope (1-bar and 3-bar)
    MACD line, signal, histogram
    MACD histogram slope
    Rate of change (ROC) 5, 10, 20
    Stochastic %K, %D

  VOLATILITY (10)
    ATR 14 (raw and % of price)
    ATR 14 / ATR 50 (regime)
    Bollinger Band width (% of price)
    Bollinger %B
    Keltner channel position
    Historical volatility 10, 20 days
    Parkinson volatility (H/L range)

  VOLUME (8)
    Volume vs 20-bar MA
    Volume ratio (bar vs 5-bar avg)
    OBV normalised
    VWAP deviation %
    Volume ROC 5
    Accumulation/Distribution oscillator

  CANDLE STRUCTURE (10)
    Body size / ATR
    Upper wick / body ratio
    Lower wick / body ratio
    Gap open % (vs prior close)
    Inside bar flag
    Engulfing pattern flag
    Number of consecutive green/red bars

  TIME (6)
    Hour of day (sin/cos encoded)
    Day of week (sin/cos encoded)
    Week of month
    Minutes to market close

  CROSS-TIMEFRAME (4)  [optional — requires daily data]
    Daily RSI, daily ATR%, daily trend direction, daily EMA50 vs EMA200

All features are computed in a single pass over the OHLCV DataFrame.
NaN rows (lookback warmup) are dropped before returning.
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)

# ── Optional: use 'ta' library for some indicators ───────────────────
try:
    import ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False
    logger.warning("'ta' library not available — some features will use manual calc.")


class FeatureEngineer:
    """
    Transforms a raw OHLCV DataFrame into an ML-ready feature matrix.

    Usage:
        fe = FeatureEngineer()
        features_df = fe.transform(ohlcv_df)
    """

    # Which columns are used downstream (ordered, for consistent feature names)
    FEATURE_GROUPS = [
        "trend", "momentum", "volatility", "volume", "candle", "time"
    ]

    def transform(
        self,
        df: pd.DataFrame,
        dropna: bool = True,
        include_time_features: bool = True,
    ) -> pd.DataFrame:
        """
        Args:
            df: raw OHLCV with columns [datetime, open, high, low, close, volume]
            dropna: drop warmup NaN rows (recommended)
        Returns:
            DataFrame of features aligned to the input index (minus warmup rows)
        """
        df = df.copy()
        df = df.sort_values("datetime").reset_index(drop=True)

        open_  = df["open"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)

        feats = {}

        # ── Trend features ────────────────────────────────────────────
        feats.update(self._trend_features(close, high, low))

        # ── Momentum features ─────────────────────────────────────────
        feats.update(self._momentum_features(close, high, low))

        # ── Volatility features ───────────────────────────────────────
        feats.update(self._volatility_features(close, high, low))

        # ── Volume features ───────────────────────────────────────────
        feats.update(self._volume_features(close, high, low, volume))

        # ── Candle structure features ─────────────────────────────────
        feats.update(self._candle_features(open_, high, low, close))

        # ── Time features ─────────────────────────────────────────────
        if include_time_features and "datetime" in df.columns:
            feats.update(self._time_features(df["datetime"]))

        result = pd.DataFrame(feats, index=df.index)
        result.insert(0, "datetime", df["datetime"])
        result.insert(1, "close", close)

        if dropna:
            n_before = len(result)
            result   = result.dropna().reset_index(drop=True)
            logger.debug(f"Feature matrix: {n_before} → {len(result)} rows after dropna")

        return result

    # ── TREND ─────────────────────────────────────────────────────────

    def _trend_features(self, close: pd.Series, high: pd.Series, low: pd.Series) -> dict:
        feats = {}

        for p in [9, 21, 50, 200]:
            ema = _ema(close, p)
            feats[f"ema{p}_rel"]    = (close - ema) / close        # % distance from price
            feats[f"ema{p}_slope"]  = ema.diff(3) / ema.shift(3)   # 3-bar slope

        feats["ema9_21_spread"]  = (_ema(close, 9) - _ema(close, 21)) / close
        feats["ema21_50_spread"] = (_ema(close, 21) - _ema(close, 50)) / close
        feats["ema50_200_spread"]= (_ema(close, 50) - _ema(close, 200)) / close
        feats["above_ema200"]    = (close > _ema(close, 200)).astype(int)

        # ADX
        feats["adx14"] = _adx(high, low, close, 14)

        # SuperTrend direction (1 = bullish, -1 = bearish)
        feats["supertrend_dir"] = _supertrend_direction(high, low, close, period=10, multiplier=3.0)

        return feats

    # ── MOMENTUM ──────────────────────────────────────────────────────

    def _momentum_features(self, close: pd.Series, high: pd.Series, low: pd.Series) -> dict:
        feats = {}

        for p in [7, 14, 21]:
            rsi              = _rsi(close, p)
            feats[f"rsi{p}"] = rsi / 100.0
            feats[f"rsi{p}_slope1"] = rsi.diff(1) / 10.0
            feats[f"rsi{p}_slope3"] = rsi.diff(3) / 10.0

        # MACD (12, 26, 9)
        ema12     = _ema(close, 12)
        ema26     = _ema(close, 26)
        macd_line = ema12 - ema26
        macd_sig  = _ema(macd_line, 9)
        macd_hist = macd_line - macd_sig

        feats["macd_line"]      = macd_line / close
        feats["macd_signal"]    = macd_sig  / close
        feats["macd_histogram"] = macd_hist / close
        feats["macd_hist_slope"]= macd_hist.diff(2) / close

        # ROC
        for p in [5, 10, 20]:
            feats[f"roc{p}"] = close.pct_change(p)

        # Stochastic %K, %D (14, 3)
        stoch_k, stoch_d = _stochastic(high, low, close, 14, 3)
        feats["stoch_k"] = stoch_k / 100.0
        feats["stoch_d"] = stoch_d / 100.0

        return feats

    # ── VOLATILITY ────────────────────────────────────────────────────

    def _volatility_features(self, close: pd.Series, high: pd.Series, low: pd.Series) -> dict:
        feats = {}

        atr14 = _atr(high, low, close, 14)
        atr50 = _atr(high, low, close, 50)

        feats["atr14_pct"]        = atr14 / close
        feats["atr14_atr50_ratio"]= atr14 / (atr50 + 1e-9)

        # Bollinger Bands (20, 2)
        sma20     = close.rolling(20).mean()
        std20     = close.rolling(20).std()
        bb_upper  = sma20 + 2 * std20
        bb_lower  = sma20 - 2 * std20
        bb_width  = (bb_upper - bb_lower) / (sma20 + 1e-9)
        bb_pct_b  = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

        feats["bb_width"]  = bb_width
        feats["bb_pct_b"]  = bb_pct_b.clip(0, 1)

        # Historical volatility (log returns std)
        log_ret = np.log(close / close.shift(1))
        feats["hvol10"] = log_ret.rolling(10).std() * np.sqrt(252)
        feats["hvol20"] = log_ret.rolling(20).std() * np.sqrt(252)

        # Parkinson volatility (uses H/L range, less noise than close-to-close)
        hl_ratio = np.log(high / (low + 1e-9))
        feats["pvol14"] = (hl_ratio ** 2).rolling(14).mean().apply(
            lambda x: np.sqrt(x / (4 * np.log(2)) * 252)
        )

        return feats

    # ── VOLUME ────────────────────────────────────────────────────────

    def _volume_features(
        self, close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series
    ) -> dict:
        feats = {}

        vol_ma20 = volume.rolling(20).mean()
        vol_ma5  = volume.rolling(5).mean()

        feats["vol_vs_ma20"]  = volume / (vol_ma20 + 1)
        feats["vol_vs_ma5"]   = volume / (vol_ma5  + 1)
        feats["vol_roc5"]     = volume.pct_change(5)

        # OBV (normalised — z-score over rolling 50 bars)
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_std  = obv.rolling(50).std()
        obv_mean = obv.rolling(50).mean()
        feats["obv_zscore"] = (obv - obv_mean) / (obv_std + 1e-9)

        # VWAP deviation (intraday VWAP reset not possible on daily data,
        # so we use rolling 20-bar VWAP as a proxy)
        typical = (high + low + close) / 3
        vwap    = (typical * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
        feats["vwap_dev_pct"] = (close - vwap) / (vwap + 1e-9)

        # Accumulation / Distribution oscillator
        clv  = ((close - low) - (high - close)) / (high - low + 1e-9)
        ad   = (clv * volume).cumsum()
        ad_fast = _ema(ad, 3)
        ad_slow = _ema(ad, 10)
        feats["ad_osc"] = (ad_fast - ad_slow) / (close + 1e-9)

        return feats

    # ── CANDLE STRUCTURE ──────────────────────────────────────────────

    def _candle_features(
        self,
        open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> dict:
        feats = {}

        body   = (close - open_).abs()
        atr14  = _atr(high, low, close, 14)
        rng    = high - low + 1e-9

        feats["body_atr_ratio"]     = body / (atr14 + 1e-9)
        feats["upper_wick_ratio"]   = (high - close.clip(lower=open_)) / rng
        feats["lower_wick_ratio"]   = (open_.clip(upper=close) - low) / rng
        feats["candle_direction"]   = np.sign(close - open_)
        feats["gap_open_pct"]       = (open_ - close.shift(1)) / (close.shift(1) + 1e-9)

        # Inside bar (high < prev high and low > prev low)
        feats["inside_bar"] = (
            (high < high.shift(1)) & (low > low.shift(1))
        ).astype(int)

        # Engulfing (body engulfs prior candle's body)
        prior_body_high = open_.shift(1).clip(lower=close.shift(1))
        prior_body_low  = open_.shift(1).clip(upper=close.shift(1))
        curr_body_high  = open_.clip(lower=close)
        curr_body_low   = open_.clip(upper=close)
        feats["engulfing"] = (
            (curr_body_high > prior_body_high) & (curr_body_low < prior_body_low)
        ).astype(int)

        # Consecutive green / red bars (capped at 5)
        direction = np.sign(close - open_)
        consec = direction.copy() * 0
        for i in range(1, len(direction)):
            if direction.iloc[i] == direction.iloc[i - 1] and direction.iloc[i] != 0:
                consec.iloc[i] = consec.iloc[i - 1] + direction.iloc[i]
            else:
                consec.iloc[i] = direction.iloc[i]
        feats["consec_bars"] = consec.clip(-5, 5)

        return feats

    # ── TIME ──────────────────────────────────────────────────────────

    def _time_features(self, dt: pd.Series) -> dict:
        feats = {}
        dt_ist = pd.to_datetime(dt)

        hour       = dt_ist.dt.hour + dt_ist.dt.minute / 60.0
        dow        = dt_ist.dt.dayofweek.astype(float)
        week_month = ((dt_ist.dt.day - 1) // 7 + 1).astype(float)

        # Cyclical encoding (sin/cos so Mon and Fri are "close")
        feats["hour_sin"]  = np.sin(2 * np.pi * hour  / 24.0)
        feats["hour_cos"]  = np.cos(2 * np.pi * hour  / 24.0)
        feats["dow_sin"]   = np.sin(2 * np.pi * dow   / 5.0)
        feats["dow_cos"]   = np.cos(2 * np.pi * dow   / 5.0)
        feats["week_month"]= week_month / 5.0

        # Minutes to 15:30 close (normalised 0–1)
        mins_to_close = (15 * 60 + 30) - (dt_ist.dt.hour * 60 + dt_ist.dt.minute)
        feats["mins_to_close"] = (mins_to_close.clip(0, 375) / 375.0)

        return feats

    def feature_names(self, include_time: bool = True) -> list[str]:
        """Return the list of feature column names (without datetime/close)."""
        # Build on tiny dummy data to get names
        idx = pd.date_range("2024-01-02 09:15", periods=250, freq="15min")
        dummy = pd.DataFrame({
            "datetime": idx,
            "open":  100 + np.random.randn(250).cumsum(),
            "high":  100 + np.random.randn(250).cumsum() + 0.5,
            "low":   100 + np.random.randn(250).cumsum() - 0.5,
            "close": 100 + np.random.randn(250).cumsum(),
            "volume":np.abs(np.random.randn(250)) * 1000,
        })
        dummy["high"]  = dummy[["open","high","close"]].max(axis=1)
        dummy["low"]   = dummy[["open","low", "close"]].min(axis=1)
        feat_df = self.transform(dummy, dropna=True, include_time_features=include_time)
        return [c for c in feat_df.columns if c not in ("datetime", "close")]


# ──────────────────────────────────────────────────────────────────────
# Technical indicator helpers (pure numpy/pandas, no TA-Lib required)
# ──────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / (avg_l + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr      = _atr(high, low, close, 1)   # single-bar TR
    dm_plus  = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    # Where DM+ < DM-, set DM+ to 0 and vice versa
    cond     = dm_plus >= dm_minus
    dm_plus  = dm_plus.where(cond, 0)
    dm_minus = dm_minus.where(~cond, 0)

    atr_p    = tr.ewm(com=period - 1, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(com=period-1, adjust=False).mean()  / (atr_p + 1e-9)
    di_minus = 100 * dm_minus.ewm(com=period-1,adjust=False).mean() / (atr_p + 1e-9)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    return dx.ewm(com=period - 1, adjust=False).mean()


def _stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    lowest  = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    stoch_k = 100 * (close - lowest) / (highest - lowest + 1e-9)
    stoch_d = stoch_k.rolling(d_period).mean()
    return stoch_k, stoch_d


def _supertrend_direction(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 10, multiplier: float = 3.0,
) -> pd.Series:
    """Return +1 (bullish) / -1 (bearish) SuperTrend direction."""
    atr   = _atr(high, low, close, period)
    mid   = (high + low) / 2
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr

    direction = pd.Series(1, index=close.index)
    final_ub  = upper.copy()
    final_lb  = lower.copy()

    for i in range(1, len(close)):
        final_ub.iloc[i] = (
            upper.iloc[i] if upper.iloc[i] < final_ub.iloc[i-1]
            or close.iloc[i-1] > final_ub.iloc[i-1]
            else final_ub.iloc[i-1]
        )
        final_lb.iloc[i] = (
            lower.iloc[i] if lower.iloc[i] > final_lb.iloc[i-1]
            or close.iloc[i-1] < final_lb.iloc[i-1]
            else final_lb.iloc[i-1]
        )
        if direction.iloc[i-1] == -1 and close.iloc[i] > final_ub.iloc[i]:
            direction.iloc[i] = 1
        elif direction.iloc[i-1] == 1 and close.iloc[i] < final_lb.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

    return direction
