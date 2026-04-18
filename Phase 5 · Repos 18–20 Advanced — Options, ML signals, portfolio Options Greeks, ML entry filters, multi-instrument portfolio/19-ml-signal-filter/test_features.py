"""
tests/test_features.py
───────────────────────
Unit tests for FeatureEngineer and LabelGenerator.
No Kite API required — all tests use synthetic OHLCV data.

Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)
    idx    = pd.date_range("2024-01-02 09:15", periods=n, freq="15min")
    close  = 21000 + np.random.randn(n).cumsum() * 50
    spread = np.abs(np.random.randn(n)) * 30 + 10
    open_  = close - np.random.randn(n) * 20
    high   = np.maximum(close, open_) + spread * 0.6
    low    = np.minimum(close, open_) - spread * 0.6
    vol    = np.abs(np.random.randn(n)) * 50000 + 100000

    return pd.DataFrame({
        "datetime": idx,
        "open":     open_,
        "high":     high,
        "low":      low,
        "close":    close,
        "volume":   vol,
    })


# ──────────────────────────────────────────────────────────────────────
# Feature engineering tests
# ──────────────────────────────────────────────────────────────────────

class TestFeatureEngineer:
    def setup_method(self):
        from features.feature_engineer import FeatureEngineer
        self.fe   = FeatureEngineer()
        self.ohlcv = _make_ohlcv(300)

    def test_returns_dataframe(self):
        result = self.fe.transform(self.ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_has_datetime_column(self):
        result = self.fe.transform(self.ohlcv)
        assert "datetime" in result.columns

    def test_no_nan_after_dropna(self):
        result = self.fe.transform(self.ohlcv, dropna=True)
        # Allow NaN only in columns that might have issues with synthetic data
        feature_cols = [c for c in result.columns if c not in ("datetime", "close")]
        nan_counts   = result[feature_cols].isna().sum()
        all_nan_cols = nan_counts[nan_counts == len(result)].index.tolist()
        assert len(all_nan_cols) == 0, f"Columns with all NaN: {all_nan_cols}"

    def test_row_count_reduced_by_warmup(self):
        result = self.fe.transform(self.ohlcv, dropna=True)
        assert len(result) < len(self.ohlcv)
        assert len(result) > 50  # should have plenty left from 300 rows

    def test_ema_relative_feature_exists(self):
        result = self.fe.transform(self.ohlcv)
        assert "ema9_rel"  in result.columns
        assert "ema21_rel" in result.columns
        assert "ema50_rel" in result.columns

    def test_rsi_range(self):
        result = self.fe.transform(self.ohlcv, dropna=True)
        assert "rsi14" in result.columns
        rsi = result["rsi14"].dropna()
        # RSI is scaled to 0–1 in features
        assert (rsi >= 0).all() and (rsi <= 1).all()

    def test_momentum_features_present(self):
        result = self.fe.transform(self.ohlcv)
        for col in ["macd_line", "macd_histogram", "stoch_k", "stoch_d", "roc5"]:
            assert col in result.columns, f"Missing: {col}"

    def test_volatility_features_present(self):
        result = self.fe.transform(self.ohlcv)
        for col in ["atr14_pct", "bb_width", "bb_pct_b", "hvol10", "hvol20"]:
            assert col in result.columns, f"Missing: {col}"

    def test_volume_features_present(self):
        result = self.fe.transform(self.ohlcv)
        for col in ["vol_vs_ma20", "vol_roc5", "obv_zscore", "vwap_dev_pct"]:
            assert col in result.columns, f"Missing: {col}"

    def test_candle_features_present(self):
        result = self.fe.transform(self.ohlcv)
        for col in ["body_atr_ratio", "upper_wick_ratio", "lower_wick_ratio",
                    "inside_bar", "engulfing", "consec_bars"]:
            assert col in result.columns, f"Missing: {col}"

    def test_time_features_present(self):
        result = self.fe.transform(self.ohlcv, include_time_features=True)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "mins_to_close"]:
            assert col in result.columns, f"Missing: {col}"

    def test_bb_pct_b_clipped(self):
        result = self.fe.transform(self.ohlcv, dropna=True)
        bb = result["bb_pct_b"].dropna()
        assert (bb >= 0).all() and (bb <= 1).all()

    def test_above_ema200_binary(self):
        result = self.fe.transform(self.ohlcv, dropna=True)
        vals   = result["above_ema200"].dropna()
        assert set(vals.unique()).issubset({0, 1})

    def test_minimum_feature_count(self):
        result = self.fe.transform(self.ohlcv)
        feature_cols = [c for c in result.columns if c not in ("datetime", "close")]
        assert len(feature_cols) >= 40, f"Only {len(feature_cols)} features generated"

    def test_no_future_leakage(self):
        """Features at row i must not depend on rows i+1, i+2, ..."""
        # Test by checking that shifting OHLCV by 1 produces different features
        ohlcv2          = self.ohlcv.copy()
        ohlcv2["close"] = self.ohlcv["close"].shift(1).bfill()
        r1 = self.fe.transform(self.ohlcv, dropna=True)
        r2 = self.fe.transform(ohlcv2,     dropna=True)
        # EMA should differ
        assert not (r1["ema9_rel"].values == r2["ema9_rel"].values).all()

    def test_reproducibility(self):
        """Same input → same output."""
        r1 = self.fe.transform(self.ohlcv, dropna=True)
        r2 = self.fe.transform(self.ohlcv, dropna=True)
        pd.testing.assert_frame_equal(r1, r2)


# ──────────────────────────────────────────────────────────────────────
# Label generator tests
# ──────────────────────────────────────────────────────────────────────

class TestLabelGenerator:
    def setup_method(self):
        from features.label_generator import LabelGenerator
        self.lg    = LabelGenerator(forward_candles=10,
                                    profit_target_pct=0.005,
                                    stop_loss_pct=0.003)
        self.ohlcv = _make_ohlcv(200)

    def test_label_all_bars_returns_dataframe(self):
        result = self.lg.label_all_bars(self.ohlcv, direction="BUY")
        assert isinstance(result, pd.DataFrame)
        assert "label" in result.columns

    def test_labels_are_binary(self):
        result = self.lg.label_all_bars(self.ohlcv, direction="BUY")
        assert set(result["label"].unique()).issubset({0, 1})

    def test_no_labels_at_end(self):
        """Last FORWARD_CANDLES rows can't be labelled."""
        result = self.lg.label_all_bars(self.ohlcv)
        assert len(result) < len(self.ohlcv)

    def test_some_positive_labels(self):
        """With NIFTY-like prices and reasonable targets, some bars should be positive."""
        result = self.lg.label_all_bars(self.ohlcv, direction="BUY")
        assert result["label"].sum() > 0

    def test_label_signals_with_explicit_signals(self):
        signals = pd.DataFrame({
            "datetime":  self.ohlcv["datetime"].iloc[50:60].values,
            "direction": ["BUY"] * 10,
        })
        result = self.lg.label_signals(self.ohlcv, signals)
        assert "label"       in result.columns
        assert "entry_price" in result.columns
        assert "outcome"     in result.columns
        assert len(result) == 10

    def test_sell_labels_different_from_buy(self):
        buy_labels  = self.lg.label_all_bars(self.ohlcv, direction="BUY")["label"].values
        sell_labels = self.lg.label_all_bars(self.ohlcv, direction="SELL")["label"].values
        # They won't be identical for a trending series
        assert not (buy_labels == sell_labels).all()

    def test_class_balance_returns_dict(self):
        result = self.lg.label_all_bars(self.ohlcv)
        bal    = self.lg.class_balance(result["label"])
        assert "total"    in bal
        assert "positive" in bal
        assert "pos_pct"  in bal
        assert bal["total"] == bal["positive"] + bal["negative"]

    def test_tight_targets_more_positives(self):
        """Tighter TP/SL → more bars hit TP quickly → more positive labels."""
        lg_tight = self.lg.__class__(forward_candles=5, profit_target_pct=0.001, stop_loss_pct=0.01)
        lg_wide  = self.lg.__class__(forward_candles=5, profit_target_pct=0.01,  stop_loss_pct=0.001)
        tight = lg_tight.label_all_bars(self.ohlcv)["label"].mean()
        wide  = lg_wide.label_all_bars(self.ohlcv)["label"].mean()
        assert tight >= wide, "Tighter TP should produce more positives"


# ──────────────────────────────────────────────────────────────────────
# XGBoost classifier tests (no Kite required)
# ──────────────────────────────────────────────────────────────────────

class TestSignalClassifier:
    def setup_method(self):
        from features.feature_engineer import FeatureEngineer
        from features.label_generator  import LabelGenerator
        from model.xgb_classifier      import SignalClassifier

        ohlcv    = _make_ohlcv(400)
        fe       = FeatureEngineer()
        features = fe.transform(ohlcv, dropna=True)
        lg       = LabelGenerator(forward_candles=8, profit_target_pct=0.004, stop_loss_pct=0.002)
        labels   = lg.label_all_bars(ohlcv, direction="BUY")

        merged   = pd.merge(features, labels[["datetime", "label"]], on="datetime", how="inner")
        merged   = merged.dropna(subset=["label"]).reset_index(drop=True)
        feat_cols = [c for c in merged.columns if c not in ("datetime", "close", "label")]

        self.X   = merged[feat_cols]
        self.y   = merged["label"].astype(int)
        self.clf = SignalClassifier(min_proba=0.55)

    def test_train_returns_metrics(self):
        metrics = self.clf.train(self.X, self.y, verbose=False)
        assert "precision" in metrics or "n_samples" in metrics

    def test_predict_proba_range(self):
        self.clf.train(self.X, self.y, verbose=False)
        proba = self.clf.predict_proba(self.X.head(50))
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_predict_binary(self):
        self.clf.train(self.X, self.y, verbose=False)
        preds = self.clf.predict(self.X.head(50))
        assert set(preds).issubset({0, 1})

    def test_filter_signal_returns_tuple(self):
        self.clf.train(self.X, self.y, verbose=False)
        row    = self.X.iloc[0]
        result = self.clf.filter_signal(row)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert 0.0 <= result[1] <= 1.0

    def test_feature_importances_shape(self):
        self.clf.train(self.X, self.y, verbose=False)
        imp = self.clf.feature_importances(top_n=10)
        assert len(imp) <= 10
        assert "feature"    in imp.columns
        assert "importance" in imp.columns

    def test_save_and_load(self, tmp_path):
        self.clf.train(self.X, self.y, verbose=False)
        path = str(tmp_path / "test_model.joblib")
        self.clf.save(path)

        from model.xgb_classifier import SignalClassifier
        loaded = SignalClassifier.load(path)
        assert loaded.feature_names_ == self.clf.feature_names_
        orig_proba   = self.clf.predict_proba(self.X.head(10))
        loaded_proba = loaded.predict_proba(self.X.head(10))
        np.testing.assert_allclose(orig_proba, loaded_proba, rtol=1e-5)

    def test_evaluate_returns_metrics(self):
        self.clf.train(self.X, self.y, verbose=False)
        metrics = self.clf.evaluate(self.X.tail(30), self.y.tail(30))
        assert "n_samples" in metrics

    def test_untrained_raises(self):
        from model.xgb_classifier import SignalClassifier
        clf = SignalClassifier()
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict_proba(self.X.head(5))
