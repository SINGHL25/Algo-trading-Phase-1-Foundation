"""
backtest/walk_forward.py
─────────────────────────
Walk-forward validation for the XGBoost signal filter.

Why walk-forward (not k-fold cross-validation)?
  Financial time series have temporal dependencies. Standard k-fold shuffles
  data randomly, causing look-ahead bias (training on future data to predict
  the past). Walk-forward strictly trains on past data only, then tests on
  the immediately following unseen period — mirroring real deployment.

Walk-forward scheme:
  ┌────────────────────────────────────────────────────────────────┐
  │ FOLD 1: Train [Jan–Jun]     │ Test [Jul]                       │
  │ FOLD 2: Train [Feb–Jul]     │ Test [Aug]   (expanding window)  │
  │ FOLD 3: Train [Mar–Aug]     │ Test [Sep]                       │
  │  ...                                                           │
  │ FOLD N: Train [N-6..N-1]   │ Test [N]                         │
  └────────────────────────────────────────────────────────────────┘

For each fold:
  1. Build feature matrix from training OHLCV
  2. Generate labels (forward TP/SL simulation)
  3. Run Optuna HPO on inner validation split (last 15% of train)
  4. Train final model on full train fold
  5. Evaluate on test fold
  6. Record: precision, recall, F1, AP, pass-rate, fold dates

Outputs a WalkForwardResult with per-fold metrics + aggregate stats.
"""

import os
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Optional
from pathlib import Path

from features.feature_engineer import FeatureEngineer
from features.label_generator  import LabelGenerator
from model.xgb_classifier      import SignalClassifier, tune_hyperparameters

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

TRAIN_MONTHS = int(os.getenv("TRAIN_MONTHS", "6"))
TEST_MONTHS  = int(os.getenv("TEST_MONTHS",  "1"))
N_FOLDS      = int(os.getenv("N_FOLDS",      "8"))
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "50"))

RESULTS_DIR = Path("data/models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FoldResult:
    fold_n:        int
    train_start:   date
    train_end:     date
    test_start:    date
    test_end:      date
    n_train:       int
    n_test:        int
    n_pos_train:   int
    n_pos_test:    int
    precision:     float
    recall:        float
    f1:            float
    roc_auc:       float
    avg_precision: float
    pass_rate_pct: float
    best_params:   dict  = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    folds:           list[FoldResult]
    feature_names:   list[str]
    direction:       str

    @property
    def mean_precision(self) -> float:
        return float(np.mean([f.precision for f in self.folds]))

    @property
    def mean_recall(self) -> float:
        return float(np.mean([f.recall for f in self.folds]))

    @property
    def mean_f1(self) -> float:
        return float(np.mean([f.f1 for f in self.folds]))

    @property
    def mean_avg_precision(self) -> float:
        return float(np.mean([f.avg_precision for f in self.folds]))

    @property
    def mean_pass_rate(self) -> float:
        return float(np.mean([f.pass_rate_pct for f in self.folds]))

    def summary_df(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            rows.append({
                "fold":           f.fold_n,
                "train":          f"{f.train_start} → {f.train_end}",
                "test":           f"{f.test_start} → {f.test_end}",
                "n_train":        f.n_train,
                "n_test":         f.n_test,
                "precision":      f.precision,
                "recall":         f.recall,
                "f1":             f.f1,
                "avg_precision":  f.avg_precision,
                "pass_rate_%":    f.pass_rate_pct,
            })
        rows.append({
            "fold":          "MEAN",
            "train":         "—",
            "test":          "—",
            "n_train":       int(np.mean([f.n_train for f in self.folds])),
            "n_test":        int(np.mean([f.n_test  for f in self.folds])),
            "precision":     round(self.mean_precision,     3),
            "recall":        round(self.mean_recall,        3),
            "f1":            round(self.mean_f1,            3),
            "avg_precision": round(self.mean_avg_precision, 3),
            "pass_rate_%":   round(self.mean_pass_rate,     1),
        })
        return pd.DataFrame(rows)

    def to_csv(self, path: str = "data/models/walk_forward_results.csv"):
        self.summary_df().to_csv(path, index=False)
        logger.info(f"Walk-forward results saved to {path}")


class WalkForwardValidator:
    def __init__(
        self,
        train_months: int  = TRAIN_MONTHS,
        test_months:  int  = TEST_MONTHS,
        n_folds:      int  = N_FOLDS,
        direction:    str  = "BUY",
        run_hpo:      bool = True,
    ):
        self.train_months = train_months
        self.test_months  = test_months
        self.n_folds      = n_folds
        self.direction    = direction.upper()
        self.run_hpo      = run_hpo
        self.fe           = FeatureEngineer()
        self.lg           = LabelGenerator()

    def run(self, ohlcv: pd.DataFrame) -> WalkForwardResult:
        """
        Run full walk-forward validation on the provided OHLCV data.

        Args:
            ohlcv: DataFrame with datetime, open, high, low, close, volume

        Returns:
            WalkForwardResult with per-fold metrics
        """
        ohlcv     = ohlcv.sort_values("datetime").reset_index(drop=True)
        ohlcv["datetime"] = pd.to_datetime(ohlcv["datetime"])

        # Build full feature matrix + labels once (labels computed per-fold below)
        logger.info("Computing full feature matrix…")
        features_all = self.fe.transform(ohlcv, dropna=True)
        labels_all   = self.lg.label_all_bars(ohlcv, direction=self.direction)

        # Align features and labels on datetime
        merged = pd.merge(features_all, labels_all[["datetime", "label"]],
                          on="datetime", how="inner")
        merged = merged.dropna(subset=["label"]).reset_index(drop=True)
        merged["datetime"] = pd.to_datetime(merged["datetime"])

        feature_cols = [
            c for c in merged.columns if c not in ("datetime", "close", "label")
        ]

        # ── Generate fold date ranges ──────────────────────────────────
        all_dates  = merged["datetime"].dt.date
        global_end = all_dates.max()

        fold_dates = []
        for fold_n in range(self.n_folds - 1, -1, -1):
            test_end   = global_end - relativedelta(months=fold_n * self.test_months)
            test_start = test_end   - relativedelta(months=self.test_months) + relativedelta(days=1)
            train_end  = test_start - relativedelta(days=1)
            train_start= train_end  - relativedelta(months=self.train_months) + relativedelta(days=1)
            fold_dates.append((train_start, train_end, test_start, test_end))

        fold_results = []

        for fold_n, (train_start, train_end, test_start, test_end) in enumerate(fold_dates, 1):
            logger.info(
                f"Fold {fold_n}/{self.n_folds}: "
                f"train {train_start}→{train_end} | "
                f"test  {test_start}→{test_end}"
            )

            train_mask = (all_dates >= train_start) & (all_dates <= train_end)
            test_mask  = (all_dates >= test_start)  & (all_dates <= test_end)

            train_df = merged[train_mask].reset_index(drop=True)
            test_df  = merged[test_mask ].reset_index(drop=True)

            if len(train_df) < 100:
                logger.warning(f"Fold {fold_n}: insufficient training data ({len(train_df)} rows). Skipping.")
                continue
            if len(test_df) < 10:
                logger.warning(f"Fold {fold_n}: insufficient test data ({len(test_df)} rows). Skipping.")
                continue

            X_train = train_df[feature_cols]
            y_train = train_df["label"].astype(int)
            X_test  = test_df[feature_cols]
            y_test  = test_df["label"].astype(int)

            n_pos_train = int(y_train.sum())
            n_pos_test  = int(y_test.sum())

            if n_pos_train < 5:
                logger.warning(f"Fold {fold_n}: too few positive labels in train ({n_pos_train}). Skipping.")
                continue

            # ── HPO on inner val split ─────────────────────────────────
            best_params = None
            if self.run_hpo and OPTUNA_TRIALS > 0:
                val_size   = max(int(len(X_train) * 0.15), 20)
                X_tr_inner = X_train.iloc[:-val_size]
                y_tr_inner = y_train.iloc[:-val_size]
                X_val_inner= X_train.iloc[-val_size:]
                y_val_inner= y_train.iloc[-val_size:]
                try:
                    best_params = tune_hyperparameters(
                        X_tr_inner, y_tr_inner,
                        X_val_inner, y_val_inner,
                        n_trials=min(OPTUNA_TRIALS, 20),  # fewer trials per fold
                    )
                except Exception as e:
                    logger.warning(f"Fold {fold_n}: HPO failed ({e}). Using defaults.")

            # ── Train + evaluate ──────────────────────────────────────
            clf = SignalClassifier(params=best_params)
            clf.train(X_train, y_train, verbose=False)
            metrics = clf.evaluate(X_test, y_test)

            fold_result = FoldResult(
                fold_n        = fold_n,
                train_start   = train_start,
                train_end     = train_end,
                test_start    = test_start,
                test_end      = test_end,
                n_train       = len(train_df),
                n_test        = len(test_df),
                n_pos_train   = n_pos_train,
                n_pos_test    = n_pos_test,
                precision     = metrics.get("precision",    0.0),
                recall        = metrics.get("recall",       0.0),
                f1            = metrics.get("f1",           0.0),
                roc_auc       = metrics.get("roc_auc",      0.0),
                avg_precision = metrics.get("avg_prec",     0.0),
                pass_rate_pct = metrics.get("pass_rate_pct",0.0),
                best_params   = best_params or {},
            )
            fold_results.append(fold_result)
            logger.info(
                f"  Fold {fold_n} results: "
                f"precision={fold_result.precision:.3f} "
                f"recall={fold_result.recall:.3f} "
                f"AP={fold_result.avg_precision:.3f} "
                f"pass_rate={fold_result.pass_rate_pct:.1f}%"
            )

        result = WalkForwardResult(
            folds         = fold_results,
            feature_names = feature_cols,
            direction     = self.direction,
        )

        logger.info(
            f"Walk-forward complete: {len(fold_results)} folds | "
            f"Mean precision={result.mean_precision:.3f} | "
            f"Mean AP={result.mean_avg_precision:.3f}"
        )

        result.to_csv()
        return result
