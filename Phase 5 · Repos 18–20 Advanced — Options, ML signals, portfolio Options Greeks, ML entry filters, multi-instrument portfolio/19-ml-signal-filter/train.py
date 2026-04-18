"""
train.py
─────────
End-to-end training pipeline for the XGBoost signal filter.

Steps:
  1. Fetch OHLCV data from Kite (or load from cache)
  2. Engineer features (60+ technical indicators)
  3. Generate labels (forward TP/SL simulation)
  4. Run walk-forward validation (N folds)
  5. Train final model on full dataset with Optuna HPO
  6. Generate all analysis charts
  7. Save model to data/models/xgb_signal_filter.joblib

Usage:
  python train.py                          # full pipeline
  python train.py --no-hpo                 # skip Optuna (faster)
  python train.py --direction SELL         # train for SELL signals
  python train.py --symbol "NSE:NIFTY 50"  # override symbol
  python train.py --no-fetch               # use cached data only

Runtime: ~10–30 min depending on OPTUNA_TRIALS and N_FOLDS
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train")

# ── Imports ───────────────────────────────────────────────────────────
from features.feature_engineer import FeatureEngineer
from features.label_generator  import LabelGenerator
from model.xgb_classifier      import SignalClassifier, tune_hyperparameters
from backtest.walk_forward     import WalkForwardValidator
from utils.visualize           import generate_all_charts


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost signal filter")
    parser.add_argument("--symbol",    default=os.getenv("SYMBOL",    "NSE:NIFTY 50"))
    parser.add_argument("--timeframe", default=os.getenv("TIMEFRAME", "15minute"))
    parser.add_argument("--days",      type=int, default=730, help="Days of history")
    parser.add_argument("--direction", default="BUY", choices=["BUY", "SELL"])
    parser.add_argument("--no-hpo",    action="store_true", help="Skip Optuna HPO")
    parser.add_argument("--no-wf",     action="store_true", help="Skip walk-forward")
    parser.add_argument("--no-fetch",  action="store_true", help="Use cached data")
    parser.add_argument("--model-out", default=os.getenv("MODEL_PATH", "data/models/xgb_signal_filter.joblib"))
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  ML Signal Filter — Training Pipeline")
    logger.info(f"  Symbol:    {args.symbol}")
    logger.info(f"  Timeframe: {args.timeframe}")
    logger.info(f"  Direction: {args.direction}")
    logger.info(f"  History:   {args.days} days")
    logger.info(f"  HPO:       {'OFF' if args.no_hpo else 'ON'}")
    logger.info(f"  WalkFwd:   {'OFF' if args.no_wf  else 'ON'}")
    logger.info("=" * 60)

    # ── 1. Fetch data ─────────────────────────────────────────────────
    logger.info("Step 1/6 — Loading OHLCV data…")
    from core.data_fetcher import DataFetcher
    fetcher = DataFetcher()
    ohlcv   = fetcher.fetch(
        symbol   = args.symbol,
        interval = args.timeframe,
        days     = args.days,
        use_cache= args.no_fetch is False,
    )
    logger.info(f"  Loaded {len(ohlcv)} rows | {ohlcv['datetime'].min()} → {ohlcv['datetime'].max()}")

    # ── 2. Feature engineering ────────────────────────────────────────
    logger.info("Step 2/6 — Engineering features…")
    fe       = FeatureEngineer()
    features = fe.transform(ohlcv, dropna=True)
    logger.info(f"  Feature matrix: {features.shape} ({features.shape[1]-2} features)")

    # ── 3. Label generation ───────────────────────────────────────────
    logger.info("Step 3/6 — Generating labels…")
    lg     = LabelGenerator()
    labels = lg.label_all_bars(ohlcv, direction=args.direction)
    bal    = lg.class_balance(labels["label"])
    logger.info(
        f"  Labels: {bal['total']} total | "
        f"{bal['positive']} positive ({bal['pos_pct']}%) | "
        f"{bal['negative']} negative ({bal['neg_pct']}%)"
    )

    if bal["positive"] < 20:
        logger.error("Too few positive labels for training. Adjust TP/SL thresholds in .env")
        sys.exit(1)

    # Merge features + labels
    merged = pd.merge(
        features, labels[["datetime", "label"]],
        on="datetime", how="inner"
    ).dropna(subset=["label"]).reset_index(drop=True)

    feature_cols = [c for c in merged.columns if c not in ("datetime", "close", "label")]
    X_full = merged[feature_cols]
    y_full = merged["label"].astype(int)

    # ── 4. Walk-forward validation ────────────────────────────────────
    wf_result = None
    if not args.no_wf:
        logger.info("Step 4/6 — Walk-forward validation…")
        validator = WalkForwardValidator(
            direction = args.direction,
            run_hpo   = not args.no_hpo,
        )
        wf_result = validator.run(ohlcv)
        logger.info("\n" + wf_result.summary_df().to_string(index=False))
        logger.info(
            f"\n  ► Mean Precision:     {wf_result.mean_precision:.3f}\n"
            f"  ► Mean Avg Precision: {wf_result.mean_avg_precision:.3f}\n"
            f"  ► Mean F1:            {wf_result.mean_f1:.3f}\n"
            f"  ► Mean Pass Rate:     {wf_result.mean_pass_rate:.1f}%"
        )
    else:
        logger.info("Step 4/6 — Walk-forward skipped.")

    # ── 5. Train final model ──────────────────────────────────────────
    logger.info("Step 5/6 — Training final model on full dataset…")

    best_params = None
    if not args.no_hpo:
        logger.info("  Running Optuna HPO on 15% validation split…")
        val_size  = max(int(len(X_full) * 0.15), 30)
        X_tr = X_full.iloc[:-val_size]
        y_tr = y_full.iloc[:-val_size]
        X_vl = X_full.iloc[-val_size:]
        y_vl = y_full.iloc[-val_size:]
        best_params = tune_hyperparameters(X_tr, y_tr, X_vl, y_vl)

    clf = SignalClassifier(params=best_params)
    train_metrics = clf.train(X_full, y_full, verbose=True)
    logger.info(f"  Train eval metrics: {train_metrics}")

    # Final held-out evaluation (last 10%)
    holdout_size = max(int(len(X_full) * 0.10), 20)
    X_test = X_full.iloc[-holdout_size:]
    y_test = y_full.iloc[-holdout_size:]
    test_metrics = clf.evaluate(X_test, y_test)
    logger.info(f"  Holdout test metrics: {test_metrics}")

    # Save model
    saved_path = clf.save(args.model_out)
    logger.info(f"  Model saved: {saved_path}")

    # ── 6. Generate charts ────────────────────────────────────────────
    logger.info("Step 6/6 — Generating analysis charts…")
    chart_paths = generate_all_charts(clf, X_test, y_test, wf_result)
    logger.info(f"  Charts saved: {chart_paths}")

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info(f"  Model:       {saved_path}")
    logger.info(f"  Features:    {len(feature_cols)}")
    logger.info(f"  Holdout P:   {test_metrics.get('precision', 'N/A')}")
    logger.info(f"  Holdout AP:  {test_metrics.get('avg_prec',  'N/A')}")
    logger.info(f"  Pass rate:   {test_metrics.get('pass_rate_pct', 'N/A')}%")
    logger.info("=" * 60)

    # Quick feature importance printout
    logger.info("\nTop 10 features by importance:")
    imp = clf.feature_importances(top_n=10)
    for _, row in imp.iterrows():
        logger.info(f"  {row['rank']:2d}. {row['feature']:<30s}  {row['importance']:.1f}")

    logger.info(f"\n  To start the filter API: python api/signal_server.py")
    logger.info(f"  Health check:            curl http://localhost:{os.getenv('API_PORT', '5001')}/health")


if __name__ == "__main__":
    main()
