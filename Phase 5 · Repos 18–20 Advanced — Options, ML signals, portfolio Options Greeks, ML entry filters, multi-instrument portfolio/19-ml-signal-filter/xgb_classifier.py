"""
model/xgb_classifier.py
────────────────────────
XGBoost-based binary classifier that learns to filter trading signals.

Architecture:
  - Input:  feature vector (60+ technical indicators)
  - Output: P(good trade) ∈ [0, 1]
  - Threshold: configurable MIN_PROBA (default 0.60)
  - Pass signal → True if predicted_proba >= MIN_PROBA

Training pipeline:
  1. Feature engineering on raw OHLCV
  2. Label generation (forward TP/SL simulation)
  3. Optional Optuna HPO (50 trials by default)
  4. Final model trained on full train set
  5. Saved to joblib

Key design decisions:
  - scale_pos_weight handles class imbalance automatically
  - early_stopping_rounds prevents overfitting
  - eval_metric = 'aucpr' (area under precision-recall) — better than AUC-ROC
    for imbalanced datasets where we care more about precision of BUY signals
  - Model is retrained in each walk-forward fold (no leakage)
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

MODEL_DIR  = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_PROBA     = float(os.getenv("MIN_PROBA",      "0.60"))
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS",    "50"))


# ── Default XGBoost parameters (tuned for signal classification) ──────
DEFAULT_PARAMS = {
    "n_estimators":     500,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "objective":        "binary:logistic",
    "eval_metric":      "aucpr",
    "tree_method":      "hist",
    "random_state":     42,
    "n_jobs":           -1,
}


class SignalClassifier:
    """
    XGBoost signal filter.

    Usage:
        clf = SignalClassifier()
        clf.train(X_train, y_train)
        proba    = clf.predict_proba(X_test)
        decision = clf.filter_signal(features_row)   # True/False
    """

    def __init__(self, params: dict = None, min_proba: float = MIN_PROBA):
        self.params    = params or DEFAULT_PARAMS.copy()
        self.min_proba = min_proba
        self.model:    Optional[xgb.XGBClassifier] = None
        self.scaler:   Optional[StandardScaler]     = None
        self.feature_names_: list[str]              = []

    # ──────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_fraction: float = 0.15,
        verbose: bool = True,
    ) -> dict:
        """
        Train the classifier.

        Args:
            X            : feature DataFrame (no datetime/close columns)
            y            : binary labels (0/1)
            eval_fraction: fraction of training data used as eval set
                           for early stopping (not the test fold)
        Returns:
            dict of training metrics
        """
        self.feature_names_ = list(X.columns)
        X_arr = X.values.astype(np.float32)

        # Scale features (XGBoost is tree-based so not strictly necessary,
        # but helps when features have very different ranges)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_arr)

        # Class imbalance: scale_pos_weight = neg_count / pos_count
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos == 0:
            raise ValueError("No positive labels in training data.")
        scale_pos_weight = n_neg / n_pos
        logger.info(
            f"Training: {len(y)} samples | "
            f"Positive: {n_pos} ({n_pos/len(y)*100:.1f}%) | "
            f"scale_pos_weight: {scale_pos_weight:.2f}"
        )

        # Internal eval split for early stopping
        X_tr, X_ev, y_tr, y_ev = train_test_split(
            X_scaled, y.values, test_size=eval_fraction,
            stratify=y.values, random_state=42
        )

        params = {**self.params, "scale_pos_weight": scale_pos_weight}
        self.model = xgb.XGBClassifier(
            **params,
            early_stopping_rounds=30,
            verbosity=0,
        )

        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_ev, y_ev)],
            verbose=100 if verbose else False,
        )

        # Metrics on eval split
        proba = self.model.predict_proba(X_ev)[:, 1]
        preds = (proba >= self.min_proba).astype(int)

        metrics = self._compute_metrics(y_ev, proba, preds, split="train_eval")
        if verbose:
            logger.info(f"Train eval metrics: {metrics}")
        return metrics

    # ──────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of good trade for each row."""
        self._require_trained()
        X_scaled = self.scaler.transform(X.values.astype(np.float32))
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (0/1) using min_proba threshold."""
        proba = self.predict_proba(X)
        return (proba >= self.min_proba).astype(int)

    def filter_signal(self, features_row: pd.Series) -> tuple[bool, float]:
        """
        Filter a single incoming signal.

        Args:
            features_row: Series of feature values for the current bar
        Returns:
            (pass_signal: bool, confidence: float)
        """
        self._require_trained()
        X = pd.DataFrame([features_row])
        # Align to training feature names
        for col in self.feature_names_:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names_]
        proba = float(self.predict_proba(X)[0])
        return proba >= self.min_proba, round(proba, 4)

    # ──────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Full evaluation on a held-out set."""
        proba = self.predict_proba(X)
        preds = (proba >= self.min_proba).astype(int)
        return self._compute_metrics(y.values, proba, preds, split="test")

    @staticmethod
    def _compute_metrics(y_true, proba, preds, split: str = "") -> dict:
        metrics = {
            "split":      split,
            "n_samples":  len(y_true),
            "n_signals_passed": int(preds.sum()),
            "pass_rate_pct":    round(preds.mean() * 100, 1),
        }
        if len(np.unique(y_true)) > 1:
            metrics["precision"] = round(precision_score(y_true, preds, zero_division=0), 4)
            metrics["recall"]    = round(recall_score(y_true,    preds, zero_division=0), 4)
            metrics["f1"]        = round(f1_score(y_true,        preds, zero_division=0), 4)
            metrics["roc_auc"]   = round(roc_auc_score(y_true,   proba),                 4)
            metrics["avg_prec"]  = round(average_precision_score(y_true, proba),          4)
        return metrics

    # ──────────────────────────────────────────────────────────────────
    # Feature importance
    # ──────────────────────────────────────────────────────────────────

    def feature_importances(self, top_n: int = 20) -> pd.DataFrame:
        """Return top-N features by gain importance."""
        self._require_trained()
        imp = self.model.get_booster().get_score(importance_type="gain")
        df  = pd.DataFrame.from_dict(imp, orient="index", columns=["importance"])
        df  = df.sort_values("importance", ascending=False).head(top_n)
        df["rank"] = range(1, len(df) + 1)
        return df.reset_index().rename(columns={"index": "feature"})

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values (requires shap library)."""
        self._require_trained()
        try:
            import shap
            X_scaled = self.scaler.transform(X.values.astype(np.float32))
            explainer = shap.TreeExplainer(self.model)
            return explainer.shap_values(X_scaled)
        except ImportError:
            raise ImportError("Install shap: pip install shap")

    # ──────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────

    def save(self, path: str = None) -> str:
        """Save model + scaler + feature names to a joblib file."""
        self._require_trained()
        path = path or str(MODEL_DIR / "xgb_signal_filter.joblib")
        bundle = {
            "model":         self.model,
            "scaler":        self.scaler,
            "feature_names": self.feature_names_,
            "min_proba":     self.min_proba,
            "params":        self.params,
        }
        joblib.dump(bundle, path)
        logger.info(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: str = None) -> "SignalClassifier":
        """Load a saved model bundle."""
        path   = path or str(MODEL_DIR / "xgb_signal_filter.joblib")
        bundle = joblib.load(path)
        inst   = cls(params=bundle["params"], min_proba=bundle["min_proba"])
        inst.model          = bundle["model"]
        inst.scaler         = bundle["scaler"]
        inst.feature_names_ = bundle["feature_names"]
        logger.info(f"Model loaded from {path} | Features: {len(inst.feature_names_)}")
        return inst

    def _require_trained(self):
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")


# ──────────────────────────────────────────────────────────────────────
# Optuna hyperparameter optimisation
# ──────────────────────────────────────────────────────────────────────

def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    n_trials: int = OPTUNA_TRIALS,
) -> dict:
    """
    Run Optuna hyperparameter search. Returns best params dict.

    Optimises for average_precision_score (AP) on the validation set —
    better than AUC-ROC for imbalanced signal classification.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed — using default params.")
        return DEFAULT_PARAMS.copy()

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw   = n_neg / max(n_pos, 1)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",     100, 800),
            "max_depth":        trial.suggest_int("max_depth",        3, 7),
            "learning_rate":    trial.suggest_float("learning_rate",  0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample",      0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma",          0.0, 0.5),
            "reg_alpha":        trial.suggest_float("reg_alpha",      0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda",     0.5, 5.0),
            "scale_pos_weight": spw,
            "objective":        "binary:logistic",
            "eval_metric":      "aucpr",
            "tree_method":      "hist",
            "random_state":     42,
            "n_jobs":           -1,
            "verbosity":        0,
        }
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_train.values.astype(np.float32))
        X_vl_sc  = scaler.transform(X_val.values.astype(np.float32))

        model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
        model.fit(X_tr_sc, y_train.values,
                  eval_set=[(X_vl_sc, y_val.values)],
                  verbose=False)

        proba = model.predict_proba(X_vl_sc)[:, 1]
        return average_precision_score(y_val.values, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "objective":    "binary:logistic",
        "eval_metric":  "aucpr",
        "tree_method":  "hist",
        "random_state": 42,
        "n_jobs":       -1,
    })

    logger.info(
        f"Optuna: best AP = {study.best_value:.4f} | "
        f"best params: {best}"
    )
    return best
