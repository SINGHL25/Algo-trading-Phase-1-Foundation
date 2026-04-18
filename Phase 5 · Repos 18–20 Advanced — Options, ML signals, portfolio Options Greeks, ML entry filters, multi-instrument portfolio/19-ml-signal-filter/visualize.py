"""
utils/visualize.py
───────────────────
Generates all plots for model analysis:

  1. feature_importance_chart()  — XGBoost gain importance bar chart
  2. shap_summary_plot()         — SHAP beeswarm (requires shap library)
  3. walk_forward_chart()        — Per-fold precision/recall/AP line chart
  4. calibration_curve()         — Predicted probability vs actual hit rate
  5. confusion_matrix_chart()    — At the configured threshold
  6. precision_recall_curve()    — PR curve with AP annotation

All charts are saved to data/models/ as high-resolution PNGs.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

matplotlib.use("Agg")   # non-interactive backend (safe for server)

CHART_DIR = Path("data/models")
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
_COLORS = {
    "primary":  "#2563EB",
    "success":  "#16A34A",
    "danger":   "#DC2626",
    "warning":  "#D97706",
    "muted":    "#6B7280",
    "bg":       "#F9FAFB",
    "grid":     "#E5E7EB",
}

def _setup_style():
    plt.rcParams.update({
        "figure.facecolor":  _COLORS["bg"],
        "axes.facecolor":    "#FFFFFF",
        "axes.edgecolor":    _COLORS["grid"],
        "axes.grid":         True,
        "grid.color":        _COLORS["grid"],
        "grid.linewidth":    0.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "600",
        "axes.labelsize":    11,
    })


# ──────────────────────────────────────────────────────────────────────
# 1. Feature Importance
# ──────────────────────────────────────────────────────────────────────

def feature_importance_chart(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: str = "data/models/feature_importance.png",
) -> str:
    """
    Horizontal bar chart of top-N features by XGBoost gain importance.

    Args:
        importance_df: DataFrame with columns [feature, importance]
                       (from SignalClassifier.feature_importances())
    """
    _setup_style()
    top = importance_df.head(top_n).copy()
    top = top.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    fig.patch.set_facecolor(_COLORS["bg"])

    bars = ax.barh(
        top["feature"], top["importance"],
        color=_COLORS["primary"], alpha=0.85, edgecolor="white", linewidth=0.5
    )

    # Value labels on bars
    for bar, val in zip(bars, top["importance"]):
        ax.text(
            bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=9, color=_COLORS["muted"]
        )

    ax.set_xlabel("Gain Importance (higher = more splits use this feature)")
    ax.set_title(f"Top {top_n} Feature Importances — XGBoost Gain")
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance chart saved: {save_path}")
    return save_path


# ──────────────────────────────────────────────────────────────────────
# 2. SHAP Summary
# ──────────────────────────────────────────────────────────────────────

def shap_summary_plot(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    save_path: str = "data/models/shap_summary.png",
) -> str:
    """Beeswarm SHAP plot showing feature impact on model output."""
    try:
        import shap
        _setup_style()
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_vals, X,
            plot_type="dot",
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"SHAP summary saved: {save_path}")
        return save_path
    except Exception as e:
        logger.warning(f"SHAP plot failed: {e}")
        return ""


# ──────────────────────────────────────────────────────────────────────
# 3. Walk-Forward Results
# ──────────────────────────────────────────────────────────────────────

def walk_forward_chart(
    wf_result,       # WalkForwardResult
    save_path: str = "data/models/walk_forward_metrics.png",
) -> str:
    """
    Multi-panel chart: per-fold precision, recall, AP, pass-rate.
    """
    _setup_style()
    folds   = wf_result.folds
    fold_ns = [f.fold_n for f in folds]

    metrics = {
        "Precision":     [f.precision     for f in folds],
        "Recall":        [f.recall        for f in folds],
        "Avg Precision": [f.avg_precision for f in folds],
        "Pass Rate %":   [f.pass_rate_pct for f in folds],
    }
    colors = [_COLORS["primary"], _COLORS["success"], _COLORS["warning"], _COLORS["muted"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Walk-Forward Validation — {wf_result.direction} signals",
        fontsize=14, fontweight="600", y=1.01
    )

    for ax, (metric, vals), color in zip(axes.flat, metrics.items(), colors):
        ax.plot(fold_ns, vals, marker="o", color=color, linewidth=2, markersize=6)
        ax.axhline(np.mean(vals), color=color, linestyle="--",
                   alpha=0.6, linewidth=1.2, label=f"Mean: {np.mean(vals):.3f}")
        ax.fill_between(fold_ns, vals, np.mean(vals), alpha=0.08, color=color)
        ax.set_title(metric)
        ax.set_xlabel("Fold")
        ax.set_xticks(fold_ns)
        ax.set_ylim(0, max(max(vals) * 1.15, 0.1))
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Walk-forward chart saved: {save_path}")
    return save_path


# ──────────────────────────────────────────────────────────────────────
# 4. Precision-Recall Curve
# ──────────────────────────────────────────────────────────────────────

def precision_recall_curve_chart(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.60,
    save_path: str = "data/models/pr_curve.png",
) -> str:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    _setup_style()

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color=_COLORS["primary"], linewidth=2.0, label=f"PR curve (AP={ap:.3f})")
    ax.fill_between(recall, precision, alpha=0.08, color=_COLORS["primary"])

    # Mark current threshold
    thresh_idx = np.searchsorted(thresholds, threshold)
    if thresh_idx < len(precision) - 1:
        ax.scatter(
            recall[thresh_idx], precision[thresh_idx],
            color=_COLORS["danger"], zorder=5, s=80,
            label=f"Threshold={threshold} → P={precision[thresh_idx]:.3f} R={recall[thresh_idx]:.3f}",
        )

    # Baseline (random classifier)
    pos_rate = y_true.mean()
    ax.axhline(pos_rate, color=_COLORS["muted"], linestyle="--",
               linewidth=1, label=f"Baseline (random) = {pos_rate:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve saved: {save_path}")
    return save_path


# ──────────────────────────────────────────────────────────────────────
# 5. Calibration Curve
# ──────────────────────────────────────────────────────────────────────

def calibration_chart(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    save_path: str = "data/models/calibration.png",
) -> str:
    """
    Shows if predicted probabilities are well-calibrated.
    A well-calibrated model: predicted 60% → ~60% of signals are actually good.
    """
    from sklearn.calibration import calibration_curve
    _setup_style()

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated", alpha=0.5)
    ax.plot(prob_pred, prob_true, marker="o", color=_COLORS["primary"],
            linewidth=2, markersize=6, label="XGBoost")

    ax.fill_between(prob_pred, prob_true, prob_pred,
                    alpha=0.08, color=_COLORS["warning"])

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual Hit Rate)")
    ax.set_title("Calibration Curve — Signal Quality Reliability")
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Calibration chart saved: {save_path}")
    return save_path


# ──────────────────────────────────────────────────────────────────────
# 6. All-in-one report
# ──────────────────────────────────────────────────────────────────────

def generate_all_charts(clf, X_test, y_test, wf_result=None) -> list[str]:
    """Convenience: generate all charts for a trained classifier."""
    paths = []

    imp = clf.feature_importances(top_n=20)
    paths.append(feature_importance_chart(imp))

    proba = clf.predict_proba(X_test)
    paths.append(precision_recall_curve_chart(y_test.values, proba))
    paths.append(calibration_chart(y_test.values, proba))

    if wf_result is not None:
        paths.append(walk_forward_chart(wf_result))

    try:
        shap_vals = clf.shap_values(X_test.head(500))
        paths.append(shap_summary_plot(shap_vals, X_test.head(500)))
    except Exception as e:
        logger.warning(f"SHAP chart skipped: {e}")

    return [p for p in paths if p]
