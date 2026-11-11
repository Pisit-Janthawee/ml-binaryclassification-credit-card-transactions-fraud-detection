"""Metrics and plots for imbalanced binary classification.

Accuracy is useless here (predicting "never fraud" scores ~99.5%), so we report
PR-AUC, ROC-AUC and precision / recall / F1 at a chosen threshold, plus a
money-based cost.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from .config import COLORS


def score(y_true, proba, threshold: float = 0.5, name: str = "model") -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "model": name,
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "threshold": threshold,
    }


def recall_at_precision(y_true, proba, min_precision: float = 0.9) -> float:
    p, r, _ = precision_recall_curve(y_true, proba)
    ok = p >= min_precision
    return float(r[ok].max()) if ok.any() else 0.0


def fraud_cost(y_true, proba, amount, threshold, review_cost: float = 10.0) -> float:
    """Business cost in $: every alert costs `review_cost` to investigate,
    every missed fraud costs its transaction amount."""
    pred = proba >= threshold
    y_true = np.asarray(y_true).astype(bool)
    missed = (~pred & y_true)
    return float(pred.sum() * review_cost + np.asarray(amount)[missed].sum())


def best_cost_threshold(y_true, proba, amount, review_cost: float = 10.0, grid=None):
    grid = np.linspace(0.01, 0.99, 99) if grid is None else grid
    costs = np.array([fraud_cost(y_true, proba, amount, t, review_cost) for t in grid])
    return float(grid[costs.argmin()]), pd.DataFrame({"threshold": grid, "cost": costs})


def best_f1_threshold(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.nanargmax(f1[:-1])])


def plot_pr_roc(results: dict, y_true, ax=None):
    """results: {name: proba}. Draws PR and ROC curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5)) if ax is None else (None, ax)
    base = np.mean(y_true)
    for name, proba in results.items():
        p, r, _ = precision_recall_curve(y_true, proba)
        axes[0].plot(r, p, label=f"{name} (AP={average_precision_score(y_true, proba):.3f})")
        fpr, tpr, _ = roc_curve(y_true, proba)
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, proba):.3f})")
    axes[0].axhline(base, ls="--", c="grey", lw=1, label=f"random ({base:.4f})")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    axes[1].plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    for a in axes:
        a.legend(fontsize=9, loc="best")
    return fig


def plot_confusion(y_true, proba, threshold, ax=None, title=None):
    cm = confusion_matrix(y_true, proba >= threshold)
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    ax.imshow(cm, cmap="Greys")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else COLORS[1], fontsize=13, weight="bold")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
           xlabel="Predicted", ylabel="Actual", title=title or f"Confusion matrix @ {threshold:.2f}")
    return ax
