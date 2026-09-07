"""
Model evaluation and decision threshold optimization module for CreditRiskML.
Calculates classification metrics, generates plots, and finds optimal decision cutoffs.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from app.config import settings

logger = logging.getLogger("credit_risk_ml.evaluate")

def compute_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes a comprehensive set of credit risk classification metrics.
    """
    y_pred = (y_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except Exception:
        pr_auc = 0.0

    brier = float(brier_score_loss(y_true, y_probs))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "threshold": round(threshold, 4),
        "confusion_matrix": cm
    }

def optimize_decision_threshold(
    y_val: np.ndarray,
    y_val_probs: np.ndarray,
    min_recall: float = 0.50
) -> Tuple[float, Dict[str, Any]]:
    """
    Scans decision thresholds from 0.10 to 0.90 to maximize F1-score
    subject to meeting the regulatory/risk minimum recall constraint.
    """
    best_threshold = 0.50
    best_f1 = -1.0
    best_metrics = {}

    thresholds = np.linspace(0.10, 0.90, 81)
    for t in thresholds:
        metrics = compute_metrics(y_val, y_val_probs, threshold=float(t))
        if metrics["recall"] >= min_recall and metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(t)
            best_metrics = metrics

    # Fallback to default 0.50 if constraint wasn't met
    if not best_metrics:
        best_threshold = 0.50
        best_metrics = compute_metrics(y_val, y_val_probs, threshold=0.50)

    logger.info(f"Optimal decision threshold selected: {best_threshold:.3f} (Val F1: {best_metrics['f1']:.3f}, Recall: {best_metrics['recall']:.3f})")
    return best_threshold, best_metrics

def generate_evaluation_reports(
    y_test: np.ndarray,
    y_test_probs: np.ndarray,
    threshold: float,
    reports_dir: Path = Path("reports")
) -> Dict[str, Any]:
    """
    Calculates final test metrics, writes metrics.json, and generates ROC, PR, and Confusion Matrix plots.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(y_test, y_test_probs, threshold=threshold)

    # 1. Save metrics.json
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved test metrics report to {metrics_path}")

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_test_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", lw=1.5, label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC) — CreditRiskML")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = reports_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # 3. Precision-Recall Curve Plot
    precisions, recalls, _ = precision_recall_curve(y_test, y_test_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(recalls, precisions, color="#10b981", lw=2, label=f"PR Curve (PR-AUC = {metrics['pr_auc']:.3f})")
    plt.axvline(x=metrics["recall"], color="#ef4444", linestyle=":", label=f"Selected Cutoff Recall = {metrics['recall']:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — CreditRiskML")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = reports_dir / "precision_recall_curve.png"
    plt.savefig(pr_path, dpi=150)
    plt.close()

    # 4. Confusion Matrix Plot
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Threshold = {threshold:.2f})")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Good (0)", "Default (1)"])
    plt.yticks(tick_marks, ["Good (0)", "Default (1)"])

    thresh_val = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh_val else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    return metrics
