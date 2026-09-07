"""
Performance monitoring module for CreditRiskML.
Evaluates observed model performance against baseline metrics using ground-truth default labels.
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.models.evaluate import compute_metrics
from src.models.mlflow_utils import load_model_registry

logger = logging.getLogger("credit_risk_ml.performance")

MIN_SAMPLES_FOR_EVALUATION = 20

def evaluate_observed_performance(
    y_true: List[int],
    y_probs: List[float],
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluates observed performance metrics and compares against active Champion baseline.
    Returns 'insufficient ground truth' if too few labels are available.
    """
    if len(y_true) < MIN_SAMPLES_FOR_EVALUATION or len(y_probs) < MIN_SAMPLES_FOR_EVALUATION:
        return {
            "status": "insufficient ground truth",
            "message": f"Only {len(y_true)} labeled records available. Minimum required is {MIN_SAMPLES_FOR_EVALUATION}.",
            "samples_count": len(y_true),
            "current_metrics": None,
            "baseline_metrics": None,
            "degradation_detected": False,
            "degradation_reasons": []
        }

    registry = load_model_registry()
    champ = registry.get("champion")
    baseline_threshold = champ.get("threshold", 0.50) if champ else 0.50
    eval_threshold = threshold if threshold is not None else baseline_threshold

    current_metrics = compute_metrics(
        y_true=np.array(y_true),
        y_probs=np.array(y_probs),
        threshold=eval_threshold
    )

    baseline_metrics = champ.get("test_metrics") if champ else None
    deltas = {}
    degradation_detected = False
    degradation_reasons = []

    if baseline_metrics:
        for metric_name in ["roc_auc", "f1", "recall", "precision"]:
            cur_val = current_metrics.get(metric_name, 0.0)
            base_val = baseline_metrics.get(metric_name, 0.0)
            diff = round(cur_val - base_val, 4)
            deltas[f"{metric_name}_diff"] = diff

            # Check degradation bounds
            if metric_name == "roc_auc" and diff < -0.05:
                degradation_detected = True
                degradation_reasons.append(f"ROC-AUC dropped by {abs(diff):.4f} (exceeds -0.05 threshold)")
            elif metric_name == "recall" and diff < -0.10:
                degradation_detected = True
                degradation_reasons.append(f"Recall dropped by {abs(diff):.4f} (exceeds -0.10 threshold)")

    return {
        "status": "evaluated",
        "samples_count": len(y_true),
        "threshold_used": eval_threshold,
        "current_metrics": current_metrics,
        "baseline_metrics": baseline_metrics,
        "metric_differences": deltas,
        "degradation_detected": degradation_detected,
        "degradation_reasons": degradation_reasons
    }
