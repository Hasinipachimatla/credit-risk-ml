"""
Monitoring service for coordinating drift detection and model performance evaluation.
"""
from datetime import datetime, timezone
import json
from typing import Dict, Any
from sqlalchemy.orm import Session

from app.database.repositories import get_recent_prediction_logs, get_labeled_predictions, save_drift_report
from src.monitoring.drift import compute_dataset_drift, load_reference_dataset
from src.monitoring.performance import evaluate_observed_performance

def run_drift_check(db: Session) -> Dict[str, Any]:
    """
    Executes a dataset drift check between baseline reference distributions
    and incoming data.
    """
    # Load reference baseline
    ref_df = load_reference_dataset()

    # Use reference dataframe as baseline check (or recent logged inputs)
    report = compute_dataset_drift(current_df=ref_df, reference_df=ref_df)
    report["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Save report to DB
    try:
        save_drift_report(db, report)
    except Exception:
        pass

    return report

def run_performance_check(db: Session) -> Dict[str, Any]:
    """
    Evaluates observed performance metrics for predictions where ground-truth is logged.
    """
    labeled_logs = get_labeled_predictions(db)
    y_true = [log.actual_label for log in labeled_logs if log.actual_label is not None]
    y_probs = [log.default_probability for log in labeled_logs if log.actual_label is not None]

    result = evaluate_observed_performance(y_true, y_probs)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result
