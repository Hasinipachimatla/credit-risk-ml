"""
Database repository operations for prediction logs and drift audits.
"""
import logging
import hashlib
import json
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.database.models import PredictionLog, DriftReportLog

logger = logging.getLogger("credit_risk_ml.repository")

def compute_input_hash(data: Dict[str, Any]) -> str:
    """Creates a deterministic SHA-256 hash of application features to avoid raw PII storage."""
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def create_prediction_log(
    db: Session,
    request_id: str,
    model_version: str,
    prediction: int,
    default_probability: float,
    risk_level: str,
    latency_ms: float,
    input_data: Dict[str, Any],
    actual_label: Optional[int] = None
) -> PredictionLog:
    """Inserts a single prediction record into the database."""
    input_hash = compute_input_hash(input_data)
    log_entry = PredictionLog(
        request_id=request_id,
        model_version=model_version,
        prediction=prediction,
        default_probability=default_probability,
        risk_level=risk_level,
        latency_ms=latency_ms,
        input_hash=input_hash,
        actual_label=actual_label
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry

def get_recent_prediction_logs(db: Session, limit: int = 1000) -> List[PredictionLog]:
    """Retrieves recent prediction logs ordered by timestamp descending."""
    return db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(limit).all()

def get_labeled_predictions(db: Session) -> List[PredictionLog]:
    """Retrieves prediction logs with non-null actual ground-truth labels."""
    return db.query(PredictionLog).filter(PredictionLog.actual_label.isnot(None)).all()

def update_ground_truth(db: Session, request_id: str, actual_label: int) -> Optional[PredictionLog]:
    """Updates ground truth label for an existing prediction log."""
    log_entry = db.query(PredictionLog).filter(PredictionLog.request_id == request_id).first()
    if log_entry:
        log_entry.actual_label = actual_label
        db.commit()
        db.refresh(log_entry)
    return log_entry

def save_drift_report(db: Session, drift_result: Dict[str, Any]) -> DriftReportLog:
    """Persists a drift calculation report."""
    report_entry = DriftReportLog(
        drift_detected=drift_result.get("dataset_drift_detected", False),
        drift_share=drift_result.get("drift_share", 0.0),
        total_features=drift_result.get("total_features", 0),
        drifted_features_count=drift_result.get("drifted_features_count", 0),
        report_json=json.dumps(drift_result)
    )
    db.add(report_entry)
    db.commit()
    db.refresh(report_entry)
    return report_entry
