"""
Monitoring, runtime metrics, and active model metadata routes.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.services.model_service import model_service
from app.services.metrics_service import metrics_tracker
from app.services.monitoring_service import run_drift_check, run_performance_check
from app.api.schemas.monitoring import (
    ModelMetadataResponse,
    DriftReportResponse,
    PerformanceReportResponse
)

router = APIRouter()

@router.get("/model", response_model=ModelMetadataResponse)
async def get_active_model_metadata():
    """
    Returns active Champion model metadata, parameters, and evaluation metrics.
    """
    metadata = model_service.get_metadata()
    if not metadata:
        raise HTTPException(status_code=404, detail="No active champion model registered yet.")

    return ModelMetadataResponse(
        model_name=metadata["model_name"],
        model_version=metadata["model_version"],
        status=metadata["status"],
        decision_threshold=metadata["threshold"],
        run_id=metadata.get("run_id"),
        validation_metrics=metadata["val_metrics"],
        test_metrics=metadata["test_metrics"]
    )

@router.get("/metrics")
async def get_operational_metrics():
    """
    Returns API runtime operational metrics including request counts and latency percentiles.
    """
    return metrics_tracker.get_metrics()

@router.get("/monitoring/drift", response_model=DriftReportResponse)
async def get_drift_report(db: Session = Depends(get_db)):
    """
    Runs Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests
    comparing feature distributions against the baseline reference dataset.
    """
    try:
        report = run_drift_check(db)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift calculation error: {str(e)}")

@router.get("/monitoring/performance", response_model=PerformanceReportResponse)
async def get_performance_report(db: Session = Depends(get_db)):
    """
    Retrieves model performance tracking against ground truth labels.
    """
    try:
        report = run_performance_check(db)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance evaluation error: {str(e)}")
