"""
Pydantic schemas for model metadata, monitoring reports, and retraining results.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class FeatureDriftStat(BaseModel):
    feature: str
    type: str
    psi: float
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    status: str
    drift_detected: bool

class DriftReportResponse(BaseModel):
    timestamp: str
    dataset_drift_detected: bool
    drift_share: float
    total_features: int
    drifted_features_count: int
    psi_threshold: float
    ks_pvalue_threshold: float
    features: List[FeatureDriftStat]

class PerformanceReportResponse(BaseModel):
    timestamp: str
    status: str
    message: Optional[str] = None
    samples_count: int
    threshold_used: Optional[float] = None
    current_metrics: Optional[Dict[str, Any]] = None
    baseline_metrics: Optional[Dict[str, Any]] = None
    metric_differences: Optional[Dict[str, float]] = None
    degradation_detected: bool
    degradation_reasons: List[str] = Field(default_factory=list)

class ModelMetadataResponse(BaseModel):
    model_name: str
    model_version: str
    status: str
    decision_threshold: float
    run_id: Optional[str] = None
    validation_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]

class RetrainResponse(BaseModel):
    status: str
    message: str
    champion_version: Optional[str] = None
    challenger_metrics: Optional[Dict[str, Any]] = None
    previous_champion_metrics: Optional[Dict[str, Any]] = None
    gate_reasons: List[str]

class HealthResponse(BaseModel):
    status: str
    app_env: str
    model_loaded: bool
    model_version: str
    database_connected: bool
    timestamp: str
