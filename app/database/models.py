"""
SQLAlchemy ORM models for CreditRiskML persistence.
Includes prediction logging, audit trails, and drift logs.
"""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, Boolean
from app.database.session import Base

def utcnow():
    return datetime.now(timezone.utc)

class PredictionLog(Base):
    """Stores inference inputs hash, model output, latency, and ground-truth matches."""
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime, default=utcnow, index=True, nullable=False)
    model_version = Column(String(32), nullable=False)
    prediction = Column(Integer, nullable=False)
    default_probability = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False)
    latency_ms = Column(Float, nullable=False)
    input_hash = Column(String(64), nullable=False)
    actual_label = Column(Integer, nullable=True, index=True)  # Populated when ground-truth is known

    def __repr__(self):
        return f"<PredictionLog(id={self.id}, req={self.request_id}, prob={self.default_probability:.2f}, risk={self.risk_level})>"

class DriftReportLog(Base):
    """Stores executed drift monitoring reports."""
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=utcnow, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    drift_share = Column(Float, nullable=False)
    total_features = Column(Integer, nullable=False)
    drifted_features_count = Column(Integer, nullable=False)
    report_json = Column(Text, nullable=False)
