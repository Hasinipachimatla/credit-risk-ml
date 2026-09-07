"""
Inference and prediction service for CreditRiskML.
Executes preprocessing, model inference, risk classification, feature explanations,
and asynchronous database logging.
"""
import time
import uuid
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.config import settings
from app.services.model_service import model_service
from app.database.session import SessionLocal
from app.database.repositories import create_prediction_log

logger = logging.getLogger("credit_risk_ml.prediction_service")

def classify_risk_level(prob: float) -> str:
    """Classifies risk level into Low, Medium, High based on default probability."""
    if prob < 0.25:
        return "Low"
    elif prob < 0.55:
        return "Medium"
    else:
        return "High"

def compute_linear_explanation(
    feature_names: List[str],
    scaled_values: np.ndarray,
    coefficients: np.ndarray,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Computes top feature contributions for linear models (coef * value).
    Positive contributions increase default risk; negative contributions decrease risk.
    """
    contributions = scaled_values * coefficients
    top_indices = np.argsort(np.abs(contributions))[::-1][:top_k]

    explanations = []
    for idx in top_indices:
        val = float(contributions[idx])
        explanations.append({
            "feature": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
            "contribution": round(val, 4),
            "direction": "increases_risk" if val > 0 else "decreases_risk"
        })
    return explanations

def async_log_prediction(
    request_id: str,
    model_version: str,
    prediction: int,
    default_probability: float,
    risk_level: str,
    latency_ms: float,
    input_dict: Dict[str, Any]
):
    """Background task to asynchronously persist prediction to the database."""
    try:
        db: Session = SessionLocal()
        try:
            create_prediction_log(
                db=db,
                request_id=request_id,
                model_version=model_version,
                prediction=prediction,
                default_probability=default_probability,
                risk_level=risk_level,
                latency_ms=latency_ms,
                input_data=input_dict
            )
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to log prediction {request_id} to database: {e}")

class PredictionService:
    """Core serving engine coordinating preprocessing, inference, explanations, and logging."""

    @staticmethod
    def predict_single(
        input_data: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], float]:
        """Runs end-to-end inference on a single applicant record."""
        start_time = time.perf_counter()
        req_id = request_id or str(uuid.uuid4())

        if not model_service.is_loaded():
            loaded = model_service.load_active_model()
            if not loaded:
                raise RuntimeError("No active credit risk model is currently loaded in memory.")

        model = model_service.model
        preprocessor = model_service.preprocessor
        threshold = model_service.threshold
        version = model_service.model_version

        # 1. Transform raw input dictionary into single-row dataframe
        input_df = pd.DataFrame([input_data])

        # 2. Transform through fitted preprocessing pipeline
        X_trans = preprocessor.transform(input_df)

        # 3. Predict default probability
        prob = float(model.predict_proba(X_trans)[0, 1])
        pred = int(prob >= threshold)
        risk_level = classify_risk_level(prob)

        # 4. Generate explanation
        explanations = []
        try:
            if hasattr(model, "coef_"):
                feature_names = preprocessor.named_steps["column_preprocessor"].get_feature_names_out()
                explanations = compute_linear_explanation(
                    feature_names=list(feature_names),
                    scaled_values=X_trans[0],
                    coefficients=model.coef_[0]
                )
            elif hasattr(model, "feature_importances_"):
                feature_names = preprocessor.named_steps["column_preprocessor"].get_feature_names_out()
                top_indices = np.argsort(model.feature_importances_)[::-1][:3]
                for idx in top_indices:
                    explanations.append({
                        "feature": str(feature_names[idx]),
                        "contribution": round(float(model.feature_importances_[idx]), 4),
                        "direction": "increases_risk"
                    })
        except Exception as err:
            logger.warning(f"Could not calculate feature explanations: {err}")

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        result = {
            "request_id": req_id,
            "prediction": pred,
            "default_probability": round(prob, 4),
            "risk_level": risk_level,
            "decision_threshold": round(threshold, 4),
            "model_version": version,
            "latency_ms": latency_ms,
            "top_risk_factors": explanations
        }

        return result, latency_ms

    @staticmethod
    def predict_batch(
        applications: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Runs vectorized batch inference across multiple applicant records."""
        start_time = time.perf_counter()

        if not model_service.is_loaded():
            model_service.load_active_model()

        model = model_service.model
        preprocessor = model_service.preprocessor
        threshold = model_service.threshold
        version = model_service.model_version

        input_df = pd.DataFrame(applications)
        X_trans = preprocessor.transform(input_df)
        probs = model.predict_proba(X_trans)[:, 1]

        results = []
        for i, prob in enumerate(probs):
            prob_val = float(prob)
            req_id = str(uuid.uuid4())
            results.append({
                "request_id": req_id,
                "prediction": int(prob_val >= threshold),
                "default_probability": round(prob_val, 4),
                "risk_level": classify_risk_level(prob_val),
                "decision_threshold": round(threshold, 4),
                "model_version": version,
                "latency_ms": 0.0,
                "top_risk_factors": []
            })

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return results, total_latency_ms
