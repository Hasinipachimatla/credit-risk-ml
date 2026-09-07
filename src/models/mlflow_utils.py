"""
MLflow experiment tracking and Model Registry utilities for CreditRiskML.
Manages run logging, artifact registration, and Champion/Challenger promotion gates.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow
from app.config import settings

logger = logging.getLogger("credit_risk_ml.mlflow_utils")

REGISTRY_FILE = Path(settings.MODEL_DIR) / "registry.json"

def setup_mlflow():
    """Initializes MLflow tracking URI and active experiment."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow configured: URI={settings.MLFLOW_TRACKING_URI}, Experiment={settings.MLFLOW_EXPERIMENT_NAME}")
    return experiment

def load_model_registry() -> Dict[str, Any]:
    """Loads the model registry state from disk."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {
        "champion": None,
        "challengers": [],
        "archived": []
    }

def save_model_registry(registry_data: Dict[str, Any]):
    """Persists model registry state to disk."""
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry_data, f, indent=2)
    logger.info(f"Updated model registry state in {REGISTRY_FILE}")

def check_promotion_criteria(
    champion_metrics: Optional[Dict[str, Any]],
    candidate_metrics: Dict[str, Any],
    min_roc_auc: float = 0.70,
    min_f1: float = 0.50,
    min_recall: float = 0.50,
    margin: float = 0.01
) -> Dict[str, Any]:
    """
    Evaluates whether a candidate model passes the automated promotion gate:
    1. Candidate ROC-AUC >= min_roc_auc
    2. Candidate F1 >= min_f1
    3. Candidate Recall >= min_recall
    4. If Champion exists: Candidate ROC-AUC >= Champion ROC-AUC + margin
    """
    cand_auc = candidate_metrics.get("roc_auc", 0.0)
    cand_f1 = candidate_metrics.get("f1", 0.0)
    cand_rec = candidate_metrics.get("recall", 0.0)

    reasons = []
    passed = True

    if cand_auc < min_roc_auc:
        passed = False
        reasons.append(f"ROC-AUC {cand_auc:.4f} is below minimum required {min_roc_auc:.4f}")

    if cand_f1 < min_f1:
        passed = False
        reasons.append(f"F1-score {cand_f1:.4f} is below minimum required {min_f1:.4f}")

    if cand_rec < min_recall:
        passed = False
        reasons.append(f"Recall {cand_rec:.4f} is below minimum required {min_recall:.4f}")

    if champion_metrics is not None:
        champ_auc = champion_metrics.get("roc_auc", 0.0)
        target_auc = champ_auc + margin
        if cand_auc < target_auc:
            passed = False
            reasons.append(f"Candidate ROC-AUC {cand_auc:.4f} does not beat Champion {champ_auc:.4f} by margin {margin:.4f}")

    return {
        "promoted": passed,
        "reasons": reasons,
        "candidate_metrics": candidate_metrics,
        "champion_metrics": champion_metrics
    }
