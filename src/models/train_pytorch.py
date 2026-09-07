"""
PyTorch training script for CreditRiskML.
Trains the tabular neural network, optimizes decision threshold,
evaluates performance, logs run to MLflow, and registers checkpoint.
"""
import os
import joblib
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import mlflow

from app.config import settings
from src.data.ingestion import TARGET_COLUMN
from src.models.train import load_splits_and_preprocessor
from src.models.pytorch_model import CreditRiskNet, train_pytorch_net, predict_proba_pytorch
from src.models.evaluate import optimize_decision_threshold, compute_metrics
from src.models.mlflow_utils import setup_mlflow, load_model_registry, save_model_registry

logger = logging.getLogger("credit_risk_ml.train_pytorch")

def run_pytorch_training():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("=== Starting PyTorch Deep Learning Training Pipeline ===")

    setup_mlflow()
    train_df, val_df, test_df, preprocessor = load_splits_and_preprocessor()

    X_train_raw = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].values.astype(np.float32)

    X_val_raw = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN].values.astype(np.float32)

    X_test_raw = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].values.astype(np.float32)

    # Transform features
    X_train = preprocessor.transform(X_train_raw).astype(np.float32)
    X_val = preprocessor.transform(X_val_raw).astype(np.float32)
    X_test = preprocessor.transform(X_test_raw).astype(np.float32)

    models_dir = Path(settings.MODEL_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "pytorch_credit_model.pt"

    with mlflow.start_run(run_name="deep_learning_PyTorchNet") as run:
        logger.info("Training Tabular CreditRiskNet...")
        model, history = train_pytorch_net(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=50,
            batch_size=32,
            lr=1e-3,
            patience=10,
            checkpoint_path=checkpoint_path
        )

        # Validation evaluation
        val_probs = predict_proba_pytorch(model, X_val)
        opt_threshold, val_metrics = optimize_decision_threshold(y_val, val_probs, min_recall=settings.MIN_RECALL)
        logger.info(f"PyTorch Validation Metrics (Threshold = {opt_threshold:.3f}): {val_metrics}")

        # Test evaluation
        test_probs = predict_proba_pytorch(model, X_test)
        test_metrics = compute_metrics(y_test, test_probs, threshold=opt_threshold)
        logger.info(f"PyTorch Test Evaluation Metrics: {test_metrics}")

        # Log to MLflow
        mlflow.log_param("model_name", "PyTorch_CreditRiskNet")
        mlflow.log_param("epochs", 50)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("threshold", opt_threshold)
        mlflow.log_param("input_dim", X_train.shape[1])

        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"val_{k}", v)

        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", v)

        # Register Challenger model in registry
        registry = load_model_registry()
        challenger_record = {
            "model_name": "PyTorch_CreditRiskNet",
            "model_version": "v1-pytorch",
            "model_path": str(checkpoint_path),
            "run_id": run.info.run_id,
            "threshold": opt_threshold,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "status": "Challenger"
        }
        registry["challengers"].append(challenger_record)
        save_model_registry(registry)

        print("\n================ PYTORCH MODEL EVALUATION ================")
        print(f"Val ROC-AUC:  {val_metrics['roc_auc']:.4f} | Val F1:  {val_metrics['f1']:.4f} | Val Recall:  {val_metrics['recall']:.4f}")
        print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f} | Test F1: {test_metrics['f1']:.4f} | Test Recall: {test_metrics['recall']:.4f}")
        print(f"Decision Threshold: {opt_threshold:.3f}")
        print(f"Saved Checkpoint: {checkpoint_path}")
        print("==========================================================\n")

    return model, val_metrics, test_metrics

if __name__ == "__main__":
    run_pytorch_training()
