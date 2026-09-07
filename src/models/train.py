"""
Master training pipeline for CreditRiskML.
Trains baseline models (Logistic Regression & Random Forest), tunes decision thresholds,
tracks experiments in MLflow, and registers the winning Champion model.
"""
import os
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import mlflow

from app.config import settings
from src.data.ingestion import TARGET_COLUMN
from src.features.preprocessing import run_preprocessing
from src.models.baseline import get_logistic_regression_model, get_random_forest_model
from src.models.evaluate import optimize_decision_threshold, compute_metrics, generate_evaluation_reports
from src.models.mlflow_utils import setup_mlflow, load_model_registry, save_model_registry, check_promotion_criteria

logger = logging.getLogger("credit_risk_ml.train")

def load_splits_and_preprocessor() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """Loads processed splits and fitted preprocessor; runs preprocessing if missing."""
    processed_dir = Path(settings.DATA_DIR) / "processed"
    preprocessor_path = Path(settings.MODEL_DIR) / "preprocessor.joblib"

    if not (processed_dir / "train.parquet").exists() or not preprocessor_path.exists():
        logger.info("Processed data or preprocessor not found. Running preprocessing first...")
        pipeline, train_df, val_df, test_df = run_preprocessing()
        return train_df, val_df, test_df, pipeline

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    pipeline = joblib.load(preprocessor_path)
    return train_df, val_df, test_df, pipeline

def train_and_evaluate_baselines():
    """Trains Logistic Regression and Random Forest models and registers the champion."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("=== Starting Model Training Pipeline ===")

    setup_mlflow()
    train_df, val_df, test_df, preprocessor = load_splits_and_preprocessor()

    X_train_raw = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].values

    X_val_raw = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN].values

    X_test_raw = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].values

    # Transform features through fitted preprocessor
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    models_to_train = {
        "LogisticRegression": get_logistic_regression_model(),
        "RandomForest": get_random_forest_model()
    }

    results = {}

    for model_name, model in models_to_train.items():
        with mlflow.start_run(run_name=f"baseline_{model_name}") as run:
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)

            # Predict probabilities on Validation set
            val_probs = model.predict_proba(X_val)[:, 1]

            # Optimize decision threshold based on Validation set
            opt_threshold, val_metrics = optimize_decision_threshold(y_val, val_probs, min_recall=settings.MIN_RECALL)
            logger.info(f"{model_name} Validation Metrics (Threshold = {opt_threshold:.3f}): {val_metrics}")

            # Log to MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("threshold", opt_threshold)
            mlflow.log_param("features_dim", X_train.shape[1])
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"val_{k}", v)

            results[model_name] = {
                "model": model,
                "threshold": opt_threshold,
                "val_metrics": val_metrics,
                "run_id": run.info.run_id
            }

    # Select best model based on Validation ROC-AUC & F1
    best_name = max(results.keys(), key=lambda name: (results[name]["val_metrics"]["roc_auc"], results[name]["val_metrics"]["f1"]))
    best_info = results[best_name]
    logger.info(f"=== Best Model Selected: {best_name} (Val ROC-AUC: {best_info['val_metrics']['roc_auc']:.4f}) ===")

    # Final Evaluation on unseen Test split
    winning_model = best_info["model"]
    winning_threshold = best_info["threshold"]
    test_probs = winning_model.predict_proba(X_test)[:, 1]

    reports_dir = Path("reports")
    test_metrics = generate_evaluation_reports(y_test, test_probs, threshold=winning_threshold, reports_dir=reports_dir)
    logger.info(f"Final Test Evaluation Metrics: {test_metrics}")

    # Save champion model artifact
    models_dir = Path(settings.MODEL_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    champion_path = models_dir / "credit_risk_model.joblib"
    joblib.dump(winning_model, champion_path)
    logger.info(f"Saved Champion model artifact to {champion_path}")

    # Update model registry
    registry = load_model_registry()
    new_champion_record = {
        "model_name": best_name,
        "model_version": settings.MODEL_VERSION,
        "model_path": str(champion_path),
        "run_id": best_info["run_id"],
        "threshold": winning_threshold,
        "val_metrics": best_info["val_metrics"],
        "test_metrics": test_metrics,
        "status": "Champion"
    }

    if registry.get("champion"):
        registry["archived"].append(registry["champion"])
    registry["champion"] = new_champion_record
    save_model_registry(registry)

    print("\n================ FINAL MODEL COMPARISON ================")
    for name, res in results.items():
        m = res["val_metrics"]
        print(f"Model: {name:<20} | Val ROC-AUC: {m['roc_auc']:.4f} | Val F1: {m['f1']:.4f} | Val Recall: {m['recall']:.4f}")
    print(f"\nChampion Promoted: {best_name} (Test ROC-AUC: {test_metrics['roc_auc']:.4f}, Test F1: {test_metrics['f1']:.4f})")
    print(f"Decision Threshold: {winning_threshold:.3f}")
    print("========================================================\n")

    return results, best_name

if __name__ == "__main__":
    train_and_evaluate_baselines()
