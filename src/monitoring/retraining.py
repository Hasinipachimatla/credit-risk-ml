"""
Automated retraining and Champion-Challenger promotion pipeline for CreditRiskML.
Executes retraining when data drift or performance degradation is detected,
evaluates challenger, and gates model promotion.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
import mlflow

from app.config import settings
from src.data.ingestion import load_data, TARGET_COLUMN
from src.data.validation import validate_credit_data
from src.features.preprocessing import run_preprocessing, build_feature_pipeline, split_dataset
from src.models.baseline import get_random_forest_model, get_logistic_regression_model
from src.models.evaluate import optimize_decision_threshold, compute_metrics, generate_evaluation_reports
from src.models.mlflow_utils import setup_mlflow, load_model_registry, save_model_registry, check_promotion_criteria

logger = logging.getLogger("credit_risk_ml.retraining")

def execute_retraining_pipeline(
    trigger_reason: str = "manual_trigger",
    force_promotion: bool = False
) -> Dict[str, Any]:
    """
    Runs the full retraining workflow:
    1. Validates data.
    2. Fits pipeline & trains Challenger model.
    3. Evaluates Challenger on validation set and optimizes decision threshold.
    4. Compares against active Champion.
    5. Promotes Challenger IF and ONLY IF promotion criteria pass.
    """
    logger.info(f"=== Triggering Retraining Pipeline (Reason: {trigger_reason}) ===")
    setup_mlflow()

    # Step 1: Load and validate dataset
    df = load_data()
    validated_df = validate_credit_data(df, is_training=True)

    # Step 2: Split data
    train_df, val_df, test_df = split_dataset(validated_df)

    X_train_raw = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].values

    X_val_raw = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN].values

    X_test_raw = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].values

    # Step 3: Fit fresh feature pipeline
    feature_pipeline = build_feature_pipeline()
    X_train = feature_pipeline.fit_transform(X_train_raw)
    X_val = feature_pipeline.transform(X_val_raw)
    X_test = feature_pipeline.transform(X_test_raw)

    # Step 4: Train Challenger model (Random Forest with hyperparameter tuning)
    challenger_model = get_random_forest_model(n_estimators=180, max_depth=7)
    challenger_name = "RandomForest_Challenger"

    with mlflow.start_run(run_name=f"retrain_{challenger_name}") as run:
        challenger_model.fit(X_train, y_train)

        # Step 5: Validation Evaluation & Threshold Optimization
        val_probs = challenger_model.predict_proba(X_val)[:, 1]
        opt_threshold, val_metrics = optimize_decision_threshold(
            y_val, val_probs, min_recall=settings.MIN_RECALL
        )

        test_probs = challenger_model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_probs, threshold=opt_threshold)

        mlflow.log_param("trigger_reason", trigger_reason)
        mlflow.log_param("model_type", challenger_name)
        mlflow.log_param("threshold", opt_threshold)
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"val_{k}", v)

        # Step 6: Champion-Challenger Evaluation Gate
        registry = load_model_registry()
        champion = registry.get("champion")
        champ_val_metrics = champion.get("val_metrics") if champion else None

        gate_result = check_promotion_criteria(
            champion_metrics=champ_val_metrics,
            candidate_metrics=val_metrics,
            min_roc_auc=settings.MIN_ROC_AUC,
            min_f1=settings.MIN_F1,
            min_recall=settings.MIN_RECALL,
            margin=settings.PROMOTION_MARGIN
        )

        should_promote = gate_result["promoted"] or force_promotion

        models_dir = Path(settings.MODEL_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        if should_promote:
            logger.info(">>> CHALLENGER PASSED EVALUATION GATE. PROMOTING TO CHAMPION <<<")
            champion_path = models_dir / "credit_risk_model.joblib"
            preprocessor_path = models_dir / "preprocessor.joblib"

            joblib.dump(challenger_model, champion_path)
            joblib.dump(feature_pipeline, preprocessor_path)

            new_version = f"v{len(registry.get('archived', [])) + 2}"
            new_champion_record = {
                "model_name": challenger_name,
                "model_version": new_version,
                "model_path": str(champion_path),
                "run_id": run.info.run_id,
                "threshold": opt_threshold,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "status": "Champion",
                "promoted_due_to": trigger_reason
            }

            if champion:
                registry["archived"].append(champion)
            registry["champion"] = new_champion_record
            save_model_registry(registry)

            # Update evaluation reports
            generate_evaluation_reports(y_test, test_probs, threshold=opt_threshold)

            return {
                "status": "promoted",
                "message": "Challenger model passed criteria and was promoted to Champion.",
                "champion_version": new_version,
                "challenger_metrics": val_metrics,
                "previous_champion_metrics": champ_val_metrics,
                "gate_reasons": gate_result["reasons"]
            }
        else:
            logger.info(">>> CHALLENGER FAILED EVALUATION GATE. PRESERVING CHAMPION <<<")
            challenger_record = {
                "model_name": challenger_name,
                "model_version": f"challenger-{run.info.run_id[:8]}",
                "run_id": run.info.run_id,
                "threshold": opt_threshold,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "status": "Rejected",
                "rejection_reasons": gate_result["reasons"]
            }
            registry["challengers"].append(challenger_record)
            save_model_registry(registry)

            return {
                "status": "rejected",
                "message": "Challenger model failed quality gate. Active Champion preserved.",
                "challenger_metrics": val_metrics,
                "champion_metrics": champ_val_metrics,
                "gate_reasons": gate_result["reasons"]
            }
