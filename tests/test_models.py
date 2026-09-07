import pytest
import numpy as np
import torch

from src.models.baseline import get_logistic_regression_model, get_random_forest_model
from src.models.evaluate import compute_metrics, optimize_decision_threshold
from src.models.pytorch_model import CreditRiskNet
from src.models.mlflow_utils import check_promotion_criteria

def test_baseline_models_fit_and_predict():
    X_dummy = np.random.randn(50, 10)
    y_dummy = np.random.randint(0, 2, size=50)

    # Test Logistic Regression
    lr = get_logistic_regression_model()
    lr.fit(X_dummy, y_dummy)
    probs_lr = lr.predict_proba(X_dummy)
    assert probs_lr.shape == (50, 2)
    assert np.all((probs_lr >= 0) & (probs_lr <= 1))

    # Test Random Forest
    rf = get_random_forest_model(n_estimators=10)
    rf.fit(X_dummy, y_dummy)
    probs_rf = rf.predict_proba(X_dummy)
    assert probs_rf.shape == (50, 2)

def test_compute_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.4, 0.6])
    metrics = compute_metrics(y_true, y_probs, threshold=0.5)

    required_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score", "confusion_matrix"]
    for key in required_keys:
        assert key in metrics

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0

def test_optimize_decision_threshold():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    best_thresh, best_metrics = optimize_decision_threshold(y_true, y_probs, min_recall=0.75)
    assert 0.0 < best_thresh < 1.0
    assert best_metrics["recall"] >= 0.75

def test_pytorch_credit_risk_net_forward():
    input_dim = 45
    batch_size = 16
    model = CreditRiskNet(input_dim=input_dim, hidden_dim1=32, hidden_dim2=16, dropout_rate=0.2)
    model.eval()

    dummy_tensor = torch.randn(batch_size, input_dim)
    with torch.no_grad():
        logits = model(dummy_tensor)
        probs = torch.sigmoid(logits)

    assert logits.shape == (batch_size, 1)
    probs_np = probs.numpy()
    assert np.all((probs_np >= 0.0) & (probs_np <= 1.0))

def test_check_promotion_criteria():
    champion_metrics = {
        "roc_auc": 0.75,
        "f1": 0.60,
        "recall": 0.70
    }

    # Case 1: Superior candidate exceeding margin and constraints
    superior_candidate = {
        "roc_auc": 0.80,
        "f1": 0.65,
        "recall": 0.75
    }
    result = check_promotion_criteria(champion_metrics, superior_candidate, min_roc_auc=0.70, margin=0.01)
    assert result["promoted"] is True

    # Case 2: Inferior candidate
    inferior_candidate = {
        "roc_auc": 0.74,
        "f1": 0.58,
        "recall": 0.65
    }
    result_inferior = check_promotion_criteria(champion_metrics, inferior_candidate, min_roc_auc=0.70, margin=0.01)
    assert result_inferior["promoted"] is False

    # Case 3: Failing minimum safety threshold
    low_safety_candidate = {
        "roc_auc": 0.82,
        "f1": 0.30,  # Below min_f1
        "recall": 0.75
    }
    result_low_safety = check_promotion_criteria(champion_metrics, low_safety_candidate, min_roc_auc=0.70, min_f1=0.50)
    assert result_low_safety["promoted"] is False
