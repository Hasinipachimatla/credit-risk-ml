import pytest
import numpy as np
import pandas as pd

from src.monitoring.drift import calculate_psi, calculate_categorical_psi, compute_dataset_drift
from src.monitoring.performance import evaluate_observed_performance

def test_calculate_psi_no_drift():
    np.random.seed(42)
    reference = np.random.normal(loc=0.0, scale=1.0, size=1000)
    current = np.random.normal(loc=0.0, scale=1.0, size=1000)

    psi = calculate_psi(reference, current)
    assert psi < 0.1, f"Expected low PSI for identical distributions, got {psi}"

def test_calculate_psi_significant_drift():
    np.random.seed(42)
    reference = np.random.normal(loc=0.0, scale=1.0, size=1000)
    current = np.random.normal(loc=3.0, scale=1.0, size=1000)

    psi = calculate_psi(reference, current)
    assert psi > 0.25, f"Expected high PSI for shifted distribution, got {psi}"

def test_calculate_categorical_psi():
    ref_cats = pd.Series(["A", "A", "B", "B", "C"] * 100)
    cur_cats_same = pd.Series(["A", "A", "B", "B", "C"] * 100)
    psi_low = calculate_categorical_psi(ref_cats, cur_cats_same)
    assert psi_low < 0.05

    cur_cats_shifted = pd.Series(["C", "C", "C", "C", "B"] * 100)
    psi_high = calculate_categorical_psi(ref_cats, cur_cats_shifted)
    assert psi_high > 0.20

def test_compute_dataset_drift_synthetic():
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "num_feat": np.random.normal(0, 1, 200),
        "cat_feat": np.random.choice(["X", "Y", "Z"], size=200)
    })
    cur_df = pd.DataFrame({
        "num_feat": np.random.normal(0, 1, 100),
        "cat_feat": np.random.choice(["X", "Y", "Z"], size=100)
    })

    drift_report = compute_dataset_drift(current_df=cur_df, reference_df=ref_df)
    assert "total_features" in drift_report
    assert drift_report["total_features"] == 2
    assert "dataset_drift_detected" in drift_report
    assert "features" in drift_report
    assert len(drift_report["features"]) == 2

def test_evaluate_observed_performance_insufficient():
    # Less than 20 samples
    y_true = [0, 1, 0, 1, 0]
    y_probs = [0.2, 0.7, 0.3, 0.8, 0.4]

    result = evaluate_observed_performance(y_true, y_probs)
    assert result["status"] == "insufficient ground truth"
    assert result["current_metrics"] is None
    assert result["degradation_reasons"] == []

def test_evaluate_observed_performance_sufficient():
    # 25 samples
    y_true = [0, 1] * 12 + [0]
    y_probs = [0.1, 0.9] * 12 + [0.2]

    result = evaluate_observed_performance(y_true, y_probs, threshold=0.5)
    assert result["status"] == "evaluated"
    assert result["current_metrics"] is not None
    assert "roc_auc" in result["current_metrics"]
    assert "degradation_detected" in result
