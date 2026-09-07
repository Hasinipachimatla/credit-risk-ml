"""
Baseline classical models for CreditRiskML.
Provides Logistic Regression and Random Forest classifiers with class imbalance handling.
"""
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_logistic_regression_model(
    C: float = 0.5,
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """
    Returns a Logistic Regression model tuned for imbalanced credit risk.
    Uses L2 regularization and balanced class weighting.
    """
    return LogisticRegression(
        C=C,
        penalty="l2",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state
    )

def get_random_forest_model(
    n_estimators: int = 150,
    max_depth: int = 6,
    min_samples_split: int = 5,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Returns a Random Forest Classifier tuned for tabular credit risk.
    Uses balanced subsample weighting to prevent majority class bias.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
