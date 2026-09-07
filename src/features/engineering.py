"""
Feature engineering module for CreditRiskML.
Calculates domain-aware credit risk ratios and indicators.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain feature engineer for German credit risk dataset.
    Adds financial burden ratios and risk indicators deterministically.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Monthly credit burden ratio
        duration = df["duration_months"].replace(0, 1)
        df["credit_to_duration_ratio"] = df["credit_amount"] / duration

        # Estimated installment amount per month
        df["installment_rate_amount"] = df["credit_amount"] * (df["installment_rate_pct"] / 100.0)

        # Borrower maturity relative to loan span
        df["age_to_duration_ratio"] = df["age_years"] / duration

        # High-risk credit purposes (Car / Business)
        df["high_risk_purpose"] = df["purpose"].isin(["A40", "A41", "A49"]).astype(int)

        # Co-debtors or guarantors presence
        df["has_co_applicant_or_guarantor"] = df["other_debtors"].isin(["A102", "A103"]).astype(int)

        # Financial safety net indicators
        df["has_savings"] = (~df["savings_account"].isin(["A65"])).astype(int)
        df["is_employed"] = (~df["employment_since"].isin(["A71"])).astype(int)

        return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience functional interface for feature engineering."""
    transformer = CreditFeatureEngineer()
    return transformer.transform(df)
