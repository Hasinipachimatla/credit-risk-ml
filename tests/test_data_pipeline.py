import pytest
import pandas as pd
import numpy as np
import pandera.errors

from src.data.validation import validate_credit_data
from src.features.engineering import CreditFeatureEngineer, engineer_features
from src.features.preprocessing import build_feature_pipeline, split_dataset
from src.data.ingestion import COLUMN_NAMES, TARGET_COLUMN

@pytest.fixture
def sample_credit_row():
    return {
        "status_checking": "A11",
        "duration_months": 24,
        "credit_history": "A32",
        "purpose": "A40",
        "credit_amount": 4500,
        "savings_account": "A61",
        "employment_since": "A73",
        "installment_rate_pct": 4,
        "personal_status_sex": "A93",
        "other_debtors": "A101",
        "residence_since_years": 4,
        "property": "A121",
        "age_years": 35,
        "other_installment_plans": "A143",
        "housing": "A152",
        "existing_credits": 1,
        "job": "A173",
        "people_liable": 1,
        "telephone": "A191",
        "foreign_worker": "A201",
        "credit_risk": 1
    }

@pytest.fixture
def sample_dataframe(sample_credit_row):
    rows = []
    for i in range(50):
        row = sample_credit_row.copy()
        row["credit_amount"] = 1000 + i * 100
        row["duration_months"] = 6 + (i % 30)
        row["credit_risk"] = 1 if i % 3 != 0 else 0
        rows.append(row)
    return pd.DataFrame(rows)

def test_validate_credit_data_valid(sample_dataframe):
    validated = validate_credit_data(sample_dataframe, is_training=True)
    assert len(validated) == len(sample_dataframe)
    assert TARGET_COLUMN in validated.columns

def test_validate_credit_data_invalid_range(sample_dataframe):
    corrupted_df = sample_dataframe.copy()
    corrupted_df.loc[0, "age_years"] = -10  # Invalid age
    with pytest.raises(pandera.errors.SchemaError):
        validate_credit_data(corrupted_df, is_training=True)

def test_validate_credit_data_invalid_category(sample_dataframe):
    corrupted_df = sample_dataframe.copy()
    corrupted_df.loc[0, "status_checking"] = "UNKNOWN_CODE"
    with pytest.raises(pandera.errors.SchemaError):
        validate_credit_data(corrupted_df, is_training=True)

def test_feature_engineering_calculations(sample_dataframe):
    fe = CreditFeatureEngineer()
    engineered = fe.transform(sample_dataframe)

    # Check that new features were created
    expected_cols = [
        "credit_to_duration_ratio",
        "installment_rate_amount",
        "age_to_duration_ratio",
        "high_risk_purpose",
        "has_co_applicant_or_guarantor",
        "has_savings",
        "is_employed"
    ]
    for col in expected_cols:
        assert col in engineered.columns

    # Verify mathematical accuracy of ratios
    row = engineered.iloc[0]
    expected_ratio = row["credit_amount"] / row["duration_months"]
    assert np.isclose(row["credit_to_duration_ratio"], expected_ratio)

    expected_installment = row["credit_amount"] * (row["installment_rate_pct"] / 100.0)
    assert np.isclose(row["installment_rate_amount"], expected_installment)

def test_split_dataset(sample_dataframe):
    train_df, val_df, test_df = split_dataset(sample_dataframe, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_dataframe)
    assert len(train_df) == 30
    assert len(val_df) == 10
    assert len(test_df) == 10

def test_build_feature_pipeline_fit_transform(sample_dataframe):
    pipeline = build_feature_pipeline()
    X = sample_dataframe.drop(columns=[TARGET_COLUMN])
    transformed = pipeline.fit_transform(X)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(sample_dataframe)
    assert transformed.shape[1] > 0
    assert not np.isnan(transformed).any()
