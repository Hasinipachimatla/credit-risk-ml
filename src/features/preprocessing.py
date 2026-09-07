"""
Preprocessing and data splitting module for CreditRiskML.
Constructs a leakage-free Scikit-Learn Pipeline and performs stratified train/val/test split.
"""
import os
import joblib
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from app.config import settings
from src.data.ingestion import load_data, TARGET_COLUMN
from src.data.validation import validate_credit_data
from src.features.engineering import CreditFeatureEngineer

logger = logging.getLogger("credit_risk_ml.preprocessing")

# Feature lists
NUMERICAL_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate_pct",
    "residence_since_years",
    "age_years",
    "existing_credits",
    "people_liable",
    # Engineered numeric features
    "credit_to_duration_ratio",
    "installment_rate_amount",
    "age_to_duration_ratio",
    "high_risk_purpose",
    "has_co_applicant_or_guarantor",
    "has_savings",
    "is_employed"
]

CATEGORICAL_FEATURES = [
    "status_checking",
    "credit_history",
    "purpose",
    "savings_account",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]

def build_feature_pipeline() -> Pipeline:
    """
    Builds the full, leak-free feature engineering and preprocessing pipeline.
    Expects raw dataframe inputs, applies feature engineering, imputation,
    scaling, and one-hot encoding.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline([
        ("feature_engineer", CreditFeatureEngineer()),
        ("column_preprocessor", column_preprocessor)
    ])

    return full_pipeline

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into stratified train, validation, and test subsets (70/15/15).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # Step 1: Split into Train (70%) and Temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df[TARGET_COLUMN],
        random_state=random_state
    )

    # Step 2: Split Temp into Validation (15%) and Test (15%)
    val_size_relative = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_size_relative),
        stratify=temp_df[TARGET_COLUMN],
        random_state=random_state
    )

    logger.info(
        f"Split dataset into Train: {len(train_df)} ({len(train_df)/len(df):.1%}), "
        f"Val: {len(val_df)} ({len(val_df)/len(df):.1%}), "
        f"Test: {len(test_df)} ({len(test_df)/len(df):.1%})"
    )
    return train_df, val_df, test_df

def run_preprocessing() -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes data loading, validation, splitting, fitting the preprocessor on train only,
    and persisting processed datasets and fitted preprocessor.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("=== Running Preprocessing Pipeline ===")

    # 1. Load and validate
    df = load_data()
    validated_df = validate_credit_data(df, is_training=True)

    # 2. Stratified 70/15/15 split
    train_df, val_df, test_df = split_dataset(validated_df)

    # 3. Fit preprocessor STRICTLY on train_df features only (zero data leakage)
    X_train_raw = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].values

    pipeline = build_feature_pipeline()
    logger.info("Fitting feature engineering & preprocessing pipeline on training split only...")
    X_train_trans = pipeline.fit_transform(X_train_raw)

    # 4. Save preprocessor artifact
    models_dir = Path(settings.MODEL_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = models_dir / "preprocessor.joblib"
    joblib.dump(pipeline, preprocessor_path)
    logger.info(f"Saved fitted preprocessor to {preprocessor_path}")

    # 5. Save processed data and baseline reference distribution
    processed_dir = Path(settings.DATA_DIR) / "processed"
    reference_dir = Path(settings.DATA_DIR) / "reference"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

    # Baseline reference features for drift detection (training split)
    X_train_raw.to_parquet(reference_dir / "baseline_features.parquet", index=False)
    logger.info(f"Saved baseline reference dataset to {reference_dir / 'baseline_features.parquet'}")
    logger.info("=== Preprocessing Pipeline Completed Successfully ===")

    return pipeline, train_df, val_df, test_df

if __name__ == "__main__":
    run_preprocessing()
