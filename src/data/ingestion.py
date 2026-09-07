"""
Data ingestion module for CreditRiskML.
Downloads and loads the Statlog German Credit Data from UCI Repository.
"""
import os
import sys
import logging
import urllib.request
import pandas as pd
from pathlib import Path
from app.config import settings

logger = logging.getLogger("credit_risk_ml.ingestion")

UCI_GERMAN_DATA_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/credit.csv"
]

COLUMN_NAMES = [
    "status_checking",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors",
    "residence_since_years",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "credit_risk"  # 1 = Good, 2 = Bad in original UCI data
]

NUMERICAL_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate_pct",
    "residence_since_years",
    "age_years",
    "existing_credits",
    "people_liable"
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

TARGET_COLUMN = "credit_risk"

def get_raw_data_dir() -> Path:
    raw_dir = Path(settings.DATA_DIR) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir

def download_german_credit_data(output_path: Path) -> pd.DataFrame:
    """Download German Credit dataset from UCI ML Repository."""
    logger.info("Attempting to fetch Statlog German Credit data from UCI...")
    raw_text = None
    for url in UCI_GERMAN_DATA_URLS:
        try:
            logger.info(f"Trying URL: {url}")
            with urllib.request.urlopen(url, timeout=15) as response:
                if response.status == 200:
                    raw_text = response.read().decode("utf-8")
                    logger.info(f"Successfully retrieved dataset from {url}")
                    break
        except Exception as e:
            logger.warning(f"Failed to fetch from {url}: {e}")

    if not raw_text:
        raise RuntimeError("Failed to download dataset from all configured URLs.")

    lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]
    rows = [line.split() for line in lines]
    df = pd.DataFrame(rows, columns=COLUMN_NAMES)

    # Cast numerical columns
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Map target: 1 -> 0 (Good / Non-default), 2 -> 1 (Bad / Default)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({1: 0, 2: 1})

    df.to_csv(output_path, index=False)
    logger.info(f"Saved raw German Credit dataset to {output_path}")
    return df

def load_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load raw credit data from disk, or download if missing.
    """
    raw_dir = get_raw_data_dir()
    data_path = raw_dir / "german_credit.csv"

    if not force_download and data_path.exists():
        logger.info(f"Loading cached raw dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        df = download_german_credit_data(data_path)

    return df

def run_ingestion() -> pd.DataFrame:
    """CLI and pipeline entrypoint for data ingestion."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("=== Starting Data Ingestion ===")
    df = load_data()
    print("--- Ingestion Summary ---")
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print("Target Distribution:")
    print(df[TARGET_COLUMN].value_counts(normalize=True).rename({0: "0 (Good/Non-default)", 1: "1 (Bad/Default)"}))
    logger.info("=== Data Ingestion Completed Successfully ===")
    return df

if __name__ == "__main__":
    run_ingestion()
