"""
Data validation module for CreditRiskML using Pandera.
Enforces strict schema rules, categorical domains, and numeric ranges.
"""
import sys
import logging
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema
from src.data.ingestion import load_data, TARGET_COLUMN

logger = logging.getLogger("credit_risk_ml.validation")

def get_credit_data_schema(is_training: bool = True) -> DataFrameSchema:
    """Builds and returns the Pandera validation schema."""
    columns = {
        "status_checking": Column(
            pa.String,
            Check.isin(["A11", "A12", "A13", "A14"]),
            nullable=False,
            description="Status of existing checking account"
        ),
        "duration_months": Column(
            pa.Float,
            Check.in_range(1, 120),
            nullable=False,
            coerce=True,
            description="Duration in months"
        ),
        "credit_history": Column(
            pa.String,
            Check.isin(["A30", "A31", "A32", "A33", "A34"]),
            nullable=False,
            description="Credit history"
        ),
        "purpose": Column(
            pa.String,
            Check.isin(["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"]),
            nullable=False,
            description="Loan purpose"
        ),
        "credit_amount": Column(
            pa.Float,
            Check.in_range(100, 50000),
            nullable=False,
            coerce=True,
            description="Credit amount in DM"
        ),
        "savings_account": Column(
            pa.String,
            Check.isin(["A61", "A62", "A63", "A64", "A65"]),
            nullable=False,
            description="Savings account / bonds"
        ),
        "employment_since": Column(
            pa.String,
            Check.isin(["A71", "A72", "A73", "A74", "A75"]),
            nullable=False,
            description="Present employment since"
        ),
        "installment_rate_pct": Column(
            pa.Float,
            Check.in_range(1, 100),
            nullable=False,
            coerce=True,
            description="Installment rate in percentage of disposable income"
        ),
        "personal_status_sex": Column(
            pa.String,
            Check.isin(["A91", "A92", "A93", "A94", "A95"]),
            nullable=False,
            description="Personal status and sex"
        ),
        "other_debtors": Column(
            pa.String,
            Check.isin(["A101", "A102", "A103"]),
            nullable=False,
            description="Other debtors / guarantors"
        ),
        "residence_since_years": Column(
            pa.Float,
            Check.in_range(1, 100),
            nullable=False,
            coerce=True,
            description="Present residence duration in years"
        ),
        "property": Column(
            pa.String,
            Check.isin(["A121", "A122", "A123", "A124"]),
            nullable=False,
            description="Property description"
        ),
        "age_years": Column(
            pa.Float,
            Check.in_range(18, 120),
            nullable=False,
            coerce=True,
            description="Age in years"
        ),
        "other_installment_plans": Column(
            pa.String,
            Check.isin(["A141", "A142", "A143"]),
            nullable=False,
            description="Other installment plans"
        ),
        "housing": Column(
            pa.String,
            Check.isin(["A151", "A152", "A153"]),
            nullable=False,
            description="Housing arrangement"
        ),
        "existing_credits": Column(
            pa.Float,
            Check.in_range(1, 20),
            nullable=False,
            coerce=True,
            description="Number of existing credits at this bank"
        ),
        "job": Column(
            pa.String,
            Check.isin(["A171", "A172", "A173", "A174"]),
            nullable=False,
            description="Employment job category"
        ),
        "people_liable": Column(
            pa.Float,
            Check.in_range(1, 20),
            nullable=False,
            coerce=True,
            description="Number of people liable for maintenance"
        ),
        "telephone": Column(
            pa.String,
            Check.isin(["A191", "A192"]),
            nullable=False,
            description="Telephone registration status"
        ),
        "foreign_worker": Column(
            pa.String,
            Check.isin(["A201", "A202"]),
            nullable=False,
            description="Foreign worker status"
        )
    }

    if is_training:
        columns[TARGET_COLUMN] = Column(
            pa.Int,
            Check.isin([0, 1]),
            nullable=False,
            coerce=True,
            description="Target: 0 for Good (Non-default), 1 for Bad (Default)"
        )

    return DataFrameSchema(columns=columns, strict=False, coerce=True)

def validate_credit_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Validates dataframe against schema and returns coerced validated dataframe."""
    schema = get_credit_data_schema(is_training=is_training)
    try:
        validated_df = schema.validate(df)
        logger.info(f"Data validation passed successfully ({len(validated_df)} rows).")
        return validated_df
    except pa.errors.SchemaErrors as err:
        logger.error(f"Schema validation failed with multiple errors: {err.failure_cases}")
        raise
    except pa.errors.SchemaError as err:
        logger.error(f"Schema validation error: {err}")
        raise

def run_validation() -> None:
    """CLI runner for validation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("=== Validating Raw Credit Dataset ===")
    df = load_data()
    validated_df = validate_credit_data(df, is_training=True)
    print("Validation Succeeded: All schema constraints, data types, and value bounds verified!")
    print(f"Validated Shape: {validated_df.shape}")

if __name__ == "__main__":
    run_validation()
