# Processed Dataset & Split Architecture

## Data Splits
The dataset is split using stratified random sampling with a fixed seed (`random_state=42`) to strictly prevent data leakage between evaluation sets:

- **Train Set (`train.parquet`)**: 700 samples (70%) — used for fitting imputation, scaling, one-hot encoding, and model parameters.
- **Validation Set (`val.parquet`)**: 150 samples (15%) — used for hyperparameter tuning and decision threshold optimization.
- **Test Set (`test.parquet`)**: 150 samples (15%) — held-out benchmark set for out-of-sample performance evaluation.

## Storage Format
All processed splits are persisted using **Apache Parquet** format for efficient columnar storage, exact data type preservation, and rapid I/O serialization.

## Feature Engineering Applied
- `debt_to_duration`: Credit amount divided by loan duration in months (monthly repayment burden).
- `installment_burden`: Installment rate percentage multiplied by credit amount.
- `age_to_duration`: Age divided by loan duration (stability index).
- `has_guarantor`: Binary indicator for presence of a guarantor or co-applicant (`other_debtors != A101`).
- `has_savings_buffer`: Binary indicator for savings >= 500 DM (`savings_account in [A63, A64]`).
- `is_employed`: Binary indicator for active employment (`employment_since != A71`).
