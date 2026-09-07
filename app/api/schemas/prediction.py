"""
Pydantic schemas for credit risk prediction requests and responses.
Matches the exact schema and domain of the German Credit Risk dataset.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class BorrowerApplication(BaseModel):
    """
    Schema for a borrower loan application with credit and financial history.
    """
    status_checking: str = Field(
        ...,
        description="Status of existing checking account (A11: < 0 DM, A12: 0-200 DM, A13: >= 200 DM, A14: no account)",
        examples=["A11"]
    )
    duration_months: float = Field(
        ...,
        ge=1,
        le=120,
        description="Duration of the credit in months",
        examples=[24.0]
    )
    credit_history: str = Field(
        ...,
        description="Credit history status (A30: no credits/all paid, A31: all credits paid back, A32: existing credits paid, A33: delay, A34: critical account)",
        examples=["A32"]
    )
    purpose: str = Field(
        ...,
        description="Purpose of credit (A40: car new, A41: car used, A42: furniture, A43: radio/tv, A44: domestic, A49: business, A410: others)",
        examples=["A40"]
    )
    credit_amount: float = Field(
        ...,
        ge=100,
        le=50000,
        description="Credit amount requested in DM",
        examples=[4500.0]
    )
    savings_account: str = Field(
        ...,
        description="Savings account/bonds (A61: < 100 DM, A62: 100-500 DM, A63: 500-1000 DM, A64: >= 1000 DM, A65: unknown/no savings)",
        examples=["A61"]
    )
    employment_since: str = Field(
        ...,
        description="Present employment duration (A71: unemployed, A72: < 1 yr, A73: 1-4 yrs, A74: 4-7 yrs, A75: >= 7 yrs)",
        examples=["A73"]
    )
    installment_rate_pct: float = Field(
        ...,
        ge=1,
        le=100,
        description="Installment rate in percentage of disposable income",
        examples=[4.0]
    )
    personal_status_sex: str = Field(
        ...,
        description="Personal status and sex (A91: male div/sep, A92: female div/sep/mar, A93: male single, A94: male mar/wid, A95: female single)",
        examples=["A93"]
    )
    other_debtors: str = Field(
        ...,
        description="Other debtors / guarantors (A101: none, A102: co-applicant, A103: guarantor)",
        examples=["A101"]
    )
    residence_since_years: float = Field(
        ...,
        ge=1,
        le=100,
        description="Present residence span in years",
        examples=[4.0]
    )
    property: str = Field(
        ...,
        description="Property ownership (A121: real estate, A122: building society savings/life insurance, A123: car/other, A124: unknown/no property)",
        examples=["A121"]
    )
    age_years: float = Field(
        ...,
        ge=18,
        le=120,
        description="Borrower age in years",
        examples=[35.0]
    )
    other_installment_plans: str = Field(
        ...,
        description="Other installment plans (A141: bank, A142: stores, A143: none)",
        examples=["A143"]
    )
    housing: str = Field(
        ...,
        description="Housing arrangement (A151: rent, A152: own, A153: for free)",
        examples=["A152"]
    )
    existing_credits: float = Field(
        ...,
        ge=1,
        le=20,
        description="Number of existing credits at this bank",
        examples=[1.0]
    )
    job: str = Field(
        ...,
        description="Employment category (A171: unemployed/unskilled non-resident, A172: unskilled resident, A173: skilled employee, A174: management/self-employed)",
        examples=["A173"]
    )
    people_liable: float = Field(
        ...,
        ge=1,
        le=20,
        description="Number of people liable for maintenance",
        examples=[1.0]
    )
    telephone: str = Field(
        ...,
        description="Registered telephone (A191: none, A192: yes registered under customer name)",
        examples=["A192"]
    )
    foreign_worker: str = Field(
        ...,
        description="Foreign worker status (A201: yes, A202: no)",
        examples=["A201"]
    )

class ExplanationMetric(BaseModel):
    feature: str
    contribution: float
    direction: str  # "increases_risk" or "decreases_risk"

class PredictionResponse(BaseModel):
    request_id: str
    prediction_id: Optional[str] = None
    prediction: int = Field(..., description="0 for Non-default (Good), 1 for Default (Bad)")
    default_probability: float = Field(..., description="Estimated probability of loan default [0.0, 1.0]")
    risk_level: str = Field(..., description="Categorical risk tier: Low, Medium, or High")
    risk_tier: Optional[str] = None
    decision_threshold: float
    threshold_used: Optional[float] = None
    model_version: str
    latency_ms: float
    top_risk_factors: List[ExplanationMetric]

class BatchPredictionRequest(BaseModel):
    applications: List[BorrowerApplication]

class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]
    total_latency_ms: float
