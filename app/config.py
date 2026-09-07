from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Application Environment
    APP_ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Database Configuration (No hardcoded credentials; defaults to local SQLite fallback)
    DATABASE_URL: str = "sqlite:///./credit_risk.db"

    # MLflow Tracking
    MLFLOW_TRACKING_URI: str = "sqlite:///./mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "CreditRiskML"

    # Model Configuration
    MODEL_NAME: str = "CreditRiskModel"
    MODEL_VERSION: str = "v1"
    MODEL_DIR: str = "models"
    DATA_DIR: str = "data"

    # Drift Detection Thresholds
    DRIFT_PSI_THRESHOLD: float = 0.25
    DRIFT_KS_PVALUE_THRESHOLD: float = 0.05

    # Model Quality Promotion Gates
    MIN_ROC_AUC: float = 0.70
    MIN_F1: float = 0.50
    MIN_RECALL: float = 0.50
    PROMOTION_MARGIN: float = 0.01

    # Decision Threshold for Default Classification
    DECISION_THRESHOLD: float = 0.50

    # CORS
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])

    # Backward compatibility alias
    @property
    def ENVIRONMENT(self) -> str:
        return self.APP_ENV

    @property
    def MODEL_PATH(self) -> str:
        return f"{self.MODEL_DIR}/credit_risk_model.joblib"

    # Load from .env file if available
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
