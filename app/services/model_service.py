"""
In-memory model cache service for CreditRiskML.
Loads Champion model, fitted preprocessor, and metadata into memory on startup.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import joblib

from app.config import settings
from src.models.mlflow_utils import load_model_registry

logger = logging.getLogger("credit_risk_ml.model_service")

class ModelService:
    """Thread-safe singleton managing the active Champion model and preprocessor."""
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.registry_entry = None
        self.threshold: float = settings.DECISION_THRESHOLD
        self.model_version: str = settings.MODEL_VERSION
        self.model_name: str = settings.MODEL_NAME

    def load_active_model(self) -> bool:
        """Loads or reloads the active Champion model and preprocessor from disk."""
        models_dir = Path(settings.MODEL_DIR)
        model_path = models_dir / "credit_risk_model.joblib"
        preprocessor_path = models_dir / "preprocessor.joblib"

        if not model_path.exists() or not preprocessor_path.exists():
            logger.warning("Active model or preprocessor artifact not found on disk.")
            return False

        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)

            registry = load_model_registry()
            champ = registry.get("champion")
            if champ:
                self.registry_entry = champ
                self.threshold = champ.get("threshold", settings.DECISION_THRESHOLD)
                self.model_version = champ.get("model_version", settings.MODEL_VERSION)
                self.model_name = champ.get("model_name", settings.MODEL_NAME)

            logger.info(f"Loaded Champion model '{self.model_name}' ({self.model_version}) into memory. Threshold = {self.threshold:.3f}")
            return True
        except Exception as e:
            logger.error(f"Failed to load active model: {e}")
            return False

    def is_loaded(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        return self.registry_entry

model_service = ModelService()
