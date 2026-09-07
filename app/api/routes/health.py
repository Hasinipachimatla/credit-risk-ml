"""
Deep health check endpoint for CreditRiskML.
Verifies API server, database connectivity, and active model status.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import settings
from app.logging_config import logger
from app.database.session import get_db
from app.services.model_service import model_service
from app.api.schemas.monitoring import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def get_health(db: Session = Depends(get_db)):
    """
    Performs deep system health check including database ping and model verification.
    """
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")

    if not model_service.is_loaded():
        model_service.load_active_model()

    model_loaded = model_service.is_loaded()
    overall_status = "healthy" if (db_connected and model_loaded) else "degraded"

    return HealthResponse(
        status=overall_status,
        app_env=settings.APP_ENV,
        model_loaded=model_loaded,
        model_version=model_service.model_version,
        database_connected=db_connected,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
