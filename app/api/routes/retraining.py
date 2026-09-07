"""
API route to trigger automated retraining and Champion-Challenger evaluation.
"""
from fastapi import APIRouter, HTTPException
from app.api.schemas.monitoring import RetrainResponse
from src.monitoring.retraining import execute_retraining_pipeline
from app.services.model_service import model_service
from app.logging_config import logger

router = APIRouter()

@router.post("/retrain", response_model=RetrainResponse)
async def trigger_retraining():
    """
    Triggers the end-to-end retraining workflow. Trains Challenger, evaluates
    on validation split, tests against Champion promotion criteria, and promotes if passed.
    """
    try:
        logger.info("Retraining endpoint invoked.")
        result = execute_retraining_pipeline(trigger_reason="api_request")

        # If a new model was promoted, reload into service cache
        if result["status"] == "promoted":
            model_service.load_active_model()
            logger.info("Active Champion reloaded into memory cache.")

        return RetrainResponse(
            status=result["status"],
            message=result["message"],
            champion_version=result.get("champion_version"),
            challenger_metrics=result.get("challenger_metrics"),
            previous_champion_metrics=result.get("previous_champion_metrics") or result.get("champion_metrics"),
            gate_reasons=result.get("gate_reasons", [])
        )
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining execution error: {str(e)}")
