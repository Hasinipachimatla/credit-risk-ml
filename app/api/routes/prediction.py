"""
Credit risk prediction API routes for single and batch borrower inference.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from app.api.schemas.prediction import (
    BorrowerApplication,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from app.services.prediction_service import PredictionService, async_log_prediction
from app.services.metrics_service import metrics_tracker
from app.logging_config import logger

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_single_borrower(
    application: BorrowerApplication,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Submits borrower credit application data to predict default probability
    and assign a risk rating (Low, Medium, High).
    """
    input_dict = application.model_dump()
    req_id = request.headers.get("X-Request-ID")

    try:
        result, latency_ms = PredictionService.predict_single(input_dict, request_id=req_id)

        # Record operational metrics
        metrics_tracker.record_request(latency_ms=latency_ms, success=True)

        # Asynchronously log prediction to PostgreSQL / database
        background_tasks.add_task(
            async_log_prediction,
            request_id=result["request_id"],
            model_version=result["model_version"],
            prediction=result["prediction"],
            default_probability=result["default_probability"],
            risk_level=result["risk_level"],
            latency_ms=latency_ms,
            input_dict=input_dict
        )

        return result
    except Exception as e:
        metrics_tracker.record_request(latency_ms=0.0, success=False)
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_borrowers(
    batch_request: BatchPredictionRequest
):
    """
    Vectorized batch inference endpoint for multiple borrower credit applications.
    """
    apps = [app.model_dump() for app in batch_request.applications]
    if not apps:
        raise HTTPException(status_code=400, detail="Empty applications batch provided.")

    try:
        results, total_latency_ms = PredictionService.predict_batch(apps)
        metrics_tracker.record_request(latency_ms=total_latency_ms, success=True)

        return BatchPredictionResponse(
            count=len(results),
            predictions=results,
            total_latency_ms=total_latency_ms
        )
    except Exception as e:
        metrics_tracker.record_request(latency_ms=0.0, success=False)
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")
