"""
CreditRiskML — FastAPI Serving & Orchestration Application.
Production-grade credit risk scoring and MLOps monitoring platform.
"""
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.logging_config import logger
from app.database.session import init_db
from app.services.model_service import model_service
from app.api.routes import health, prediction, monitoring, retraining

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    logger.info("Initializing CreditRiskML application...")

    # 1. Initialize database schema
    init_db()

    # 2. Preload active Champion model into memory cache
    loaded = model_service.load_active_model()
    if loaded:
        logger.info(f"Champion model successfully preloaded (Version: {model_service.model_version})")
    else:
        logger.warning("No trained model found on startup. Train a model to enable predictions.")

    yield
    logger.info("Shutting down CreditRiskML application...")

app = FastAPI(
    title="CreditRiskML API",
    description="Production-grade Credit Risk Machine Learning & MLOps Platform API",
    version="1.0.0",
    lifespan=lifespan
)

# 1. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Request ID & Latency Measurement Middleware
@app.middleware("http")
async def add_process_time_and_request_id(request: Request, call_next):
    start_time = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    response: Response = await call_next(request)

    process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Process-Time-MS"] = str(process_time_ms)
    response.headers["X-Request-ID"] = request_id
    return response

# 3. Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during request processing."
        }
    )
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
REPORTS_DIR = Path("reports")

# 4. Mount Static Files & Reports Assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if REPORTS_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

# 5. Register Routers
app.include_router(health.router, tags=["Health"])
app.include_router(prediction.router, tags=["Prediction"])
app.include_router(monitoring.router, tags=["Monitoring"])
app.include_router(retraining.router, tags=["Retraining"])

@app.get("/")
async def read_root(request: Request):
    accept = request.headers.get("accept", "")
    if accept.startswith("text/html"):
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    logger.info("Root endpoint accessed")
    return {
        "name": "CreditRiskML Platform API",
        "version": "1.0.0",
        "description": "Production-grade ML platform for credit risk modeling and MLOps.",
        "dashboard": "/dashboard",
        "documentation": "/docs",
        "active_model": model_service.model_name if model_service.is_loaded() else "None",
        "model_version": model_service.model_version
    }

@app.get("/dashboard", response_class=FileResponse)
async def get_dashboard():
    index_file = STATIC_DIR / "index.html"
    return FileResponse(index_file)

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.APP_ENV == "development"
    )
