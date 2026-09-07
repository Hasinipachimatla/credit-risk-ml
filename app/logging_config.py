import logging
import sys
from app.config import settings

def setup_logging():
    # Setup basic logging configuration
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Force re-configuration of basicConfig
    )

    # Minimize noise from third-party logs
    logging.getLogger("uvicorn.error").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger("credit_risk_ml")
    logger.info(f"Logging configured at level {settings.LOG_LEVEL}")
    return logger

# Initialize logger instance
logger = setup_logging()
