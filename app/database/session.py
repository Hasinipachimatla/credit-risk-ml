"""
Database session and connection management for CreditRiskML.
Uses SQLAlchemy 2.0 with connection pooling and graceful SQLite fallback.
"""
import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from app.config import settings

logger = logging.getLogger("credit_risk_ml.database")

Base = declarative_base()

def get_engine():
    """Builds SQLAlchemy engine with appropriate dialect arguments."""
    db_url = settings.DATABASE_URL

    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
        engine = create_engine(db_url, connect_args=connect_args)
    else:
        # PostgreSQL with connection pooling
        engine = create_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )

    return engine

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes tables in database."""
    try:
        from app.database import models  # Ensure models are imported
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.warning(f"Database initialization encountered an issue: {e}")

def get_db() -> Generator[Session, None, None]:
    """Dependency for providing request-scoped database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
