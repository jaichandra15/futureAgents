"""
SQLAlchemy async session management and database initialization.

Handles connection pooling, pgvector extension setup, and table creation.
"""

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.database.models import Base

# Database connection URL from environment or default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://ai_admin:secure_password_here@postgres:5432/lightning_rag",
)

# Create async engine with connection pooling
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL debugging
    future=True,
    pool_pre_ping=True,  # Verify connections before using them
    poolclass=NullPool,  # Use NullPool to avoid connection issues in containers
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db() -> None:
    """
    Initialize the database.

    Ensures pgvector extension is enabled, creates all tables from models.
    Called on application startup via lifespan context.
    """
    async with engine.begin() as conn:
        # Enable pgvector extension for vector similarity search
        await conn.execute(__import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # Create all tables from SQLAlchemy models
        await conn.run_sync(Base.metadata.create_all)

    print("✓ Database initialized with pgvector extension")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to inject async database sessions.

    Usage in route:
        @app.get("/")
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...

    Yields:
        AsyncSession: Database session for the request
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
