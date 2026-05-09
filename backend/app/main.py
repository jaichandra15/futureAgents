"""
FastAPI application entry point for Multimodal Offline RAG backend.

Initializes FastAPI with CORS configuration, database setup, and routers.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router
from app.database.session import init_db

# ─── Lifespan Context ──────────────────────────────────────────────────────
# Runs on startup and shutdown


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup/shutdown.

    Startup: Initialize database with pgvector extension.
    Shutdown: Clean up resources.
    """
    # Startup event
    print("🚀 Starting Multimodal Offline RAG backend...")
    await init_db()
    print("✓ Backend ready")

    yield

    # Shutdown event
    print("🛑 Shutting down backend...")


# Initialize FastAPI application with lifespan
app = FastAPI(
    title="Multimodal Offline RAG API",
    description="Chat interface with RAG and Text-to-SQL capabilities",
    version="0.1.0",
    lifespan=lifespan,
)

# ─── CORS Configuration ─────────────────────────────────────────────────────
# Allow requests from the Next.js frontend running on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Also support Vite dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Include Routers ────────────────────────────────────────────────────────
app.include_router(router)


@app.get("/health")
async def health_check():
    """
    Health check endpoint for container orchestration.

    Returns:
        dict: Status indicator for readiness probes
    """
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "message": "Multimodal Offline RAG API",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
