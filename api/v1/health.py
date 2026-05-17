"""
api/v1/health.py
CorePilora AI — Health Monitoring Endpoints

System health checks:
- API alive
- Database connection
- Redis connection
- Groq API reachability
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter
from sqlalchemy import text as sa_text

from config.settings import settings
from config.redis import redis_health_check
from config.database import engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


# ─────────────────────────────────────────────────────────────────────────────
# BASIC HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/")
async def health_check():
    """Basic health check. Returns 200 if API is running."""
    return {
        "status": "healthy",
        "service": "CorePilora AI ISA",
        "isa_name": settings.ISA_NAME,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEEP HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/deep")
async def deep_health_check():
    """
    Deep health check — verifies all system dependencies.
    Returns status of each component.
    """
    results = {
        "api":      "healthy",
        "database": "unknown",
        "redis":    "unknown",
        "groq":     "unknown",
    }

    # Database
    try:
        async with engine.connect() as conn:
            await conn.execute(sa_text("SELECT 1"))
        results["database"] = "healthy"
    except Exception as e:
        results["database"] = f"unhealthy: {str(e)[:100]}"
        logger.error("HEALTH | Database check failed | error=%s", str(e))

    # Redis
    try:
        redis_ok = await redis_health_check()
        results["redis"] = "healthy" if redis_ok else "unhealthy"
    except Exception as e:
        results["redis"] = f"unhealthy: {str(e)[:100]}"
        logger.error("HEALTH | Redis check failed | error=%s", str(e))

    # Groq
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                },
            )
            if response.status_code == 200:
                results["groq"] = "healthy"
            else:
                results["groq"] = f"unhealthy: status {response.status_code}"
    except Exception as e:
        results["groq"] = f"unhealthy: {str(e)[:100]}"
        logger.error("HEALTH | Groq check failed | error=%s", str(e))

    all_healthy = all(v == "healthy" for v in results.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": results,
    }