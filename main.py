"""
main.py
CorePilora AI — Application Entry Point

FastAPI application with:
- Lifespan handler (startup/shutdown)
- Middleware stack (CORS, request logging)
- Route registration
- Database table creation
- Redis connection management
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from config.database import create_tables
from config.redis import close_redis
from services.audio_cache import close_binary_redis
from services.nurture_runner import run_nurture_loop
from api.v1.router import v1_router

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — STARTUP / SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
    - Create database tables
    - Log system ready

    Shutdown:
    - Close Redis text client
    - Close Redis binary client (audio cache)
    """
    # ── Startup ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("CorePilora AI ISA — Starting")
    logger.info("ISA Name: %s", settings.ISA_NAME)
    logger.info("Environment: %s", settings.APP_ENV)
    logger.info("Primary Market: %s", settings.PRIMARY_MARKET)
    logger.info("=" * 60)

    await create_tables()
    logger.info("DATABASE | Tables created")

    try:
        from services.audio_cache import warm_up_cache
        cached = await warm_up_cache()
        logger.info("AUDIO CACHE | Warm up complete | cached=%s", cached)
    except Exception as _e:
        logger.warning("AUDIO CACHE | Warm up skipped | error=%s", str(_e))

    nurture_task = asyncio.create_task(run_nurture_loop())
    logger.info("NURTURE RUNNER | Background task started")

    logger.info("SYSTEM | CorePilora AI is LIVE")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    nurture_task.cancel()
    try:
        await nurture_task
    except asyncio.CancelledError:
        pass
    await close_redis()
    await close_binary_redis()
    logger.info("SYSTEM | CorePilora AI shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# APP INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CorePilora AI ISA",
    description="Autonomous Real Estate Lead Intelligent System",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# MIDDLEWARE — CORS
# ─────────────────────────────────────────────────────────────────────────────

_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# MIDDLEWARE — REQUEST LOGGING
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start_time = time.time()
    method     = request.method
    path       = request.url.path

    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(
            "REQUEST | %s %s | ERROR | %s",
            method, path, str(e),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    duration_ms = round((time.time() - start_time) * 1000, 2)

    logger.info(
        "REQUEST | %s %s | %s | %sms",
        method,
        path,
        response.status_code,
        duration_ms,
    )

    return response


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER ROUTES
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(v1_router)


# ─────────────────────────────────────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "CorePilora AI ISA",
        "isa_name": settings.ISA_NAME,
        "status": "operational",
        "version": "1.0.0",
    }