"""
api/v1/router.py
CorePilora AI — Main API Router

Registers all v1 routes:
- Webhook endpoints (lead intake)
- Voice endpoints (Twilio inbound/outbound)
- Health endpoints
"""

from __future__ import annotations

from fastapi import APIRouter

from api.v1.webhook import router as webhook_router
from api.v1.health import router as health_router
from api.v1.voice import router as voice_router

# ─────────────────────────────────────────────────────────────────────────────
# MAIN V1 ROUTER
# ─────────────────────────────────────────────────────────────────────────────

v1_router = APIRouter(prefix="/api/v1")

# Register sub-routers
v1_router.include_router(webhook_router)
v1_router.include_router(health_router)
v1_router.include_router(voice_router)