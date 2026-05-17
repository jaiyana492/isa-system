"""
config/redis.py
CorePilora AI — Redis Connection & Client

REDIS_URL pulled from settings — not os.getenv.
Single config pattern across entire project.
"""

from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, TimeoutError

from config.settings import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TTL CONSTANTS — seconds
# ─────────────────────────────────────────────────────────────────────────────

TTL_SESSION     = 3600        # 1 hour   — active call session
TTL_LEAD_CACHE  = 1800        # 30 mins  — lead context cache
TTL_LEAD_LOCK   = 300         # 5 mins   — covers full LangGraph pipeline execution
TTL_RATE_LIMIT  = 60          # 1 min    — rate limit window
TTL_NURTURE_JOB = 86400 * 30  # 30 days  — nurture queue job
TTL_LPMAMA      = 3600        # 1 hour   — LPMAMA state
TTL_CONV_STATE  = 3600        # 1 hour   — conversation state


# ─────────────────────────────────────────────────────────────────────────────
# REDIS CLIENT — SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_redis_client: Optional[Redis] = None


async def get_redis() -> Redis:
    """
    Return Redis client singleton.
    Creates connection on first call.
    Reuses on all subsequent calls.
    """
    global _redis_client

    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        logger.info(
            "REDIS | Client initialized | url=%s",
            settings.REDIS_URL,
        )

    return _redis_client


async def close_redis() -> None:
    """
    Close Redis connection cleanly on shutdown.
    Called from main.py lifespan handler.
    """
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("REDIS | Connection closed")


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

async def redis_health_check() -> bool:
    """
    Ping Redis. Returns True if healthy. False if down.
    """
    try:
        client = await get_redis()
        await client.ping()
        return True
    except (ConnectionError, TimeoutError) as e:
        logger.error("REDIS | Health check failed | error=%s", str(e))
        return False


# ─────────────────────────────────────────────────────────────────────────────
# KEY BUILDERS
# Never construct Redis keys inline anywhere in the codebase.
# Always use these functions.
# ─────────────────────────────────────────────────────────────────────────────

def key_session(lead_id: str) -> str:
    return f"session:{lead_id}"


def key_lead_cache(lead_id: str) -> str:
    return f"lead:cache:{lead_id}"


def key_lead_lock(phone: str) -> str:
    return f"lead:lock:{phone}"


def key_lpmama(lead_id: str) -> str:
    return f"lpmama:{lead_id}"


def key_nurture_queue(lead_id: str) -> str:
    return f"nurture:queue:{lead_id}"


def key_rate_limit(phone: str) -> str:
    return f"rate:call:{phone}"


def key_conversation_state(session_id: str) -> str:
    return f"conv:state:{session_id}"